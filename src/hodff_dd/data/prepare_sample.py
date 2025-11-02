"""
Prepare a small, balanced sample (REAL/FAKE) from a full deepfake dataset.

Supported sources (auto-detected in this order):
  1) Kaggle DFDC (metadata.json alongside videos)
  2) FaceForensics++ (original_sequences / manipulated_sequences, quality=c23 by default)
  3) CSV metadata (labels.csv/metadata.csv with label column)
  4) Folder/name heuristics (paths containing 'real|pristine|original' vs 'fake|manipulated')

Outputs:
  data/raw/real/*.mp4
  data/raw/fake/*.mp4
plus an index at outputs/sample_index.csv

Usage examples:
  # Kaggle DFDC
  python -m src.hodff_dd.data.prepare_sample --source /path/to/DFDC --num-per-class 5

  # FaceForensics++ (use c23 by default, as in the paper)
  python -m src.hodff_dd.data.prepare_sample --source /path/to/FaceForensics++ --num-per-class 5 --ffpp-quality c23

  # Generic folder with 'real' and 'fake' dirs inside
  python -m src.hodff_dd.data.prepare_sample --source /path/to/full_dataset --num-per-class 5

Notes:
  - Default copy mode is "copy". You can use --mode symlink or --mode link (hard link).
  - On Windows, symlinks may require admin/developer mode; fallback to copy if it fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..utils.common import ensure_dir, set_seed


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
REAL_HINTS = re.compile(r"(?:^|/)(?:real|pristine|original)s?(?:/|$)", re.IGNORECASE)
FAKE_HINTS = re.compile(r"(?:^|/)(?:fake|manipulated|synth|synthesis|synthetic)s?(?:/|$)", re.IGNORECASE)


FFPP_SUBSETS = ["Deepfakes", "FaceSwap", "Face2Face", "FaceShifter", "NeuralTextures"]


@dataclass
class SamplePlan:
    real: List[Path]
    fake: List[Path]
    dataset_type: str


def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def unique_dest(dest_dir: Path, base_name: str) -> Path:
    """Avoid overwriting by appending -1, -2, ... if needed."""
    out = dest_dir / base_name
    if not out.exists():
        return out
    stem, ext = os.path.splitext(base_name)
    k = 1
    while True:
        cand = dest_dir / f"{stem}-{k}{ext}"
        if not cand.exists():
            return cand
        k += 1


def copy_like(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "link":
        try:
            os.link(src, dst)   # hard link
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------- Collectors for specific datasets ----------

def collect_kaggle_dfdc(root: Path) -> Optional[Tuple[List[Path], List[Path]]]:
    """
    Kaggle DFDC has a metadata.json mapping filename -> {label: 'REAL'|'FAKE', ...}
    Videos typically sit in the same folder as metadata.json.
    We will search all metadata.json files under root.
    """
    meta_files = list(root.rglob("metadata.json"))
    if not meta_files:
        return None

    real, fake = [], []
    for mf in meta_files:
        try:
            data = json.loads(mf.read_text())
        except Exception:
            continue
        base = mf.parent
        for fname, info in data.items():
            label = str(info.get("label", "")).lower()
            vp = base / fname
            if not vp.exists():
                # try to locate by name anywhere under base (slow but limited)
                hits = list(base.rglob(fname))
                if hits:
                    vp = hits[0]
            if not vp.exists() or not is_video(vp):
                continue
            if label == "real":
                real.append(vp)
            elif label == "fake":
                fake.append(vp)
    if not real and not fake:
        return None
    return real, fake

def collect_celebdf_v2(root: Path) -> Optional[Tuple[List[Path], List[Path]]]:
    """
    Celeb-DF (v2) typical layout:
      Celeb-real/         -> REAL
      YouTube-real/       -> REAL (optional presence)
      Celeb-synthesis/    -> FAKE
    We gather videos recursively from these folders if they exist.
    """
    real_dirs = [root / "Celeb-real", root / "YouTube-real"]
    fake_dirs = [root / "Celeb-synthesis"]

    has_any = any(d.exists() for d in real_dirs + fake_dirs)
    if not has_any:
        return None

    real, fake = [], []
    for d in real_dirs:
        if d.exists():
            real.extend(p for p in d.rglob("*") if p.is_file() and is_video(p))
    for d in fake_dirs:
        if d.exists():
            fake.extend(p for p in d.rglob("*") if p.is_file() and is_video(p))

    # If nothing discovered, fall back to None so other detectors can try.
    if not real and not fake:
        return None
    return real, fake

def collect_ffpp(root: Path, quality: str = "c23") -> Optional[Tuple[List[Path], List[Path]]]:
    """
    FaceForensics++ layout (commonly):
      original_sequences/youtube/{c23|c40|raw}/videos/*.mp4   -> REAL
      manipulated_sequences/{subset}/{c23|c40|raw}/videos/*.mp4 -> FAKE
    """
    orig_root = root / "original_sequences"
    mani_root = root / "manipulated_sequences"
    if not orig_root.exists() or not mani_root.exists():
        return None

    real_dir = orig_root / "youtube" / quality / "videos"
    real = [p for p in real_dir.glob("*.mp4")] if real_dir.exists() else []

    fake: List[Path] = []
    for subset in FFPP_SUBSETS:
        vdir = mani_root / subset / quality / "videos"
        if vdir.exists():
            fake.extend(p for p in vdir.glob("*.mp4"))

    if not real and not fake:
        return None
    return real, fake


def collect_csv_labeled(root: Path) -> Optional[Tuple[List[Path], List[Path]]]:
    """
    Look for labels in a CSV file. Expected columns:
      - one of: label, class, target   (values 'real'/'fake' or 0/1)
      - and one of: path, filepath, video, filename, file
    Paths are assumed relative to the CSV file's directory.
    """
    csv_candidates = list(root.rglob("*.csv"))
    if not csv_candidates:
        return None

    real, fake = [], []
    for cf in csv_candidates:
        try:
            with cf.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                cols = {c.lower(): c for c in reader.fieldnames or []}
                label_col = next((cols[c] for c in ["label", "class", "target"] if c in cols), None)
                path_col = next((cols[c] for c in ["path", "filepath", "video", "filename", "file"] if c in cols), None)
                if not (label_col and path_col):
                    continue
                for row in reader:
                    rel = row[path_col].strip()
                    lab = str(row[label_col]).strip().lower()
                    vp = (cf.parent / rel).resolve()
                    if not vp.exists() or not is_video(vp):
                        continue
                    if lab in {"real", "0"}:
                        real.append(vp)
                    elif lab in {"fake", "1"}:
                        fake.append(vp)
        except Exception:
            continue

    if not real and not fake:
        return None
    return real, fake


def collect_by_dirnames(root: Path) -> Optional[Tuple[List[Path], List[Path]]]:
    """Heuristic: classify by directory name hints in the path."""
    all_videos = [p for p in root.rglob("*") if p.is_file() and is_video(p)]
    if not all_videos:
        return None

    real, fake = [], []
    for vp in all_videos:
        s = str(vp.as_posix())
        if REAL_HINTS.search(s):
            real.append(vp)
        elif FAKE_HINTS.search(s):
            fake.append(vp)

    # If nothing matched, try looser heuristic: any video is candidate; we cannot label.
    if not real and not fake:
        return None
    return real, fake


def collect_candidates(source: Path, ffpp_quality: str = "c23") -> SamplePlan:
    # 1) Kaggle DFDC
    k = collect_kaggle_dfdc(source)
    if k:
        return SamplePlan(real=k[0], fake=k[1], dataset_type="kaggle_dfdc")

    # 2) FaceForensics++
    f = collect_ffpp(source, quality=ffpp_quality)
    if f:
        return SamplePlan(real=f[0], fake=f[1], dataset_type=f"ffpp_{ffpp_quality}")

    # 3) Celeb-DF (v2)
    cdf = collect_celebdf_v2(source)
    if cdf:
        return SamplePlan(real=cdf[0], fake=cdf[1], dataset_type="celebdf_v2")

    # 4) CSV-labeled (generic)
    c = collect_csv_labeled(source)
    if c:
        return SamplePlan(real=c[0], fake=c[1], dataset_type="csv_labeled")

    # 5) Directory/name hints (generic)
    d = collect_by_dirnames(source)
    if d:
        return SamplePlan(real=d[0], fake=d[1], dataset_type="dir_hints")

    return SamplePlan(real=[], fake=[], dataset_type="unknown")



# ---------- Sampling & writing ----------

def balanced_sample(real: List[Path], fake: List[Path], k: int, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    real_shuf = real[:]
    fake_shuf = fake[:]
    rng.shuffle(real_shuf)
    rng.shuffle(fake_shuf)
    k_real = min(k, len(real_shuf))
    k_fake = min(k, len(fake_shuf))
    return real_shuf[:k_real], fake_shuf[:k_fake]


def write_index(rows: List[Dict[str, str]], index_csv: Path) -> None:
    ensure_dir(index_csv.parent)
    import pandas as pd  # optional dep already in requirements
    import io
    cols = ["class", "src", "dst", "dataset"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(index_csv, index=False)
    print(f"[INFO] Wrote index: {index_csv}")
    # Also keep a JSON for convenience
    idx_json = index_csv.with_suffix(".json")
    idx_json.write_text(json.dumps([dict(r) for r in rows], indent=2))
    print(f"[INFO] Wrote JSON index: {idx_json}")


def main():
    ap = argparse.ArgumentParser(description="Create a small, balanced sample dataset (REAL/FAKE) under data/raw/")
    ap.add_argument("--source", type=str, required=True, help="Path to the full dataset root")
    ap.add_argument("--dest-root", type=str, default="data/raw", help="Destination root (default: data/raw)")
    ap.add_argument("--num-per-class", type=int, default=5, help="Target #videos per class (default: 5)")
    ap.add_argument("--mode", type=str, choices=["copy", "symlink", "link"], default="copy", help="Copy strategy")
    ap.add_argument("--ffpp-quality", type=str, default="c23", choices=["c23", "c40", "raw"], help="FF++ quality tier")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Print what would be done, without writing files.")
    args = ap.parse_args()

    set_seed(args.seed)
    source = Path(args.source).resolve()
    dest_root = Path(args.dest_root)
    dest_real = dest_root / "real"
    dest_fake = dest_root / "fake"
    ensure_dir(dest_real)
    ensure_dir(dest_fake)

    plan = collect_candidates(source, ffpp_quality=args.ffpp_quality)
    print(f"[INFO] Detected dataset type: {plan.dataset_type}")
    print(f"[INFO] Found {len(plan.real)} REAL candidates, {len(plan.fake)} FAKE candidates")

    if not plan.real or not plan.fake:
        raise SystemExit("Not enough labeled candidates discovered. Please verify --source path/structure.")

    chosen_real, chosen_fake = balanced_sample(plan.real, plan.fake, args.num_per_class, seed=args.seed)
    print(f"[INFO] Sampling up to {args.num_per_class} per class "
          f"-> REAL: {len(chosen_real)}  FAKE: {len(chosen_fake)}")

    rows: List[Dict[str, str]] = []
    if args.dry_run:
        for s in chosen_real:
            print(f"[DRY] real <- {s}")
        for s in chosen_fake:
            print(f"[DRY] fake <- {s}")
        return

    for s in chosen_real:
        dst = unique_dest(dest_real, s.name)
        copy_like(s, dst, args.mode)
        rows.append({"class": "real", "src": str(s), "dst": str(dst), "dataset": plan.dataset_type})

    for s in chosen_fake:
        dst = unique_dest(dest_fake, s.name)
        copy_like(s, dst, args.mode)
        rows.append({"class": "fake", "src": str(s), "dst": str(dst), "dataset": plan.dataset_type})

    write_index(rows, Path("outputs") / "sample_index.csv")
    print("[INFO] Done. You can now run preprocessing/training on the sampled data.")

if __name__ == "__main__":
    main()
