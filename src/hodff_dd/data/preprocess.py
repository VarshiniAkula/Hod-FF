import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
try:
    import cv2
    cv2.setNumThreads(0)
except Exception:
    pass

import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
from mtcnn import MTCNN
import numpy as np
from ..utils.common import load_yaml, ensure_dir

def _crop_with_margin(img, box, margin=20):
    x, y, w, h = box
    H, W = img.shape[:2]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(W, x + w + margin)
    y1 = min(H, y + h + margin)
    return img[y0:y1, x0:x1]

def extract_faces_from_video(video_path, out_dir, every_n=10, margin=20, face_size=299):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    det = MTCNN()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] cannot open {video_path}")
        return 0
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % every_n == 0:
            # Convert BGR to RGB for detector, then back when cropping
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = det.detect_faces(rgb)
            if faces:
                # pick the largest face
                best = max(faces, key=lambda d: d['box'][2] * d['box'][3])
                crop = _crop_with_margin(rgb, best["box"], margin=margin)
                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                face = cv2.resize(crop, (face_size, face_size), interpolation=cv2.INTER_AREA)
                out_path = out_dir / f"frame_{idx:06d}.jpg"
                cv2.imwrite(str(out_path), face)
                saved += 1
        idx += 1
    cap.release()
    return saved

def main(cfg_path):
    cfg = load_yaml(cfg_path)
    root = Path(cfg["data_root"])
    raw_root = root / cfg["raw_dir"]
    proc_root = root / cfg["processed_dir"]
    ensure_dir(proc_root)

    total_saved = 0
    for cls in cfg["classes"]:
        in_dir = raw_root / cls
        out_dir_cls = proc_root / cls
        ensure_dir(out_dir_cls)
        videos = sorted([p for p in in_dir.glob("*.mp4")])
        for vp in videos:
            out_video_dir = out_dir_cls / vp.stem
            ensure_dir(out_video_dir)
            saved = extract_faces_from_video(
                vp, out_video_dir,
                every_n=cfg["frame_sample_rate"],
                margin=cfg["mtcnn_margin"],
                face_size=cfg["face_size"]
            )
            print(f"[{cls}] {vp.name}: saved {saved} faces -> {out_video_dir}")
            total_saved += saved
    print(f"Done. Total face frames saved: {total_saved}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/default.yaml")
    args = ap.parse_args()
    main(args.config)
