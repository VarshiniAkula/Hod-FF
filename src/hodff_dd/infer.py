import argparse
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

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from .utils.common import load_yaml, ensure_dir, majority_vote
from .models.hodff_dd import build_hodff_dd
from .data.dataset import build_index, split_index, _load_sequence

def sequences_for_video(video_dir, img_size, seq_len, max_sequences):
    frames = sorted(Path(video_dir).glob("*.jpg"))
    if len(frames) == 0: return []
    # same sampling as training
    step = max(1, (len(frames) - seq_len) // max(1, max_sequences))
    starts = list(range(0, max(1, len(frames) - seq_len + 1), step))
    if len(starts) > max_sequences:
        starts = starts[:max_sequences]
    seqs = []
    for s in starts:
        seqs.append(frames[s:s+seq_len])
    return seqs

def main(cfg_path, weights_path):
    cfg = load_yaml(cfg_path)
    data_root = Path(cfg["data_root"])
    proc_root = data_root / cfg["processed_dir"]
    out_pred = Path(cfg["predictions_dir"])
    ensure_dir(out_pred)

    seq_len = cfg["sequence_length"]
    img_size = cfg["face_size"]
    index = build_index(proc_root, cfg["classes"], seq_len, cfg["max_sequences_per_video"])
    train_idx, val_idx, test_idx = split_index(index, cfg["val_split"], cfg["test_split"], cfg["seed"])

    # Build and load model
    model = build_hodff_dd(seq_len=seq_len, img_size=img_size, freeze_backbones=True)
    model.load_weights(weights_path)

    # Group test sequences by video
    test_items = [index[i] for i in test_idx]
    videos = {}
    for it in test_items:
        key = (it["class"], it["video"])
        videos.setdefault(key, []).append(it)

    rows = []
    for (cls, vid), items in videos.items():
        all_frame_scores = []
        for it in items:
            frames = [Path(p) for p in it["frames"]]
            X = _load_sequence(frames, img_size, seq_len)[None, ...]  # (1, T, H, W, 3)
            preds = model.predict(X, verbose=0)[0]                    # (T, 2)
            all_frame_scores.append(preds)
        all_frame_scores = np.concatenate(all_frame_scores, axis=0)    # (N_frames_over_seqs, 2)
        pred_label, counts = majority_vote(all_frame_scores)
        rows.append({
            "video": vid,
            "true_class": cls,
            "predicted": "real" if pred_label == 0 else "fake",
            "frames_real_votes": counts["real"],
            "frames_fake_votes": counts["fake"]
        })

    df = pd.DataFrame(rows).sort_values(by=["true_class", "video"])
    out_csv = out_pred / "predictions.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved predictions: {out_csv}")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/default.yaml")
    ap.add_argument("--weights", type=str, default="outputs/checkpoints/best.keras")
    args = ap.parse_args()
    main(args.config, args.weights)
