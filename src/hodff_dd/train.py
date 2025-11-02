# src/hodff_dd/train.py
import os
# Keep native libs single-threaded for stability on macOS/CPU
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
try:
    import cv2  # type: ignore
    cv2.setNumThreads(0)
except Exception:
    pass

import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from .utils.common import load_yaml, ensure_dir, set_seed
from .models.hodff_dd import build_hodff_dd
from .data.dataset import build_index, split_index, SequenceDataset
from .optimizers.spotted_hyena import sho_initialize_final_head


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    set_seed(cfg.get("seed", 42))

    # --- Paths ---
    data_root = Path(cfg["data_root"])
    proc_root = data_root / cfg["processed_dir"]
    out_ckpt = Path(cfg.get("checkpoints_dir", "outputs/checkpoints"))
    out_logs = Path(cfg.get("logs_dir", "outputs/logs"))
    ensure_dir(out_ckpt)
    ensure_dir(out_logs)

    # --- Build dataset indices ---
    seq_len = int(cfg.get("sequence_length", 8))
    img_size = int(cfg.get("face_size", 299))
    index = build_index(
        proc_root,
        cfg.get("classes", ["real", "fake"]),
        seq_len,
        int(cfg.get("max_sequences_per_video", 4)),
    )
    if len(index) < 4:
        raise RuntimeError(
            "Not enough sequences found. Try lowering sequence_length, "
            "increasing max_sequences_per_video, or preprocessing more videos."
        )

    train_idx, val_idx, test_idx = split_index(
        index,
        float(cfg.get("val_split", 0.2)),
        float(cfg.get("test_split", 0.2)),
        int(cfg.get("seed", 42)),
    )
    train_ds = SequenceDataset(
        index, train_idx, img_size, seq_len,
        int(cfg.get("batch_size", 2)),
        shuffle=True
    )
    val_ds = SequenceDataset(
        index, val_idx, img_size, seq_len,
        int(cfg.get("batch_size", 2)),
        shuffle=False
    )

    # --- Build model (this is the line that went missing) ---
    model = build_hodff_dd(
        seq_len=seq_len,
        img_size=img_size,
        freeze_backbones=bool(cfg.get("freeze_backbones", True)),
    )
    model.summary()

    # --- Compile (cast LR to float for Keras 3) ---
    lr = cfg.get("learning_rate", 1e-5)
    try:
        lr = float(lr)
    except Exception as e:
        raise ValueError(f"learning_rate must be numeric; got {lr!r}") from e

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # --- SHO initialization of final head (quick pass) ---
    sho_cfg = cfg.get("sho", {"enabled": True, "population": 6, "iterations": 5})
    if sho_cfg.get("enabled", True):
        print("[INFO] Running SHO initialization on the final head...")
        sho_initialize_final_head(
            model, train_ds,
            population=int(sho_cfg.get("population", 6)),
            iterations=int(sho_cfg.get("iterations", 5)),
            verbose=True
        )

    # --- Callbacks ---
    # Save WEIGHTS ONLY so infer.py can load them via model.load_weights(...)
    ckpt_path = out_ckpt / "best.weights.h5"
    callbacks = [
        ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,  # important for compatibility with infer.py
            verbose=1,
        ),
        EarlyStopping(monitor="val_loss", patience=int(cfg.get("patience", 4)), restore_best_weights=True),
        CSVLogger(str(out_logs / "train_log.csv")),
    ]

    # --- Train ---
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg.get("epochs", 10)),
        callbacks=callbacks,
    )

    print(f"[INFO] Best weights saved to: {ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/default.yaml")
    args = ap.parse_args()
    main(args.config)
