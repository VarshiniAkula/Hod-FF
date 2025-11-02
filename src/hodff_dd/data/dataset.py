from pathlib import Path
import numpy as np
import cv2, json, math, random
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from ..utils.common import ensure_dir, one_hot

def _list_video_dirs(processed_root, classes):
    items = []
    for label, cls in enumerate(classes):
        for vd in sorted((processed_root / cls).glob("*")):
            if vd.is_dir():
                items.append((vd, label))
    return items

def _list_frames(video_dir):
    return sorted([p for p in video_dir.glob("*.jpg")])

def _sample_sequences(frames, seq_len, max_sequences):
    if len(frames) < seq_len:
        return [frames]  # will pad later
    # pick up to max_sequences evenly spaced windows
    step = max(1, (len(frames) - seq_len) // max(1, max_sequences))
    starts = list(range(0, max(1, len(frames) - seq_len + 1), step))
    if len(starts) > max_sequences:
        starts = starts[:max_sequences]
    seqs = []
    for s in starts:
        seqs.append(frames[s:s+seq_len])
    return seqs

def _load_sequence(frames, img_size, seq_len):
    imgs = []
    for i in range(seq_len):
        if i < len(frames):
            im = cv2.imread(str(frames[i]))
        else:
            im = cv2.imread(str(frames[-1]))  # pad with last
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (img_size, img_size))
        imgs.append(im)
    arr = np.stack(imgs, axis=0).astype(np.float32)  # (T, H, W, 3) 0..255
    return arr

def build_index(processed_root, classes, seq_len, max_sequences):
    items = _list_video_dirs(processed_root, classes)
    index = []
    for vd, label in items:
        frames = _list_frames(vd)
        if len(frames) == 0:
            continue
        sequences = _sample_sequences(frames, seq_len, max_sequences)
        for seq in sequences:
            index.append({
                "frames": [str(x) for x in seq],
                "label": int(label),
                "video": vd.name,
                "class": classes[label]
            })
    return index

def split_index(index, val_split=0.2, test_split=0.2, seed=42):
    labels = [it["label"] for it in index]
    idxs = list(range(len(index)))
    train_idx, test_idx = train_test_split(
        idxs, test_size=test_split, stratify=labels, random_state=seed
    )
    train_labels = [labels[i] for i in train_idx]
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_split/(1.0 - test_split),
        stratify=train_labels, random_state=seed
    )
    return train_idx, val_idx, test_idx

class SequenceDataset(Sequence):
    def __init__(self, index, idxs, img_size=299, seq_len=8, batch_size=2, shuffle=True):
        self.index = index
        self.idxs = list(idxs)
        self.img_size = img_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.idxs) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.idxs)

    def __getitem__(self, i):
        batch_idxs = self.idxs[i*self.batch_size:(i+1)*self.batch_size]
        X = []
        Y = []
        for bi in batch_idxs:
            item = self.index[bi]
            frames = [Path(p) for p in item["frames"]]
            arr = _load_sequence(frames, self.img_size, self.seq_len)  # (T, H, W, 3)
            X.append(arr)
            # frame-level labels (same label repeated for each frame)
            y = np.tile(one_hot([item["label"]], 2)[0], (self.seq_len, 1))
            Y.append(y)
        X = np.stack(X, axis=0)  # (B, T, H, W, 3)
        Y = np.stack(Y, axis=0)  # (B, T, 2)
        return X, Y
