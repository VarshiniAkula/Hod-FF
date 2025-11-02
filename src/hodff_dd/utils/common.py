import os, json, random, yaml
from pathlib import Path
import numpy as np
import tensorflow as tf

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def prewhiten(x):
    """
    FaceNet prewhitening as a Keras Lambda-friendly op.
    x: 4D tensor [batch, H, W, C] with float32 0..255
    """
    x = tf.cast(x, tf.float32)
    mean, var = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
    stddev = tf.sqrt(var)
    # Avoid div-by-zero per FaceNet's definition
    pixel_count = tf.cast(tf.shape(x)[1] * tf.shape(x)[2] * tf.shape(x)[3], tf.float32)
    std_adj = tf.maximum(stddev, 1.0 / tf.sqrt(pixel_count))
    y = (x - mean) / std_adj
    return y

def preprocess_irv2(x):
    # Expect float32 0..255 -> -1..1
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    return preprocess_input(x)

def one_hot(labels, num_classes=2):
    arr = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, lbl in enumerate(labels):
        arr[i, lbl] = 1.0
    return arr

def majority_vote(frame_scores):
    """
    frame_scores: (num_frames, 2) softmax scores
    Returns predicted_label, counts
    """
    preds = np.argmax(frame_scores, axis=1)
    real = int(np.sum(preds == 0))
    fake = int(np.sum(preds == 1))
    label = 0 if real >= fake else 1
    return label, {"real": real, "fake": fake}

def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
