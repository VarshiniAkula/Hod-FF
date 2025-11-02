from tensorflow.keras import layers, Model
from tensorflow.keras.applications import InceptionResNetV2
from . import hodff_dd as _ignore   # circular import guard in some IDEs
from ..utils.common import prewhiten, preprocess_irv2

def get_irv2_base(trainable=False):
    base = InceptionResNetV2(
        include_top=False, weights="imagenet", pooling="avg"
    )
    base.trainable = trainable
    return base

def get_irv1_facenet_base(trainable=False):
    """
    Use FaceNet (keras-facenet) which is InceptionResNetV1-based.
    Output: 512-d embedding per image of size 160x160.
    """
    try:
        from keras_facenet import FaceNet
    except Exception as e:
        raise RuntimeError(
            "Please `pip install keras-facenet` to use the InceptionResNetV1/FaceNet branch."
        ) from e
    embedder = FaceNet()
    base = embedder.model  # tf.keras.Model
    base.trainable = trainable
    return base

def build_dual_feature_extractors(seq_len, img_size, freeze_backbones=True):
    """
    Returns:
      input_frames: (T, img_size, img_size, 3)
      feats_v1: (T, 512)
      feats_v2: (T, D2)
    """
    input_frames = layers.Input(shape=(seq_len, img_size, img_size, 3), name="frames")

    # Branch: InceptionResNetV1 (FaceNet) at 160x160, with prewhitening
    x_v1 = layers.TimeDistributed(layers.Resizing(160, 160), name="v1_resize")(input_frames)
    x_v1 = layers.TimeDistributed(layers.Lambda(prewhiten), name="v1_prewhiten")(x_v1)
    base_v1 = get_irv1_facenet_base(trainable=not freeze_backbones)
    feats_v1 = layers.TimeDistributed(base_v1, name="v1_facenet")(x_v1)  # (T, 512)

    # Branch: InceptionResNetV2 at 299x299, with IRv2 preprocessing
    x_v2 = layers.TimeDistributed(layers.Lambda(preprocess_irv2), name="v2_preproc")(input_frames)
    base_v2 = get_irv2_base(trainable=not freeze_backbones)
    feats_v2 = layers.TimeDistributed(base_v2, name="v2_irv2")(x_v2)     # (T, D2)

    return input_frames, feats_v1, feats_v2
