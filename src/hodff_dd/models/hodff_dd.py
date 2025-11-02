from tensorflow.keras import layers, Model
from .backbones import build_dual_feature_extractors

def build_hodff_dd(seq_len=8, img_size=299, freeze_backbones=True):
    inp, feats_v1, feats_v2 = build_dual_feature_extractors(
        seq_len=seq_len, img_size=img_size, freeze_backbones=freeze_backbones
    )
    fused = layers.Concatenate(name="concat_v1_v2")([feats_v1, feats_v2])   # (T, d1+d2)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="bilstm_128")(fused)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bilstm_64")(x)
    x = layers.TimeDistributed(layers.Dense(64, activation="relu"), name="td_dense_64")(x)
    out = layers.TimeDistributed(layers.Dense(2, activation="softmax"), name="frame_logits")(x)
    model = Model(inp, out, name="HODFF_DD")
    return model
