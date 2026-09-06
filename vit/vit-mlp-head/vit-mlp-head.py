import numpy as np


def classification_head(
    encoder_output: np.ndarray,
    W_head: np.ndarray
) -> np.ndarray:

    # 1. Extract [CLS] token (position 0)
    h = encoder_output[:, 0, :]

    # 2. LayerNorm
    mean = np.mean(h, axis=-1, keepdims=True)
    var = np.mean((h - mean) ** 2, axis=-1, keepdims=True)

    h_norm = (h - mean) / np.sqrt(var + 1e-6)

    # 3. Classification projection
    logits = h_norm @ W_head

    return logits.astype(np.float64)