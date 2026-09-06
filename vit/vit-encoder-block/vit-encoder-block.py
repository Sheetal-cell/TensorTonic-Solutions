import numpy as np

import numpy as np


def vit_encoder_block(x: np.ndarray, num_heads: int,
                      Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                      Wo: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 output of one pre-normalized ViT encoder block.
    """

    B, N, D = x.shape
    dh = D // num_heads
    eps = 1e-6

    # --------------------------------------------------
    # LayerNorm
    # --------------------------------------------------
    def layer_norm(a):
        mean = np.mean(a, axis=-1, keepdims=True)
        var = np.mean((a - mean) ** 2, axis=-1, keepdims=True)
        return (a - mean) / np.sqrt(var + eps)

    # --------------------------------------------------
    # GELU
    # --------------------------------------------------
    def gelu(a):
        return 0.5 * a * (
            1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) *
                (a + 0.044715 * a ** 3)
            )
        )

    # --------------------------------------------------
    # 1. Pre-LayerNorm
    # --------------------------------------------------
    x_norm = layer_norm(x)

    # --------------------------------------------------
    # 2. Compute Q, K, V
    # --------------------------------------------------
    Q = x_norm @ Wq
    K = x_norm @ Wk
    V = x_norm @ Wv

    # --------------------------------------------------
    # 3. Split into multiple heads
    # Shape: (B, N, D) -> (B, H, N, dh)
    # --------------------------------------------------
    Q = Q.reshape(B, N, num_heads, dh).transpose(0, 2, 1, 3)
    K = K.reshape(B, N, num_heads, dh).transpose(0, 2, 1, 3)
    V = V.reshape(B, N, num_heads, dh).transpose(0, 2, 1, 3)

    # --------------------------------------------------
    # 4. Attention scores
    # Q @ K^T / sqrt(dh)
    # --------------------------------------------------
    scores = Q @ K.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(dh)

    # --------------------------------------------------
    # 5. Softmax over key/token dimension
    # --------------------------------------------------
    scores = scores - np.max(
        scores, axis=-1, keepdims=True
    )

    attention = np.exp(scores)
    attention = attention / np.sum(
        attention, axis=-1, keepdims=True
    )

    # --------------------------------------------------
    # 6. Attention weighted values
    # --------------------------------------------------
    A = attention @ V

    # --------------------------------------------------
    # 7. Concatenate heads
    # (B,H,N,dh) -> (B,N,D)
    # --------------------------------------------------
    A = A.transpose(0, 2, 1, 3)
    A = A.reshape(B, N, D)

    # --------------------------------------------------
    # 8. Output projection
    # --------------------------------------------------
    attn_output = A @ Wo

    # --------------------------------------------------
    # 9. First residual
    # --------------------------------------------------
    y = x + attn_output

    # --------------------------------------------------
    # 10. Second Pre-LayerNorm
    # --------------------------------------------------
    y_norm = layer_norm(y)

    # --------------------------------------------------
    # 11. Two-layer GELU MLP
    # --------------------------------------------------
    hidden = y_norm @ W1
    hidden = gelu(hidden)
    mlp_output = hidden @ W2

    # --------------------------------------------------
    # 12. Second residual
    # --------------------------------------------------
    z = y + mlp_output

    return z.astype(np.float64)