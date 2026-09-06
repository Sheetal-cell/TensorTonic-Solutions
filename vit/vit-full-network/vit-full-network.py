import numpy as np

def vit_forward(image: np.ndarray, patch_size: int, num_heads: int,
                W_patch: np.ndarray, patch_bias: np.ndarray,
                cls_token: np.ndarray, pos_embed: np.ndarray,
                encoder_weights: list, W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 Vision Transformer logits with shape (B, C).
    """

    # --------------------------------------------------
    # 1. PATCH EMBEDDING
    # --------------------------------------------------

    B, H, W, C_in = image.shape
    P = patch_size

    num_patches_h = H // P
    num_patches_w = W // P

    H_complete = num_patches_h * P
    W_complete = num_patches_w * P

    patches = image[
        :,
        :H_complete,
        :W_complete,
        :
    ]

    patches = patches.reshape(
        B,
        num_patches_h,
        P,
        num_patches_w,
        P,
        C_in
    )

    patches = patches.transpose(
        0, 1, 3, 2, 4, 5
    )

    N = num_patches_h * num_patches_w

    patches = patches.reshape(
        B,
        N,
        P * P * C_in
    )

    # Linear projection
    tokens = patches @ W_patch + patch_bias


    # --------------------------------------------------
    # 2. ADD CLS TOKEN
    # --------------------------------------------------

    cls_tokens = np.broadcast_to(
        cls_token,
        (B, 1, tokens.shape[-1])
    )

    tokens = np.concatenate(
        [cls_tokens, tokens],
        axis=1
    )


    # --------------------------------------------------
    # 3. ADD POSITION EMBEDDING
    # --------------------------------------------------

    tokens = tokens + pos_embed


    # --------------------------------------------------
    # 4. TRANSFORMER ENCODER BLOCKS
    # --------------------------------------------------

    for weights in encoder_weights:

        D = tokens.shape[-1]
        dh = D // num_heads
        eps = 1e-6

        # ---------- LayerNorm ----------
        mean = np.mean(
            tokens,
            axis=-1,
            keepdims=True
        )

        var = np.mean(
            (tokens - mean) ** 2,
            axis=-1,
            keepdims=True
        )

        x_norm = (
            (tokens - mean)
            / np.sqrt(var + eps)
        )


        # ---------- Q, K, V ----------
        Q = x_norm @ weights["Wq"]
        K = x_norm @ weights["Wk"]
        V = x_norm @ weights["Wv"]


        # ---------- Split into heads ----------
        Q = Q.reshape(
            B, N + 1, num_heads, dh
        ).transpose(
            0, 2, 1, 3
        )

        K = K.reshape(
            B, N + 1, num_heads, dh
        ).transpose(
            0, 2, 1, 3
        )

        V = V.reshape(
            B, N + 1, num_heads, dh
        ).transpose(
            0, 2, 1, 3
        )


        # ---------- Attention scores ----------
        scores = Q @ K.transpose(
            0, 1, 3, 2
        )

        scores = scores / np.sqrt(dh)


        # ---------- Softmax ----------
        scores = scores - np.max(
            scores,
            axis=-1,
            keepdims=True
        )

        attention = np.exp(scores)

        attention = attention / np.sum(
            attention,
            axis=-1,
            keepdims=True
        )


        # ---------- Weighted values ----------
        A = attention @ V


        # ---------- Concatenate heads ----------
        A = A.transpose(
            0, 2, 1, 3
        )

        A = A.reshape(
            B,
            N + 1,
            D
        )


        # ---------- Output projection ----------
        attn_output = A @ weights["Wo"]


        # ---------- Residual connection ----------
        y = tokens + attn_output


        # ---------- Second LayerNorm ----------
        mean = np.mean(
            y,
            axis=-1,
            keepdims=True
        )

        var = np.mean(
            (y - mean) ** 2,
            axis=-1,
            keepdims=True
        )

        y_norm = (
            (y - mean)
            / np.sqrt(var + eps)
        )


        # ---------- MLP ----------
        hidden = y_norm @ weights["W1"]

        hidden = 0.5 * hidden * (
            1.0 + np.tanh(
                np.sqrt(2.0 / np.pi)
                * (
                    hidden
                    + 0.044715 * hidden ** 3
                )
            )
        )

        mlp_output = hidden @ weights["W2"]


        # ---------- Residual connection ----------
        tokens = y + mlp_output


    # --------------------------------------------------
    # 5. EXTRACT CLS TOKEN
    # --------------------------------------------------

    h = tokens[:, 0, :]


    # --------------------------------------------------
    # 6. FINAL LAYER NORMALIZATION
    # --------------------------------------------------

    mean = np.mean(
        h,
        axis=-1,
        keepdims=True
    )

    var = np.mean(
        (h - mean) ** 2,
        axis=-1,
        keepdims=True
    )

    h_norm = (
        (h - mean)
        / np.sqrt(var + 1e-6)
    )


    # --------------------------------------------------
    # 7. CLASSIFICATION HEAD
    # --------------------------------------------------

    logits = h_norm @ W_head

    return logits.astype(np.float64)