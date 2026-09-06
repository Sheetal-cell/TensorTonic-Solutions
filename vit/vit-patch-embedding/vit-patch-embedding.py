import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Returns float64 patch embeddings with shape (B, N, D).
    """
    B, H, W, C = image.shape

    # Number of complete patches along height and width
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    # Extract complete patches
    patches = image[
        :,
        :num_patches_h * patch_size,
        :num_patches_w * patch_size,
        :
    ]

    # Reshape into patches
    patches = patches.reshape(
        B,
        num_patches_h,
        patch_size,
        num_patches_w,
        patch_size,
        C
    )

    # Reorder so patches are arranged row by row
    patches = patches.transpose(
        0, 1, 3, 2, 4, 5
    )

    # Flatten each patch
    patches = patches.reshape(
        B,
        num_patches_h * num_patches_w,
        patch_size * patch_size * C
    )

    # Linear projection + bias
    embeddings = patches @ W_proj + bias

    return embeddings.astype(np.float64)