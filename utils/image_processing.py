import os
import numpy as np
from PIL import Image


def load_and_prepare_image(path: str, target_size: int) -> np.ndarray:
    """
    Load an image from disk, convert to grayscale, crop to square centered,
    resize to (target_size x target_size), and normalize to [0,1].
    If the file doesn’t exist (or fails), return a random image of that size.

    Args:
        path (str): Path to the image file.
        target_size (int): Size of the image.

    Returns:
            np.ndarray: A (self.image_size x self.image_size) normalized grayscale image.
    """
    if not os.path.exists(path):
        print(f"[Warning] Image not found at '{path}'. Generating random image.")
        return np.random.rand(target_size, target_size).astype(np.float32)

    try:
        img = Image.open(path).convert("L")
        w, h = img.size
        # Crop to square
        crop_side = min(w, h)
        left = (w - crop_side) // 2
        top = (h - crop_side) // 2
        img = img.crop((left, top, left + crop_side, top + crop_side))
        # Resize
        img = img.resize((target_size, target_size), Image.LANCZOS)  # pylint: disable=no-member

        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    except Exception as e:
        print(f"[Error] Failed to load image '{path}': {e}. Using random fallback.")
        return np.random.rand(target_size, target_size).astype(np.float32)
