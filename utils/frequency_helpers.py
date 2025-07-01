import numpy as np


def compute_resampled_freqs(pixel_size: float, image_size: int):
    """
    Given a pixel_size and image_size, return (fx, fy, nyquist) as meshgrid arrays.
    """
    nyquist = 0.5 / pixel_size
    freq_x = np.linspace(-nyquist, nyquist, image_size)
    freq_y = np.linspace(-nyquist, nyquist, image_size)
    fx, fy = np.meshgrid(freq_x, freq_y, sparse=True)
    return fx, fy, nyquist
