from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np


# ------------------------------------------------------------------------------
# 1. Helper function: fit a polynomial DQE curve and return a numpy-vectorized function
# ------------------------------------------------------------------------------
def build_polynomial_dqe(
    x: List[float], y: List[float], degree: Optional[int] = None
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Fit a polynomial to the given (x, y) points and return a function freq->DQE.
    If degree=None, choose 3 if len(x)>3 else 2.

    Returns a function f(ratio: np.ndarray) -> np.ndarray of non-negative DQE values.
    """
    if degree is None:
        degree = 3 if len(x) > 3 else 2

    # np.polyfit expects highest-order first
    coeffs = np.polyfit(x, y, degree)
    # But np.polynomial.Polynomial wants lowest-order first
    poly = np.polynomial.Polynomial(coeffs[::-1])

    def dqe_func(ratio: np.ndarray) -> np.ndarray:
        # Clip ratio to [0,1] (outside that range, DQE→0)
        r = np.clip(ratio, 0.0, 1.0)
        vals = poly(r)
        # Ensure non-negative
        return np.maximum(vals, 0.0)

    return dqe_func


# ------------------------------------------------------------------------------
# 2. Dataclass for a single detector
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Detector:
    """
    Represents one physical detector configuration.

    Attributes:
        name: Human-readable name (shown in combo box).
        dqe:  A function that takes f/nyquist (0..1) → DQE value (0..1).
        binning_factor: Multiplier to apply to pixel size to get effective Nyquist.
                        For example: 1.0 means “no binning,” 0.5 means “super-resolution,” etc.
    """

    name: str
    dqe: Callable[[np.ndarray], np.ndarray]
    binning_factor: float = 1.0

    def get_dqe(self, ratio: np.ndarray) -> np.ndarray:
        """
        Wraps self.dqe so that any out-of-bounds ratio is automatically clipped.
        """
        return self.dqe(ratio)


# ------------------------------------------------------------------------------
# 3. Precompute polynomial DQE functions for known detectors
# ------------------------------------------------------------------------------
# Gatan K3 (DDD) example points:
K3_DQE_X = [0.0, 0.5, 1.0]
K3_DQE_Y = [0.95, 0.71, 0.40]

# SO-163 “Film” example:
FILM_DQE_X = [0.0, 0.25, 0.5, 0.75, 1.0]
FILM_DQE_Y = [0.37, 0.32, 0.33, 0.22, 0.07]

# TVIPS 224 “CCD” example:
CCD_DQE_X = [0.0, 0.25, 0.5, 0.75, 1.0]
CCD_DQE_Y = [0.37, 0.16, 0.13, 0.10, 0.05]

# Build polynomial DQE functions (all clipped ≥0 under the hood)
_K3_POLY = build_polynomial_dqe(K3_DQE_X, K3_DQE_Y)
_FILM_POLY = build_polynomial_dqe(FILM_DQE_X, FILM_DQE_Y)
_CCD_POLY = build_polynomial_dqe(CCD_DQE_X, CCD_DQE_Y)


# ------------------------------------------------------------------------------
# 4. “Registry” of all supported detectors
# ------------------------------------------------------------------------------
ALL_DETECTORS: List[Detector] = [
    Detector(
        name="DDD Counting",
        dqe=_K3_POLY,
        binning_factor=1.0,
    ),
    Detector(
        name="DDD Super‐Resolution Counting",
        dqe=_K3_POLY,
        binning_factor=0.5,  # super-res mode
    ),
    Detector(
        name="Film",
        dqe=_FILM_POLY,
        binning_factor=1.0,
    ),
    Detector(
        name="CCD",
        dqe=_CCD_POLY,
        binning_factor=1.0,
    ),
]


# ------------------------------------------------------------------------------
# 5. Helpers to look up by name or index
# ------------------------------------------------------------------------------
def get_detector_by_name(name: str) -> Optional[Detector]:
    """
    Return the Detector whose .name matches exactly, or None if not found.
    """
    for det in ALL_DETECTORS:
        if det.name == name:
            return det
    return None


def get_detector_by_index(idx: int) -> Detector:
    """
    Return ALL_DETECTORS[idx]. Raises IndexError if out of range.
    """
    return ALL_DETECTORS[idx]
