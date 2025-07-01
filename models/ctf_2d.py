import numpy as np
from .ctf_base import CTFBase
from .ctf_envelope import CTFEnvelope
from .microscope import Microscope


class CTF2D(CTFBase):
    def __init__(
        self,
        microscope: Microscope = None,
        defocus: float = 1.0,
        defocus_diff: float = 0.0,
        defocus_az: float = 0.0,
        amplitude_contrast: float = 0.1,
        phase_shift_deg: float = 0.0,
        include_temporal_env: bool = True,
        include_spatial_env: bool = True,
        include_detector_env: bool = True,
        include_ice: bool = False,
        ice_thickness: float = 0.0,
        detector_model=None,
        pixel_size: float = 1.0,
    ):
        self._init_done = False

        super().__init__(
            microscope=microscope,
            defocus=defocus,
            defocus_diff=defocus_diff,
            defocus_az=defocus_az,
            amplitude_contrast=amplitude_contrast,
            phase_shift_deg=phase_shift_deg,
        )

        self.envelope = CTFEnvelope(
            microscope=self.microscope,
            defocus=defocus,
            defocus_diff=defocus_diff,
            defocus_az=defocus_az,
            include_temporal=include_temporal_env,
            include_spatial=include_spatial_env,
            include_detector=include_detector_env,
            include_ice=include_ice,
            detector_model=detector_model,
            ice_thickness=ice_thickness,
            pixel_size=pixel_size,
        )

        self._init_done = True

    @property
    def defocus(self):
        return super().defocus

    @defocus.setter
    def defocus(self, value):
        CTFBase.defocus.fset(self, value)
        if getattr(self, "_init_done", False):
            self.envelope.defocus = value  # Keep in syncs

    @property
    def defocus_diff(self):
        return super().defocus_diff

    @defocus_diff.setter
    def defocus_diff(self, value):
        CTFBase.defocus_diff.fset(self, value)
        if getattr(self, "_init_done", False):
            self.envelope.defocus_diff = value  # Keep in syncs

    @property
    def defocus_az(self):
        return super().defocus_az

    @defocus_az.setter
    def defocus_az(self, value):
        CTFBase.defocus_az.fset(self, value)
        if getattr(self, "_init_done", False):
            self.envelope.defocus_az = value  # Keep in syncs

    def ctf(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        raw = self.ctf_2d(fx, fy)  # from CTFBase (calls γ₂ internally)
        env = self.envelope.total_2d(fx, fy)
        return raw * env
