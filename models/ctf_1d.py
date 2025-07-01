import numpy as np
from .ctf_base import CTFBase
from .ctf_envelope import CTFEnvelope
from .microscope import Microscope


class CTF1D(CTFBase):
    def __init__(
        self,
        microscope: Microscope = None,
        defocus: float = 1.0,
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
            amplitude_contrast=amplitude_contrast,
            phase_shift_deg=phase_shift_deg,
        )

        self.envelope = CTFEnvelope(
            microscope=self.microscope,
            defocus=defocus,
            include_temporal=include_temporal_env,
            include_spatial=include_spatial_env,
            include_detector=include_detector_env,
            include_ice=include_ice,
            detector_model=detector_model,
            ice_thickness=ice_thickness,
            pixel_size=pixel_size,
        )

        self._init_done = True

        # Ensure envelope.defocus is synced with final defocus value
        self.defocus = defocus

    @property
    def defocus(self):
        return super().defocus

    @defocus.setter
    def defocus(self, value):
        CTFBase.defocus.fset(self, value)
        if getattr(self, "_init_done", False):
            self.envelope.defocus = value  # Keep in syncs

    def ctf(self, freq: np.ndarray) -> np.ndarray:
        raw = self.ctf_1d(freq)  # from CTFBase
        env = self.envelope.total_1d(freq)
        return raw * env
