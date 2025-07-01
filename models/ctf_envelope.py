import numpy as np
from .microscope import Microscope
from .detector import Detector, get_detector_by_index


class CTFEnvelope:
    """
    Encapsulates envelopes for:
      1) temporal (pure microscope),
      2) spatial (1D vs. 2D is different),
      3) detector (isotropic → use radial freq),
      4) ice (isotropic → use radial freq).

    The caller multiplies raw CTF × total_1d or total_2d.
    """

    def __init__(
        self,
        microscope: Microscope = None,
        defocus: float = 1.0,
        defocus_diff: float = 0.0,
        defocus_az: float = 0.0,
        include_temporal: bool = True,
        include_spatial: bool = True,
        include_detector: bool = True,
        include_ice: bool = True,
        detector_model: Detector = None,
        ice_thickness: float = 0.0,
        pixel_size: float = 1.0,
    ):
        # Backing fields
        self._microscope = None
        self._defocus = None
        self._defocus_diff = None
        self._defocus_az = None

        self._include_temporal = None
        self._include_spatial = None
        self._include_detector = None
        self._include_ice = None

        self._detector_model = None
        self._ice_thickness = None

        self._pixel_size = None

        # Assign via setters
        self.microscope = microscope or Microscope()
        self.defocus = defocus
        self.defocus_diff = defocus_diff
        self.defocus_az = defocus_az
        self.include_temporal = include_temporal
        self.include_spatial = include_spatial
        self.include_detector = include_detector
        self.include_ice = include_ice
        self.detector_model = detector_model
        self.ice_thickness = ice_thickness
        self.pixel_size = pixel_size

    # ------------------------------------------------------------
    # Microscope
    # ------------------------------------------------------------
    @property
    def microscope(self) -> Microscope:
        return self._microscope

    @microscope.setter
    def microscope(self, value: Microscope) -> None:
        if not isinstance(value, Microscope):
            raise ValueError("microscope must be a Microscope instance.")
        self._microscope = value

    # ------------------------------------------------------------
    # defocus (µm)
    # ------------------------------------------------------------
    @property
    def defocus(self) -> float:
        return self._defocus

    @defocus.setter
    def defocus(self, value: float) -> None:
        self._defocus = float(value)

    # ------------------------------------------------------------
    # defocus_diff (µm)
    # ------------------------------------------------------------
    @property
    def defocus_diff(self) -> float:
        return self._defocus_diff

    @defocus_diff.setter
    def defocus_diff(self, value: float) -> None:
        self._defocus_diff = float(value)

    # ------------------------------------------------------------
    # defocus_az (degrees)
    # ------------------------------------------------------------
    @property
    def defocus_az(self) -> float:
        return self._defocus_az

    @defocus_az.setter
    def defocus_az(self, value: float) -> None:
        self._defocus_az = float(value)

    # ------------------------------------------------------------
    # include_temporal, include_spatial, include_detector, include_ice
    # ------------------------------------------------------------
    @property
    def include_temporal(self) -> bool:
        return self._include_temporal

    @include_temporal.setter
    def include_temporal(self, value: bool) -> None:
        self._include_temporal = bool(value)

    @property
    def include_spatial(self) -> bool:
        return self._include_spatial

    @include_spatial.setter
    def include_spatial(self, value: bool) -> None:
        self._include_spatial = bool(value)

    @property
    def include_detector(self) -> bool:
        return self._include_detector

    @include_detector.setter
    def include_detector(self, value: bool) -> None:
        self._include_detector = bool(value)

    @property
    def include_ice(self) -> bool:
        return self._include_ice

    @include_ice.setter
    def include_ice(self, value: bool) -> None:
        self._include_ice = bool(value)

    # ------------------------------------------------------------
    # detector_model (Detector)
    # ------------------------------------------------------------
    @property
    def detector_model(self) -> Detector | None:
        return self._detector_model

    @detector_model.setter
    def detector_model(self, value: int | Detector | None) -> None:
        # 1) None → disable detector envelope
        if value is None:
            self._detector_model = None
            return

        # 2) Already a Detector instance?
        if isinstance(value, Detector):
            self._detector_model = value
            return

        # 3) An integer index?
        if isinstance(value, int):
            try:
                detector = get_detector_by_index(value)
            except (IndexError, TypeError):
                raise ValueError(f"No detector at index {value!r}")
            if not isinstance(detector, Detector):
                raise ValueError(f"get_detector_by_index({value!r}) didn’t return a Detector")
            self._detector_model = detector
            return

        # 4) Anything else is invalid
        raise ValueError(
            "detector_model must be either:\n"
            "  • None\n"
            "  • a Detector instance\n"
            "  • an integer index of a Detector\n"
            f"received {value!r}"
        )

    # ------------------------------------------------------------
    # ice_thickness (nm)
    # ------------------------------------------------------------
    @property
    def ice_thickness(self) -> float:
        return self._ice_thickness

    @ice_thickness.setter
    def ice_thickness(self, value: float) -> None:
        if value < 0:
            raise ValueError("Ice thickness must be ≥ 0.")
        self._ice_thickness = float(value)

    # ------------------------------------------------------------
    # nyquist (Å⁻¹)
    # ------------------------------------------------------------
    @property
    def nyquist(self) -> float:
        return 1 / (2 * self._pixel_size * self.detector_model.binning_factor)

    @nyquist.setter
    def nyquist(self, value: float) -> None:
        if value <= 0:
            raise ValueError("nyquist must be positive.")
        self._pixel_size = 1 / (2 * value * self.detector_model.binning_factor)

    # ------------------------------------------------------------
    # pixel_size (Å/pixel)
    # ------------------------------------------------------------
    @property
    def pixel_size(self) -> float:
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, value: float):
        if value <= 0:
            raise ValueError("Pixel size must be positive.")
        self._pixel_size = value

    # ------------------------------------------------------------
    # 1) Temporal envelope (isotropic; works in 1D or 2D)
    # ------------------------------------------------------------
    def temporal(self, freq: np.ndarray) -> np.ndarray:
        """
        E_T(f) = exp( − (π^2 λ^2 f_s^2 f^4) / 2 )
        where f_s = Cc * sqrt((ΔV/V)^2 + (2ΔI/I)^2 + (ΔE/eV)^2).
        Returns 1 if include_temporal=False; otherwise calls microscope.temporal_envelope(freq).
        freq can be a 1D array or a 2D grid of radial frequencies.
        """
        if not self.include_temporal:
            return np.ones_like(freq)

        ΔV_over_V = self.microscope.voltage_stability
        ΔI_over_I = self.microscope.obj_lens_stability
        ΔE_over_eV = self.microscope.electron_source_spread / (self.microscope.voltage * 1e3)

        f_s = (
            self.microscope.cc * 1e7 * np.sqrt(ΔV_over_V**2 + (2 * ΔI_over_I) ** 2 + ΔE_over_eV**2)
        )
        λ = self.microscope.wavelength

        exponent = -(np.pi**2 * λ**2 * f_s**2 * freq**4) / 2.0
        return np.exp(exponent)

    # ------------------------------------------------------------
    # 2) Spatial envelope 1D and 2D differ, so keep them separate
    # ------------------------------------------------------------
    def spatial_1d(self, freq: np.ndarray) -> np.ndarray:
        """
        1D spatial envelope: exp[−((π·e_a/λ)^2 · (C_s λ^3 f^3 + d_f λ f)^2)].
        d_f = (defocus + defocus_diff) * 1e4  (µm→Å).
        """
        if not self.include_spatial:
            return np.ones_like(freq)

        λ = self.microscope.wavelength  # Å
        Cs = self.microscope.cs * 1e7  # mm→Å
        d_f = self.defocus * 1e4  # µm→Å
        e_a = self.microscope.electron_source_angle

        term = (Cs * (λ**3) * freq**3) + (d_f * λ * freq)
        exponent = -((np.pi * e_a / λ) ** 2) * (term**2)
        return np.exp(exponent)

    def spatial_2d(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        """
        2D spatial envelope (astigmatism included):
          f = sqrt(fx² + fy²), φ = arctan2(fy, fx), φ_a = defocus_az (rad),
          d_u = (defocus + defocus_diff) * 1e4, d_v = (defocus − defocus_diff) * 1e4,
          d_f_eff = 0.5 [d_u + d_v + (d_u−d_v) cos 2(φ−φ_a)],
          then exp[−((π·e_a/λ)^2 · (C_s λ^3 f^3 + d_f_eff λ f)^2)].
        """
        if not self.include_spatial:
            return np.ones_like(fx)

        λ = self.microscope.wavelength
        Cs = self.microscope.cs * 1e7
        e_a = self.microscope.electron_source_angle

        f = np.sqrt(fx**2 + fy**2)
        φ = np.arctan2(fy, fx)
        φ_a = np.deg2rad(self.defocus_az)

        d_u = (self.defocus + self.defocus_diff / 2) * 1e4
        d_v = (self.defocus - self.defocus_diff / 2) * 1e4
        d_f_eff = 0.5 * (d_u + d_v + (d_u - d_v) * np.cos(2.0 * (φ - φ_a)))

        term = (Cs * (λ**3) * f**3) + (d_f_eff * λ * f)
        exponent = -((np.pi * e_a / λ) ** 2) * (term**2)
        return np.exp(exponent)

    # ------------------------------------------------------------
    # 3) Detector envelope: isotropic, so use one radial method
    # ------------------------------------------------------------
    def detector(self, freq: np.ndarray) -> np.ndarray:
        """
        Isotropic detector envelope. Given an array of radial frequencies (freq),
        computes ratio = freq/nyquist, then:
          raw_dqe = detector_model.get_dqe(ratio)   # already clips ratio internally
          raw_dqe[freq > nyquist] = 0
          return raw_dqe / detector_model.get_dqe(0.0)

        If include_detector=False or detector_model is None, returns ones.
        """
        if not self.include_detector or self.detector_model is None:
            return np.ones_like(freq)

        # Compute DQE at each ratio (Detector.dqe already clips ratio)
        ratio = freq / self.nyquist
        raw_dqe = self.detector_model.get_dqe(ratio)

        # Zero out above Nyquist
        raw_dqe = np.where(freq > self.nyquist, 0.0, raw_dqe)

        # Normalize so DQE(0)=1 (assuming detector_model.dqe(0)>0)
        ref_val = self.detector_model.get_dqe(np.array([0.0]))[0]
        if ref_val <= 0:
            return np.zeros_like(freq)
        return raw_dqe / ref_val

    # ------------------------------------------------------------
    # 4) Ice envelope: also isotropic, so one radial method
    # ------------------------------------------------------------
    def ice(self, freq: np.ndarray) -> np.ndarray:
        """
        Isotropic ice attenuation: sinc(π·λ·f²·t/2)/(π·λ·f²·t/2).
        If include_ice=False or ice_thickness<=0, returns ones.
        """
        if not self.include_ice or self.ice_thickness <= 0:
            return np.ones_like(freq)

        λ = self.microscope.wavelength
        t = self.ice_thickness * 10  # nm -> Å
        argument = λ * freq**2 * t / 2.0
        return np.sinc(argument)

    # ------------------------------------------------------------
    # Combined envelope for 1D
    # ------------------------------------------------------------
    def total_1d(self, freq: np.ndarray) -> np.ndarray:
        """
        Total 1D envelope = E_T(freq) · E_S_1D(freq) · E_D(freq) · E_I(freq),
        where E_D and E_I are isotropic (radial).
        """
        return self.temporal(freq) * self.spatial_1d(freq) * self.detector(freq) * self.ice(freq)

    # ------------------------------------------------------------
    # Combined envelope for 2D
    # ------------------------------------------------------------
    def total_2d(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        """
        Total 2D envelope = E_T(f) · E_S_2D(fx,fy) · E_D(f) · E_I(f),
        where f = sqrt(fx² + fy²).  Detector and ice both use that same f.
        """
        # radial frequency
        f = np.sqrt(fx**2 + fy**2)

        return (
            self.temporal(f)  # E_T on radial freq
            * self.spatial_2d(fx, fy)  # E_S_2D needs fx,fy separately
            * self.detector(f)  # E_D on radial freq
            * self.ice(f)  # E_I on radial freq
        )
