import numpy as np
from .microscope import Microscope


class CTFBase:
    """
    Base class for computing contrast transfer functions (1D or 2D).
    Holds defocus-related parameters (in µm), amplitude contrast, and phase shift.
    Provides separate methods for 1D and 2D phase aberration (with astigmatism).
    """

    def __init__(
        self,
        microscope: Microscope = None,
        defocus: float = 1.0,
        defocus_diff: float = 0.0,
        defocus_az: float = 0.0,
        amplitude_contrast: float = 0.1,
        phase_shift_deg: float = 0.0,
    ):
        # Private backing fields
        self._microscope = None

        # defocus and defocus_diff are stored in µm
        self._defocus = None
        self._defocus_diff = None
        self._defocus_az = None  # in degrees

        self._amplitude_contrast = None
        self._phase_shift_deg = None

        # Assign via setters for validation
        self.microscope = microscope or Microscope()
        self.defocus = defocus
        self.defocus_diff = defocus_diff
        self.defocus_az = defocus_az
        self.amplitude_contrast = amplitude_contrast
        self.phase_shift_deg = phase_shift_deg

    # ------------------------------------------------------------
    # microscope
    # ------------------------------------------------------------
    @property
    def microscope(self) -> Microscope:
        """The Microscope instance (pure optics)."""
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
        """Defocus (µm). Can be positive or negative."""
        return self._defocus

    @defocus.setter
    def defocus(self, value: float) -> None:
        self._defocus = float(value)

    # ------------------------------------------------------------
    # defocus_diff (µm)
    # ------------------------------------------------------------
    @property
    def defocus_diff(self) -> float:
        """Defocus difference (µm) used for astigmatism."""
        return self._defocus_diff

    @defocus_diff.setter
    def defocus_diff(self, value: float) -> None:
        self._defocus_diff = float(value)

    # ------------------------------------------------------------
    # defocus_az (degrees)
    # ------------------------------------------------------------
    @property
    def defocus_az(self) -> float:
        """Defocus azimuth angle (degrees) for astigmatism."""
        return self._defocus_az

    @defocus_az.setter
    def defocus_az(self, value: float) -> None:
        self._defocus_az = float(value)

    # ------------------------------------------------------------
    # amplitude_contrast (0–1)
    # ------------------------------------------------------------
    @property
    def amplitude_contrast(self) -> float:
        """Amplitude contrast (fraction between 0 and 1)."""
        return self._amplitude_contrast

    @amplitude_contrast.setter
    def amplitude_contrast(self, value: float) -> None:
        if not (0 <= value <= 1):
            raise ValueError("Amplitude contrast must be between 0 and 1.")
        self._amplitude_contrast = float(value)

    # ------------------------------------------------------------
    # phase_shift_deg (degrees)
    # ------------------------------------------------------------
    @property
    def phase_shift_deg(self) -> float:
        """Phase shift (degrees)."""
        return self._phase_shift_deg

    @phase_shift_deg.setter
    def phase_shift_deg(self, value: float) -> None:
        self._phase_shift_deg = float(value)

    # ------------------------------------------------------------
    # Phase aberration for 1D: γ₁(f)
    # ------------------------------------------------------------
    def phase_aberration_1d(self, freq: np.ndarray) -> np.ndarray:
        """
        Compute the 1D aberration phase γ₁(f) = −(π/2)·C_s·λ³·f⁴ + π·d_f·λ·f² + φ₀,
        where:
          - freq: spatial frequency array (Å⁻¹)
          - λ = electron wavelength (Å) from microscope
          - C_s = spherical aberration in mm → converted to Å (Cs * 1e7)
          - d_f = defocus + defocus_diff (both in µm → convert to Å by * 1e4)
          - φ₀ = phase_shift_deg (converted to radians)
        Returns: array of γ₁(f) (radians), vectorized over freq.
        """
        λ = self.microscope.wavelength  # in Å
        Cs = self.microscope.cs * 1e7  # Cs (mm → Å)
        f = freq  # array or scalar

        # Convert defocus (µm) to Å: 1 µm = 10^4 Å
        d_f = (self.defocus + self.defocus_diff) * 1e4  # in Å

        # Compute base phase
        # γ₁(f) = −(π/2)*Cs*λ³*f⁴ + π*d_f*λ*f² + φ₀
        base_phase = (
            -(np.pi / 2.0) * Cs * (λ**3) * (f**4)
            + np.pi * d_f * λ * (f**2)
            + np.deg2rad(self.phase_shift_deg)
        )
        return base_phase

    # ------------------------------------------------------------
    # Phase aberration for 2D: γ₂(fx, fy)
    # ------------------------------------------------------------
    def phase_aberration_2d(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        """
        Compute the 2D aberration phase γ₂(fx, fy) including astigmatism:
          1. f = sqrt(fx² + fy²), φ = arctan2(fy, fx)
          2. λ = microscope.wavelength (Å)
          3. Cs = microscope.cs * 1e7 (Cs in Å)
          4. d_u = (defocus + defocus_diff) * 1e4  (µm→Å)
             d_v = (defocus − defocus_diff) * 1e4  (µm→Å)
          5. φ_a = defocus_az in radians
          6. d_f_eff = 0.5 [ d_u + d_v + (d_u − d_v)·cos(2(φ − φ_a)) ]
          7. γ₂ = −(π/2)*Cs*λ³*f⁴ + π*d_f_eff*λ*f² + φ₀
        Returns: array of γ₂(fx,fy), vectorized over fx, fy.
        """
        λ = self.microscope.wavelength  # in Å
        Cs = self.microscope.cs * 1e7  # Cs (mm → Å)

        # Radial frequency and angle
        f = np.sqrt(fx**2 + fy**2)
        φ = np.arctan2(fy, fx)  # in radians
        φ_a = np.deg2rad(self.defocus_az)  # convert to radians

        # Convert defocus values (µm) to Å
        d_u = (self.defocus + self.defocus_diff / 2) * 1e4  # in Å
        d_v = (self.defocus - self.defocus_diff / 2) * 1e4  # in Å

        # Effective defocus with astigmatism
        d_f_eff = 0.5 * (d_u + d_v + (d_u - d_v) * np.cos(2.0 * (φ - φ_a)))  # shape matches fx, fy

        # φ₀ in radians
        phi0 = np.deg2rad(self.phase_shift_deg)

        # Compute γ₂ for each (fx, fy)
        # γ₂ = −(π/2)*Cs*λ³*f⁴ + π*d_f_eff*λ*f² + φ₀
        base_phase = -(np.pi / 2.0) * Cs * (λ**3) * (f**4) + np.pi * d_f_eff * λ * (f**2) + phi0
        return base_phase

    # ------------------------------------------------------------
    # Raw CTF 1D: CTF₁(f) = −sin[γ₁(f) + arcsin(A_c)]
    # ------------------------------------------------------------
    def ctf_1d(self, freq: np.ndarray) -> np.ndarray:
        """
        Compute raw 1D CTF (no envelopes or ice).
        Returns array of same shape as freq.
        """
        gamma = self.phase_aberration_1d(freq)
        return -np.sin(gamma + np.arcsin(self.amplitude_contrast))

    # ------------------------------------------------------------
    # Raw CTF 2D: CTF₂(fx, fy) = −sin[γ₂(fx, fy) + arcsin(A_c)]
    # ------------------------------------------------------------
    def ctf_2d(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        """
        Compute raw 2D CTF (no envelopes or ice).
        Returns array matching fx/fy shape.
        """
        gamma = self.phase_aberration_2d(fx, fy)
        return -np.sin(gamma + np.arcsin(self.amplitude_contrast))
