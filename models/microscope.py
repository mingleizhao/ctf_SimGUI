import numpy as np


class Microscope:
    """
    Encapsulates all _pure_ microscope parameters and the temporal envelope.
    No defocus or spatial/detector envelope lives here anymore.
    """

    # Physical constants (SI)
    _m_e = 9.10938356e-31  # electron rest mass (kg)
    _eV = 1.602176634e-19  # electron volt (J)
    _c = 2.99792458e8  # speed of light (m/s)
    _h = 6.62607015e-34  # Planck constant (J·s)

    def __init__(
        self,
        voltage: float = 300.0,  # in keV
        voltage_stability: float = 3.3333e-8,  # relative ΔV/V
        electron_source_angle: float = 1e-4,  # radians
        electron_source_spread: float = 0.7,  # in eV
        cs: float = 2.7,  # spherical aberration (mm)
        cc: float = 3.4,  # chromatic aberration (mm)
        obj_lens_stability: float = 1.6666e-8,  # ΔI/I
    ):
        # Private backing fields:
        self._voltage = None
        self._voltage_stability = None
        self._electron_source_angle = None
        self._electron_source_spread = None
        self._cs = None
        self._cc = None
        self._obj_lens_stability = None

        # Assign via setters to validate
        self.voltage = voltage
        self.voltage_stability = voltage_stability
        self.electron_source_angle = electron_source_angle
        self.electron_source_spread = electron_source_spread
        self.cs = cs
        self.cc = cc
        self.obj_lens_stability = obj_lens_stability

        # Cache for wavelength, so we don’t recalc unless voltage changes
        self._cached_wavelength = None

    # ------------------------------------------------------------
    # voltage (keV)
    # ------------------------------------------------------------
    @property
    def voltage(self) -> float:
        """Acceleration voltage in keV."""
        return self._voltage

    @voltage.setter
    def voltage(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Voltage must be positive (in keV).")
        self._voltage = float(value)
        self._cached_wavelength = None

    # ------------------------------------------------------------
    # voltage_stability (ΔV/V)
    # ------------------------------------------------------------
    @property
    def voltage_stability(self) -> float:
        """Relative voltage stability ΔV/V."""
        return self._voltage_stability

    @voltage_stability.setter
    def voltage_stability(self, value: float) -> None:
        if not (0 <= value <= 1):
            raise ValueError("Voltage stability must be between 0 and 1.")
        self._voltage_stability = float(value)

    # ------------------------------------------------------------
    # electron_source_angle (rad)
    # ------------------------------------------------------------
    @property
    def electron_source_angle(self) -> float:
        """Electron source convergence semi-angle (radians)."""
        return self._electron_source_angle

    @electron_source_angle.setter
    def electron_source_angle(self, value: float) -> None:
        if value < 0:
            raise ValueError("Electron source angle must be non-negative.")
        self._electron_source_angle = float(value)

    # ------------------------------------------------------------
    # electron_source_spread (eV)
    # ------------------------------------------------------------
    @property
    def electron_source_spread(self) -> float:
        """Electron source energy spread (in eV)."""
        return self._electron_source_spread

    @electron_source_spread.setter
    def electron_source_spread(self, value: float) -> None:
        if value < 0:
            raise ValueError("Electron source spread must be non-negative.")
        self._electron_source_spread = float(value)

    # ------------------------------------------------------------
    # cs (spherical aberration, mm)
    # ------------------------------------------------------------
    @property
    def cs(self) -> float:
        """Spherical aberration (Cs) in mm."""
        return self._cs

    @cs.setter
    def cs(self, value: float) -> None:
        if value < 0:
            raise ValueError("Spherical aberration (Cs) must be non-negative.")
        self._cs = float(value)
        self._cached_wavelength = None

    # ------------------------------------------------------------
    # cc (chromatic aberration, mm)
    # ------------------------------------------------------------
    @property
    def cc(self) -> float:
        """Chromatic aberration (Cc) in mm."""
        return self._cc

    @cc.setter
    def cc(self, value: float) -> None:
        if value < 0:
            raise ValueError("Chromatic aberration (Cc) must be non-negative.")
        self._cc = float(value)

    # ------------------------------------------------------------
    # obj_lens_stability (ΔI/I)
    # ------------------------------------------------------------
    @property
    def obj_lens_stability(self) -> float:
        """Objective lens current stability ΔI/I."""
        return self._obj_lens_stability

    @obj_lens_stability.setter
    def obj_lens_stability(self, value: float) -> None:
        if value < 0:
            raise ValueError("Objective lens stability (ΔI/I) must be non-negative.")
        self._obj_lens_stability = float(value)

    # ------------------------------------------------------------
    # Computed property: electron wavelength λ (in Å)
    # ------------------------------------------------------------
    @property
    def wavelength(self) -> float:
        """
        Calculate the electron wavelength λ (in Å) given accelerating voltage in keV.
        λ = h / sqrt(2 m_e eV (1 + eV/(2 m_e c^2))) (converted to Å from m).
        """
        if self._cached_wavelength is None:
            # convert keV → Joules
            eV_joules = self._voltage * 1e3 * Microscope._eV
            numerator = Microscope._h
            denominator = np.sqrt(
                2
                * Microscope._m_e
                * eV_joules
                * (1 + eV_joules / (2 * Microscope._m_e * Microscope._c**2))
            )
            lambda_m = numerator / denominator  # in meters
            lambda_ang = lambda_m * 1e10  # convert m → Å
            self._cached_wavelength = lambda_ang

        return self._cached_wavelength
