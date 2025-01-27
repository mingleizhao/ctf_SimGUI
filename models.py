import math
from enum import Enum
import numpy as np
from typing import Callable, List
from numpy.typing import NDArray

###############################################################################
# Physical Constants
###############################################################################
SPEED_OF_LIGHT: float = 2.99792458e8      # speed of light in m/s
ELEMENTARY_CHARGE: float = 1.602176634e-19  # electron charge in coulomb
ELECTRON_MASS: float = 9.10938356e-31     # electron mass in kg
PLANCK_CONSTANT: float = 6.62607015e-34   # Planck's constant in joule/hz

###############################################################################
# Detector Classes and Configurations
###############################################################################

class DetectorDQE:
    """
    Provides methods to build polynomial functions for the Detective Quantum Efficiency (DQE) curves.
    """
    @staticmethod
    def build_polynomial_DQE(
        DQE_X: list[float],
        DQE_Y: list[float],
        degree: int | None = None
    ) -> Callable[[NDArray], NDArray]:
        """
        Fit a polynomial to the given (DQE_X, DQE_Y) data and return a function freq->DQE.

        Args:
            DQE_X (list[float]): X coordinates from a measured DQE curve.
            DQE_Y (list[float]): Y coordinates from a measured DQE curve.
            degree (int | None, optional): Polynomial degree. If None, automatically chosen. Defaults to None.

        Returns:
            Callable[[NDArray], NDArray]: A function that maps frequency to DQE values.
        """
        if degree is None:
            degree = 3 if len(DQE_X) > 3 else 2

        coeffs = np.polyfit(DQE_X, DQE_Y, degree)  # returns highest degree first
        poly = np.polynomial.Polynomial(coeffs[::-1])  # reorder for np.polynomial

        return lambda x: np.maximum(poly(x), 0)
    

# Predefined DQE values for various detector types
# DQE values are drawn from published curves online.
# If you want to include more detectors, include the new DQE values and build the dqe_func.
# Also update the DetectorConfigs class below.

# From Gatan
K3_DQE_X: List[float] = [0, 0.5, 1]
K3_DQE_Y: List[float] = [0.95, 0.71, 0.40]

# SO-163
FILM_DQE_X: List[float] = [0, 0.25, 0.5, 0.75, 1]
FILM_DQE_Y: List[float] = [0.37, 0.32, 0.33, 0.22, 0.07]

# TVIPS 224
CCD_DQE_X: List[float] = [0, 0.25, 0.5, 0.75, 1]
CCD_DQE_Y: List[float] = [0.37, 0.16, 0.13, 0.1, 0.05]

# Build polynomial DQE func
DDD_DQE_FUNC = DetectorDQE.build_polynomial_DQE(K3_DQE_X, K3_DQE_Y)
FILM_DQE_FUNC = DetectorDQE.build_polynomial_DQE(FILM_DQE_X, FILM_DQE_Y)
CCD_DQE_FUNC = DetectorDQE.build_polynomial_DQE(CCD_DQE_X, CCD_DQE_Y)

class DetectorConfigs(Enum):
    """
    Enumerates detector configurations, including their DQE functions and binning factors.

    Attributes:
        ID_0: DDD super resolution counting (binning factor: 0.5).
        ID_1: DDD counting (binning factor: 1.0).
        ID_2: Film (binning factor: 1.0).
        ID_3: CCD (binning factor: 1.0).
    """
    ID_0 = {"name": "DDD counting", "dqe": DDD_DQE_FUNC, "binning_factor": 1.0}
    ID_1 = {"name": "DDD super resolution counting", "dqe": DDD_DQE_FUNC, "binning_factor": 0.5}
    ID_2 = {"name": "Film", "dqe": FILM_DQE_FUNC, "binning_factor": 1.0}
    ID_3 = {"name": "CCD", "dqe": CCD_DQE_FUNC, "binning_factor": 1.0}


###############################################################################
# Microscope Class
###############################################################################

class Microscope:
    """Represents microscope parameters and calculates relevant electron-optical properties.

    Attributes:
        _voltage (float): Acceleration voltage in kV.
        _cc (float): Chromatic aberration constant in mm.
        _cs (float): Spherical aberration constant in mm.
        cs_ang (float): Spherical aberration in angstrom.
        _voltage_stability (float): Voltage stability factor.
        _obj_lens_stability (float): Objective lens stability factor.
        _electron_source_spread (float): Electron source spread in eV.
        _electron_source_angle (float): Electron source angle in radians.
        callbacks (List[Callable[[Microscope], None]]): List of listeners to notify on updates.
        wavelength (float): Calculated electron wavelength in angstrom (updated internally).
        focus_spread (float): Computed focus spread factor (updated internally).
        Et (Callable[[NDArray], NDArray]): Temporal envelope function.
    """

    def __init__(
        self,
        voltage: float = 300,
        chromatic_aberration: float = 3.4,
        spherical_aberration: float = 2.7,
        voltage_stability: float = 3.3333e-8,
        obj_lens_stability: float = 1.6666e-8,
        electron_source_spread: float = 0.7,
        electron_source_angle: float = 1e-4
    ) -> None:
        """
        Initializes the microscope parameters and precomputes electron-optical properties.

        Args:
            voltage (float, optional): Acceleration voltage in kV. Defaults to 300.
            chromatic_aberration (float, optional): Chromatic aberration in mm. Defaults to 3.4.
            spherical_aberration (float, optional): Spherical aberration in mm. Defaults to 2.7.
            voltage_stability (float, optional): Voltage stability factor. Defaults to 3.3333e-8.
            obj_lens_stability (float, optional): Objective lens stability factor. Defaults to 1.6666e-8.
            electron_source_spread (float, optional): Electron source spread in eV. Defaults to 0.7.
            electron_source_angle (float, optional): Electron source angle in rad. Defaults to 1e-4.
        """
        self._voltage: float = voltage
        self._cc: float = chromatic_aberration
        self._cs: float = spherical_aberration
        self.cs_ang: float = self._cs * 1e7
        self._voltage_stability: float = voltage_stability
        self._obj_lens_stability: float = obj_lens_stability
        self._electron_source_spread: float = electron_source_spread
        self._electron_source_angle: float = electron_source_angle

        self.callbacks: List[Callable[[Microscope], None]] = []
        self._recompute_parameters()

    def add_callback(self, callback: Callable[['Microscope'], None]) -> None:
        """
        Register a callback to be notified when the microscope parameters change.

        Args:
            callback (Callable[[Microscope], None]): A function that accepts a Microscope instance.
        """
        self.callbacks.append(callback)

    def _notify_callbacks(self) -> None:
        """
        Notify all registered callbacks of an update to the microscope parameters.
        """
        for callback in self.callbacks:
            callback(self)

    @property
    def voltage_stability(self) -> float:
        """float: Voltage stability factor."""
        return self._voltage_stability
    
    @voltage_stability.setter
    def voltage_stability(self, value: float) -> None:
        self._voltage_stability = value
        self._recompute_parameters()
        self._notify_callbacks()

    @property
    def cc(self) -> float:
        """float: Chromatic aberration constant in mm."""
        return self._cc
    
    @cc.setter
    def cc(self, value: float) -> None:
        self._cc = value
        self._recompute_parameters()
        self._notify_callbacks()

    @property
    def cs(self) -> float:
        """float: Spherical aberration constant in mm."""
        return self._cs
    
    @cs.setter
    def cs(self, value: float) -> None:
        self._cs = value
        self.cs_ang = self._cs * 1e7  # Convert mm to angstrom
        self._notify_callbacks()

    @property
    def voltage(self) -> float:
        """float: Acceleration voltage in kV."""
        return self._voltage
    
    @voltage.setter
    def voltage(self, value: float) -> None:
        self._voltage = value
        self._recompute_parameters()
        self._notify_callbacks()

    @property
    def obj_lens_stability(self) -> float:
        """float: Objective lens stability factor."""
        return self._obj_lens_stability
    
    @obj_lens_stability.setter
    def obj_lens_stability(self, value: float) -> None:
        self._obj_lens_stability = value
        self._recompute_parameters()
        self._notify_callbacks()

    @property
    def electron_source_spread(self) -> float:
        """float: Electron source spread in eV."""
        return self._electron_source_spread
    
    @electron_source_spread.setter
    def electron_source_spread(self, value: float) -> None:
        self._electron_source_spread = value
        self._recompute_parameters()
        self._notify_callbacks()

    @property
    def electron_source_angle(self) -> float:
        """float: Electron source angle in radians."""
        return self._electron_source_angle
    
    @electron_source_angle.setter
    def electron_source_angle(self, value: float) -> None:
        self._electron_source_angle = value
        self._recompute_parameters()
        self._notify_callbacks()

    def _recompute_parameters(self) -> None:
        """
        Recompute electron-optical properties based on the current microscope parameters.
        """
        voltage_si: float = self.voltage * 1000.0  # Convert kV to volts
        self.cc_ang: float = self.cc * 1e7         # Convert mm to angstrom

        # Calculate electron wavelength in angstrom
        self.wavelength: float = (
            PLANCK_CONSTANT /
            math.sqrt(
                2 * ELECTRON_MASS * ELEMENTARY_CHARGE * voltage_si *
                (1 + (ELEMENTARY_CHARGE * voltage_si) / (2 * ELECTRON_MASS * SPEED_OF_LIGHT**2))
            )
        ) * 1e10

        # Calculate focus spread
        self.focus_spread: float = (
            self.cc_ang *
            math.sqrt(
                (self.voltage_stability) ** 2
                + 4 * (self.obj_lens_stability) ** 2
                + (self.electron_source_spread / voltage_si) ** 2
            )
        )

        # Temporal envelope function
        self.Et: Callable[[NDArray], NDArray] = lambda x: np.exp(
            -0.5 * (np.pi * self.wavelength * self.focus_spread) ** 2 * x ** 4
        )


###############################################################################
# CTF Classes
###############################################################################

class CTFBase:
    """
    A base class for CTF calculations with support for callbacks and runtime updates.

    Attributes:
        _pixel_size (float): Pixel size in angstroms (Å).
        _amplitude_contrast (float): Amplitude contrast factor.
        _phase_shift_deg (float): Additional phase shift in degrees.
        microscope (Microscope): Instance of the Microscope class.
        detector_config (DetectorConfigs): Detector configuration as a DetectorConfigs Enum.
        nyquist (float): Nyquist frequency based on the pixel size and detector configuration.
        _include_temporal_env (bool): Whether to include the temporal envelope in calculations.
        _include_spatial_env (bool): Whether to include the spatial envelope in calculations.
        _include_detector_env (bool): Whether to include the detector envelope in calculations.
        callbacks (List[Callable[[CTFBase], None]]): Callbacks to notify on updates.   
    """

    def __init__(
        self,
        pixel_size: float = 1.0,
        amplitude_contrast: float = 0.1,
        phase_shift_deg: float = 0.0,
        microscope_param: Microscope = Microscope(),
        detector_config: DetectorConfigs = DetectorConfigs.ID_0,
        include_temporal_env: bool = True,
        include_spatial_env: bool = True,
        include_detector_env: bool = True
    ) -> None:
        """
        Initialize the CTFBase class with default or specified parameters.

        Args:
            pixel_size (float, optional): Pixel size in angstroms (Å). Defaults to 1.0.
            amplitude_contrast (float, optional): Amplitude contrast factor. Defaults to 0.1.
            phase_shift_deg (float, optional): Additional phase shift in degrees. Defaults to 0.0.
            microscope_param (Microscope, optional): Microscope instance. Defaults to a new Microscope.
            detector_config (DetectorConfigs, optional): Detector configuration. Defaults to DetectorConfigs.ID_1.
            include_temporal_env (bool, optional): Include temporal envelope. Defaults to True.
            include_spatial_env (bool, optional): Include spatial envelope. Defaults to True.
            include_detector_env (bool, optional): Include detector envelope. Defaults to True.
        """
        self._pixel_size: float = pixel_size
        self._amplitude_contrast: float = amplitude_contrast
        self._phase_shift_deg: float = phase_shift_deg
        self.microscope: Microscope = microscope_param
        self.detector_config: DetectorConfigs = detector_config
        self._include_temporal_env: bool = include_temporal_env
        self._include_spatial_env: bool = include_spatial_env
        self._include_detector_env: bool = include_detector_env

        self.callbacks: List[Callable[['CTFBase'], None]] = []

        # Register callbacks from dependent classes
        self.microscope.add_callback(self._on_dependency_update)

        self._recompute_parameters()

    def _recompute_parameters(self) -> None:
        """
        Recompute dependent parameters based on current attributes. 
        """
        self.nyquist: float = 1 / (2.0 * self._pixel_size * self.detector_config.value["binning_factor"])
        self.dqe_func: Callable[[NDArray], NDArray] = self.detector_config.value["dqe"] 
        
        if self._include_temporal_env:
            self.Et: Callable[[NDArray], NDArray] = self.microscope.Et
        else:
            self.Et = lambda x: np.ones_like(x)
        if self._include_detector_env:
            self.Ed: Callable[[NDArray], NDArray] = self._make_detector_envelope()
        else:
            self.Ed = lambda x: np.ones_like(x)

        self.amplitude_contrast_phase: float = math.asin(self.amplitude_contrast)
        self.phase_shift_rad: float = math.radians(self._phase_shift_deg)
        self.cs_part: float = (math.pi / 2.0) * self.microscope.cs_ang * self.microscope.wavelength**3

    def _make_detector_envelope(self) -> Callable[[NDArray], NDArray]:
        """
        Create the envelope function based on the DQE function, scaled by the Nyquist frequency.

        Returns:
            Callable[[NDArray], NDArray]: Envelope function for the detector.
        """
        def envelope(freq: NDArray) -> NDArray:
            freq_scaled = freq / self.nyquist
            raw_dqe = self.dqe_func(freq_scaled)
            # Zero out dqe when frequencies are beyond Nyquist
            raw_dqe[freq > self.nyquist] = 0
            # Normalize by DQE at zero freq or a reference
            return raw_dqe / self.dqe_func(0) 
        return envelope

    def _on_dependency_update(self, _) -> None:
        """
        Update this class's parameters when the Microscope or Detector changes.
        """
        self._recompute_parameters()

    def add_callback(self, callback: Callable[['CTFBase'], None]) -> None:
        """
        Register a callback to be notified on updates.

        Args:
            callback (Callable[[CTFBase], None]): A function that accepts a CTFBase instance.
        """
        self.callbacks.append(callback)

    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks that parameters have changed."""
        for callback in self.callbacks:
            callback(self)

    @property
    def pixel_size(self) -> float:
        """float: Pixel size in Å."""
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, value: float) -> None:
        self._pixel_size = value
        self._recompute_parameters()
    
    @property
    def detector(self) -> str:
        """str: The configuration of the detector."""
        return self.detector_config.value["name"]

    @detector.setter
    def detector(self, value: int) -> None:
        """Set the detector configure based on id

        Args:
            value (int): The key of detector configure as defined in DetectorConfigs

        Raises:
            ValueError: If the detector type has not been registered
        """
        id = f"ID_{value}"
        if DetectorConfigs[id] not in DetectorConfigs:
            raise ValueError(f"Invalid detector configure key: {value}")
        if DetectorConfigs[id] != self.detector_config:
            self.detector_config = DetectorConfigs[id]
            self._recompute_parameters()

    @property
    def amplitude_contrast(self) -> float:
        """float: Amplitude contrast factor."""
        return self._amplitude_contrast

    @amplitude_contrast.setter
    def amplitude_contrast(self, value: float) -> None:
        self._amplitude_contrast = value
        self._recompute_parameters()

    @property
    def phase_shift_deg(self) -> float:
        """float: Additional phase shift in degrees."""
        return self._phase_shift_deg

    @phase_shift_deg.setter
    def phase_shift_deg(self, value: float) -> None:
        self._phase_shift_deg = value
        self._recompute_parameters()

    @property
    def include_temporal_env(self) -> bool:
        """bool: Flag to include the temporal envelope."""
        return self._include_temporal_env
    
    @include_temporal_env.setter
    def include_temporal_env(self, value: bool) -> None:
        self._include_temporal_env = bool(value)
        self._recompute_parameters()
    
    @property
    def include_detector_env(self) -> bool:
        """bool: Flag to include the detector envelope."""
        return self._include_detector_env
    
    @include_detector_env.setter
    def include_detector_env(self, value: bool) -> None:
        self._include_detector_env = bool(value)
        self._recompute_parameters()

    @property
    def include_spatial_env(self) -> bool:
        """bool: Flag to include the spatial envelope."""
        return self._include_spatial_env
    
    @include_spatial_env.setter
    def include_spatial_env(self, value: bool) -> None:
        self._include_spatial_env = bool(value)
        self._notify_callbacks()

    def _common_spatial_envelope_term(self, freq: float | NDArray, defocus_Å: float) -> NDArray:
        """
        Compute the spatial envelope term for a given frequency and defocus.

        Args:
            freq (float | NDArray): Spatial frequency (1/Å).
            defocus_Å (float): Defocus in angstrom.

        Returns:
            NDArray: Envelope multiplier at the given frequency.
        """
        return np.exp(
            - (math.pi * self.microscope.electron_source_angle / self.microscope.wavelength)**2 *
            (self.microscope.cs_ang * self.microscope.wavelength**3 * freq**3
             + defocus_Å * self.microscope.wavelength * freq)**2
        )


class CTF1D(CTFBase):
    """
    A subclass for the 1D CTF scenario, where defocus is a single float (µm).
    """

    def __init__(
        self,
        defocus_um: float = 1.0,
        **kwargs
    ) -> None:
        """
        Initialize a 1D CTF with a single defocus parameter.

        Args:
            defocus_um (float, optional): Defocus in micrometers. Defaults to 1.0.
            kwargs: Additional keyword arguments passed to the CTFBase constructor.
        """
        super().__init__(**kwargs)
        self._defocus_um: float = defocus_um
        self.df_angstrom: float = self._defocus_um * 1e4
        super().add_callback(self._on_dependency_update)
        self._setup_ctf_1d()

    @property
    def defocus_um(self) -> float:
        """float: Defocus in micrometers."""
        return self._defocus_um

    @defocus_um.setter
    def defocus_um(self, value: float) -> None:
        self._defocus_um = value
        self.df_angstrom = self._defocus_um * 1e4
        self._setup_ctf_1d()
        self._notify_callbacks()

    def _setup_ctf_1d(self) -> None:
        """
        Create the lambda functions for 1D-CTF, envelopes, and dampened CTF.
        """
        df_part: float = math.pi * self.df_angstrom * self.microscope.wavelength

        def phase_function(x: NDArray) -> NDArray:
            return df_part * x**2 - self.cs_part * x**4

        self.ctf_1d = lambda x: -np.sin(
            phase_function(x) + self.amplitude_contrast_phase + self.phase_shift_rad
        )

        def spatial_envelope_1d(x: NDArray) -> NDArray:
            return self._common_spatial_envelope_term(x, self.df_angstrom)

        if self.include_spatial_env:
            self.Es_1d = spatial_envelope_1d
        else:
            self.Es_1d = lambda x: np.ones_like(x)

        self.Etotal_1d = lambda x: self.Et(x) * self.Es_1d(x) * self.Ed(x)
        self.dampened_ctf_1d = lambda x: self.ctf_1d(x) * self.Etotal_1d(x)

    def _on_dependency_update(self, _) -> None:
        """
        Automatically recompute the 1D CTF when dependencies update.
        """
        super()._on_dependency_update(_)
        self._setup_ctf_1d()


class CTF2D(CTFBase):
    """
    A subclass for the 2D CTF scenario, where defocus is a tuple (du, dv, angle).
    """

    def __init__(
        self,
        defocus_tuple: tuple[float, float, float] = (1.0, 1.0, 0.0),
        **kwargs
    ) -> None:
        """
        Initialize a 2D CTF with defocus parameters for astigmatism and angle.

        Args:
            defocus_tuple (tuple[float, float, float], optional): (du, dv, angle) in (µm, µm, deg).
                du = major axis defocus, dv = minor axis defocus, angle = azimuthal angle in degrees.
                Defaults to (1.0, 1.0, 0.0).
            kwargs: Additional keyword arguments passed to the CTFBase constructor.
        """
        super().__init__(**kwargs)
        du_um, dv_um, da_deg = defocus_tuple
        self._du_um: float = du_um
        self._dv_um: float = dv_um
        self._da_deg: float = da_deg

        self._df: float = (self._du_um + self._dv_um) / 2.0
        self._df_diff: float = (self._du_um - self._dv_um)
        self.da_rad: float = math.radians(self._da_deg)

        self._convert_unit_for_defocus()
        super().add_callback(self._on_dependency_update)
        self._setup_ctf_2d()

    @property
    def df(self) -> float:
        """float: Average defocus in micrometers (µm)."""
        return self._df
    
    @df.setter
    def df(self, value: float) -> None:
        self._df = value
        self._du_um = value + self._df_diff / 2.0
        self._dv_um = value - self._df_diff / 2.0
        self._convert_unit_for_defocus()
        self._setup_ctf_2d()
        self._notify_callbacks()

    @property
    def df_diff(self) -> float:
        """float: Difference in defocus between major and minor axes (µm)."""
        return self._df_diff
    
    @df_diff.setter
    def df_diff(self, value: float) -> None:
        self._df_diff = value
        self._du_um = self._df + value / 2.0
        self._dv_um = self._df - value / 2.0
        self._convert_unit_for_defocus()
        self._setup_ctf_2d()
        self._notify_callbacks()

    @property
    def df_az(self) -> float:
        """float: Azimuthal angle for the defocus (in degrees)."""
        return self._da_deg
    
    @df_az.setter
    def df_az(self, value: float) -> None:
        self._da_deg = value
        self.da_rad = math.radians(value)
        self._setup_ctf_2d()
        self._notify_callbacks()

    def _convert_unit_for_defocus(self) -> None:
        """Convert defocus in µm to angstrom (Å)."""
        self.du_angstrom: float = self._du_um * 1e4
        self.dv_angstrom: float = self._dv_um * 1e4

    def _setup_ctf_2d(self) -> None:
        """
        Create the lambda functions for 2D-CTF, envelopes, etc.
        """
        delta_df: float = self.du_angstrom - self.dv_angstrom

        def tilt_angle(x: NDArray, y: NDArray) -> NDArray:
            return np.arctan2(y, x) - self.da_rad

        def defocus(x: NDArray, y: NDArray) -> NDArray:
            return 0.5 * (
                self.du_angstrom + self.dv_angstrom
                + delta_df * np.cos(2.0 * tilt_angle(x, y))
            )

        def freq_sq(x: NDArray, y: NDArray) -> NDArray:
            return x**2 + y**2

        def phase_function(x: NDArray, y: NDArray) -> NDArray:
            """
            Phase shift as a function of x,y frequencies and the defocus.
            """
            return (
                math.pi * self.microscope.wavelength * defocus(x, y) * freq_sq(x, y)
                - self.cs_part * freq_sq(x, y)**2
            )

        self.ctf_2d = lambda x, y: -np.sin(
            phase_function(x, y) + self.amplitude_contrast_phase + self.phase_shift_rad
        )

        def radial_freq(x: NDArray, y: NDArray) -> NDArray:
            return np.sqrt(freq_sq(x, y))

        def spatial_envelope_2d(x: NDArray, y: NDArray) -> NDArray:
            return self._common_spatial_envelope_term(radial_freq(x, y), defocus(x, y))

        if self._include_spatial_env:
            self.Es_2d = spatial_envelope_2d
        else:
            self.Es_2d = lambda x, y: np.ones_like(x + y)

        self.Etotal_2d = lambda x, y: (
            self.Et(radial_freq(x, y)) *
            self.Es_2d(x, y) *
            self.Ed(radial_freq(x, y))
        )

        self.dampened_ctf_2d = lambda x, y: self.ctf_2d(x, y) * self.Etotal_2d(x, y)

    def _on_dependency_update(self, _) -> None:
        """
        Automatically recompute the 2D CTF when dependencies update.
        """
        super()._on_dependency_update(_)
        self._setup_ctf_2d()


class CTFIce1D(CTF1D):
    """
    A subclass of CTF1D for simulating the Contrast Transfer Function (CTF) in the presence of ice in 1D.

    Attributes:
        ice_thickness_ang (float): Ice thickness in nanometers.
        ctf_ice (Callable[[NDArray], NDArray]): Function for the CTF with ice effects.
        dampened_ctf_ice (Callable[[NDArray], NDArray]): Function for the dampened CTF with ice effects.
    """

    def __init__(self, ice_thickness: float = 50.0, **kwargs) -> None:
        """
        Initialize the CTFIce1D instance with an optional ice thickness.

        Args:
            ice_thickness (float, optional): Thickness of the ice layer in nanometers. Defaults to 50.
            **kwargs: Additional arguments passed to the CTF1D parent class.
        """
        super().__init__(**kwargs)
        self._ice_thickness_nm: float = ice_thickness
        self.ice_thickness_ang: float = ice_thickness * 10.0  # Convert nm to angstroms
        super().add_callback(self._on_dependency_update)
        self._setup_ctf_ice_1d()

    @property
    def ice_thickness(self) -> float:
        """float: Ice thickness in nanometers."""
        return self._ice_thickness_nm

    @ice_thickness.setter
    def ice_thickness(self, value: float) -> None:
        """
        Update the ice thickness and recompute the CTF with ice effects.

        Args:
            value (float): New ice thickness in nanometers.
        """
        self._ice_thickness_nm = value
        self.ice_thickness_ang = value * 10.0  # Convert nm to angstroms
        self._setup_ctf_ice_1d()

    def _setup_ctf_ice_1d(self) -> None:
        """Define the CTF and dampened CTF with ice effects in 1D."""
        def coeff(x: NDArray) -> NDArray:
            return math.pi * self.microscope.wavelength * x**2

        def phase(x: NDArray) -> NDArray:
            return (
                self.amplitude_contrast_phase +
                self.phase_shift_rad -
                self.cs_part * x**4 +
                self.df_angstrom * coeff(x)
            )

        self.ctf_ice = lambda x: (
            -2 / coeff(x) * np.sin(coeff(x) * self.ice_thickness_ang / 2.0)
            * np.sin(phase(x)) / self.ice_thickness_ang
        )
        self.dampened_ctf_ice = lambda x: self.ctf_ice(x) * self.Etotal_1d(x)

    def _on_dependency_update(self, _: CTF1D) -> None:
        """
        Automatically recompute the CTF with ice effects when dependencies update.

        Args:
            _ (CTF1D): The updated CTF1D instance triggering the callback.
        """
        super()._on_dependency_update(_)
        self._setup_ctf_ice_1d()


class CTFIce2D(CTF2D):
    """
    A subclass of CTF2D for simulating the Contrast Transfer Function (CTF) in the presence of uniform ice in 2D.

    Attributes:
        ice_thickness_ang (float): Ice thickness in nanometers.
        ctf_ice (Callable[[NDArray, NDArray], NDArray]): Function for the CTF with ice effects.
        dampened_ctf_ice (Callable[[NDArray, NDArray], NDArray]): Function for the dampened CTF with ice effects.
    """

    def __init__(self, ice_thickness: float = 50.0, **kwargs) -> None:
        """
        Initialize the CTFIce2D instance with an optional ice thickness.

        Args:
            ice_thickness (float, optional): Thickness of the ice layer in nanometers. Defaults to 50.
            **kwargs: Additional arguments passed to the CTF2D parent class.
        """
        super().__init__(**kwargs)
        self._ice_thickness_nm: float = ice_thickness
        self.ice_thickness_ang: float = ice_thickness * 10.0  # Convert nm to angstroms
        super().add_callback(self._on_dependency_update)
        self._setup_ctf_ice_2d()

    @property
    def ice_thickness(self) -> float:
        """float: Ice thickness in nanometers."""
        return self._ice_thickness_nm

    @ice_thickness.setter
    def ice_thickness(self, value: float) -> None:
        """
        Update the ice thickness and recompute the CTF with ice effects.

        Args:
            value (float): New ice thickness in nanometers.
        """
        self._ice_thickness_nm = value
        self.ice_thickness_ang = value * 10.0  # Convert nm to angstroms
        self._setup_ctf_ice_2d()

    def _setup_ctf_ice_2d(self) -> None:
        """Define the CTF and dampened CTF with ice effects in 2D."""
        delta_df: float = self.du_angstrom - self.dv_angstrom

        def tilt_angle(x: NDArray, y: NDArray) -> NDArray:
            return np.arctan2(y, x) - self.da_rad

        def defocus(x: NDArray, y: NDArray) -> NDArray:
            return 0.5 * (
                self.du_angstrom + self.dv_angstrom
                + delta_df * np.cos(2.0 * tilt_angle(x, y))
            )

        def freq_sq(x: NDArray, y: NDArray) -> NDArray:
            return x**2 + y**2

        def coeff(x: NDArray, y: NDArray) -> NDArray:
            return math.pi * self.microscope.wavelength * freq_sq(x, y)

        def phase(x: NDArray, y: NDArray) -> NDArray:
            return (
                self.amplitude_contrast_phase +
                self.phase_shift_rad -
                self.cs_part * freq_sq(x, y)**2 +
                defocus(x, y) * coeff(x, y)
            )

        self.ctf_ice = lambda x, y: (
            -2 / coeff(x, y) / self.ice_thickness_ang
            * np.sin(coeff(x, y) * self.ice_thickness_ang / 2.0)
            * np.sin(phase(x, y))
        )
        self.dampened_ctf_ice = lambda x, y: self.ctf_ice(x, y) * self.Etotal_2d(x, y)

    def _on_dependency_update(self, _: CTF2D) -> None:
        """
        Automatically recompute the CTF with ice effects when dependencies update.

        Args:
            _ (CTF2D): The updated CTF2D instance triggering the callback.
        """
        super()._on_dependency_update(_)
        self._setup_ctf_ice_2d()