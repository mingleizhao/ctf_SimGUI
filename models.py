import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from numpy.typing import ArrayLike, NDArray

# Physical Constants
SPEED_OF_LIGHT = 2.99792458e8  # speed of light in m/s
ELEMENTARY_CHARGE = 1.602176634e-19  # electron charge in coulomb
ELECTRON_MASS = 9.10938356e-31  # electron mass in kg
PLANCK_CONSTANT = 6.62607015e-34  # Planck's constant in joule/hz

# Detector parameters  
DETECTOR_REGISTERS = {
        0: "DDD super resolution counting",
        1: "DDD counting",
        2: "Film",
        3: "CCD"
}
# DQE values are drawn from published curves online.
# From Gatan
K3_DQE_X = [0, 0.5, 1]
K3_DQE_Y = [0.95, 0.71, 0.40]
# SO-163
FILM_DQE_X = [0, 0.25, 0.5, 0.75, 1]
FILM_DQE_Y = [0.37, 0.32, 0.33, 0.22, 0.07]
# TVIPS 224
CCD_DQE_X = [0, 0.25, 0.5, 0.75, 1]
CCD_DQE_Y = [0.37, 0.16, 0.13, 0.1, 0.05]

class Microscope:
    def __init__(
            self, 
            voltage=300,  # in kv
            chromatic_aberration=3.4,  # in mm
            spherical_aberration=2.7,  # in mm
            voltage_stability=3.3333e-8,
            obj_lens_stability=1.6666e-8,
            electron_source_spread=0.7,  # eV
            electron_source_angle=1e-4  # in rad
        ):
        self._voltage = voltage  # in kv       
        self._cc = chromatic_aberration       
        self._cs = spherical_aberration
        self.cs_ang = self._cs * 1e7        
        self._voltage_stability = voltage_stability
        self._obj_lens_stability = obj_lens_stability
        self._electron_source_spread = electron_source_spread
        self._electron_source_angle = electron_source_angle
        self._recompute_parameters()
        self.callbacks = []  # List of listeners

    def add_callback(self, callback):
        """Register a callback to be notified on updates."""
        self.callbacks.append(callback)

    def _notify_callbacks(self):
        for callback in self.callbacks:
            callback(self)  # Pass self as argument

    @property
    def voltage_stability(self):
        return self._voltage_stability
    
    @voltage_stability.setter
    def voltage_stability(self, value):
        self._voltage_stability = value
        self._recompute_parameters()
        self._notify_callbacks()

    @property
    def cc(self):
        return self._cc
    
    @cc.setter
    def cc(self, value):
        self._cc = value
        self._recompute_parameters()
        self._notify_callbacks()

    @property
    def cs(self):
        return self._cs
    
    @cs.setter
    def cs(self, value):
        self._cs = value
        self.cs_ang = self._cs * 1e7  # in angstrom
        self._notify_callbacks()

    @property
    def voltage(self):
        return self._voltage
    
    @voltage.setter
    def voltage(self, value):
        self._voltage = value
        self._recompute_parameters()
        self._notify_callbacks()

    @property
    def obj_lens_stability(self):
        return self._obj_lens_stability
    
    @obj_lens_stability.setter
    def obj_lens_stability(self, value):
        self._obj_lens_stability = value
        self._recompute_parameters()
        self._notify_callbacks()

    @property
    def electron_source_spread(self):
        return self._electron_source_spread
    
    @electron_source_spread.setter
    def electron_source_spread(self, value):
        self._electron_source_spread = value
        self._recompute_parameters()
        self._notify_callbacks()

    @property
    def electron_source_angle(self):
        return self._electron_source_angle
    
    @electron_source_angle.setter
    def electron_source_angle(self, value):
        self._electron_source_angle = value
        self._recompute_parameters()
        self._notify_callbacks()

    def _recompute_parameters(self):
        self.voltage_si = self.voltage * 1000.0  # in volts
        self.cc_ang = self.cc * 1e7  # in angstrom
        self.wavelength = PLANCK_CONSTANT / math.sqrt(2 * ELECTRON_MASS * ELEMENTARY_CHARGE * (self.voltage_si) * (1 + ELEMENTARY_CHARGE * self.voltage_si 
                                    / (2 * ELECTRON_MASS * SPEED_OF_LIGHT ** 2))) * 1e10
        self.focus_spread = self.cc_ang * math.sqrt((self.voltage_stability) ** 2 + 4 * (self.obj_lens_stability) ** 2 
                                        + (self.electron_source_spread / self.voltage_si) ** 2)
        self.Et = lambda x: np.exp(-0.5 * (np.pi * self.wavelength * self.focus_spread) ** 2 * x ** 4)

class Detector:
    DETECTOR_CONFIGS = {}

    def __init__(self, pixel_size=1.0, detector_type='DDD counting'):
        if detector_type not in DETECTOR_REGISTERS.values():
            raise ValueError(f"Invalid detector_type: {detector_type}. Allowed: {list(DETECTOR_REGISTERS.values())}")
        
        self._pixel_size = pixel_size
        self._detector_type = detector_type
        self.dqe_func, self.binning_factor = self._select_detector(detector_type)

        self._recompute_parameters()
        self.callbacks = []  # List of listeners

    @classmethod
    def initialize_detector_configs(cls):
        # precalculated DQE functions for three types of detectors
        ddd_DQE_function = cls.build_polynomial_DQE(K3_DQE_X, K3_DQE_Y)
        film_DQE_function = cls.build_polynomial_DQE(FILM_DQE_X, FILM_DQE_Y)
        ccd_DQE_function = cls.build_polynomial_DQE(CCD_DQE_X, CCD_DQE_Y)
        
        cls.DETECTOR_CONFIGS = {
                "DDD super resolution counting": (ddd_DQE_function, 0.5),
                "DDD counting": (ddd_DQE_function, 1.0),
                "Film": (film_DQE_function, 1.0),
                "CCD": (ccd_DQE_function, 1.0),
        }

    def add_callback(self, callback):
        """Register a callback to be notified on updates."""
        self.callbacks.append(callback)

    def _notify_callbacks(self):
        for callback in self.callbacks:
            callback(self)  # Pass self as argument

    @property
    def pixel_size(self):
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, value):
        if value <= 0:
            raise ValueError("pixel_size must be positive.")
        self._pixel_size = value
        self._recompute_parameters()
        self._notify_callbacks()

    @property
    def detector_type(self):
        return self._detector_type

    @detector_type.setter
    def detector_type(self, value):
        if DETECTOR_REGISTERS[value] != self._detector_type:
            self._detector_type = DETECTOR_REGISTERS[value]
            self.dqe_func, self.binning_factor = self._select_detector(self._detector_type)
            self._recompute_parameters()
            self._notify_callbacks()

    def _recompute_parameters(self):
        self.nyquist = 1 / (2.0 * self.pixel_size * self.binning_factor)
        self.Ed = self._make_detector_envelope()

    def _make_detector_envelope(self):
        def envelope(freq: np.ndarray):
            freq_scaled = freq / self.nyquist
            raw_dqe = self.dqe_func(freq_scaled)
            raw_dqe[freq > self.nyquist] = 0
            return raw_dqe / self.dqe_func(0) #raw_dqe[0] # if max_val > 0 else np.zeros_like(raw_dqe)

        return envelope

    def _select_detector(self, detector_type: str):
        return Detector.DETECTOR_CONFIGS.get(detector_type, Detector.DETECTOR_CONFIGS["DDD counting"])
    
    @staticmethod
    def build_polynomial_DQE(DQE_X: list[float], DQE_Y: list[float], degree: int = None) -> Callable[[np.ndarray], np.ndarray]:
        """
        Fit a polynomial to the given (DQE_X, DQE_Y) data and return a function freq->DQE.
        """
        if degree is None:
            # Automatic choice
            if len(DQE_X) <= 3:
                degree = 2
            else:
                degree = 3

        coeffs = np.polyfit(DQE_X, DQE_Y, degree)  # returns highest degree first
        # Reverse coefficients for np.polynomial.Polynomial
        poly = np.polynomial.Polynomial(coeffs[::-1])

        return lambda x: np.maximum(poly(x), 0)

Detector.initialize_detector_configs()

class CTFBase:
    """
    A base class for CTF calculations with support for callbacks and runtime updates.
    """
    def __init__(self, 
                 amplitude_contrast: float = 0.1,
                 phase_shift_deg: float = 0.0,
                 microscope_param: Microscope = Microscope(),
                 detector_param: Detector = Detector(),
                 include_temporal_env: bool = True,
                 include_spatial_env: bool = True,
                 include_detector_env: bool = True):
        """
        Initialize CTFBase and set up dependencies with Microscope and Detector.
        """
        self._amplitude_contrast = amplitude_contrast
        self._phase_shift_deg = phase_shift_deg
        self.microscope = microscope_param
        self.detector = detector_param
        self._include_temporal_env = include_temporal_env
        self._include_spatial_env = include_spatial_env
        self._include_detector_env = include_detector_env

        # Callbacks
        self.callbacks = []

        # Compute derived parameters
        self._recompute_parameters()

        # Register callbacks from dependent classes
        self.microscope.add_callback(self._on_dependency_update)
        self.detector.add_callback(self._on_dependency_update)

    def _recompute_parameters(self):
        """
        Recompute parameters based on current state.
        """
        self.wavelength = self.microscope.wavelength
        self.cs_ang = self.microscope.cs_ang  # Spherical aberration in Å
        self.electron_source_angle = self.microscope.electron_source_angle
        if self._include_temporal_env:
            self.Et = self.microscope.Et  # Temporal envelope
        else:
            self.Et = lambda x: np.ones_like(x)
        if self._include_detector_env:
            self.Ed = self.detector.Ed    # Detector envelope
        else:
            self.Ed = lambda x: np.ones_like(x)

        self.amplitude_contrast_phase = math.asin(self.amplitude_contrast)
        self.phase_shift_rad = math.radians(self.phase_shift_deg)
        self.cs_part = (math.pi / 2.0) * self.cs_ang * self.wavelength**3

        # Notify callbacks after recomputation
        # self._notify_callbacks()

    def _on_dependency_update(self, _):
        """
        Callback when Microscope or Detector is updated.
        """
        self._recompute_parameters()

    def add_callback(self, callback):
        """Register a callback to be notified on updates."""
        self.callbacks.append(callback)

    def _notify_callbacks(self):
        """Notify all registered callbacks."""
        for callback in self.callbacks:
            callback(self)

    # Properties
    @property
    def amplitude_contrast(self):
        return self._amplitude_contrast

    @amplitude_contrast.setter
    def amplitude_contrast(self, value):
        self._amplitude_contrast = value
        self._recompute_parameters()

    @property
    def phase_shift_deg(self):
        return self._phase_shift_deg

    @phase_shift_deg.setter
    def phase_shift_deg(self, value):
        self._phase_shift_deg = value
        self._recompute_parameters()

    @property
    def include_temporal_env(self):
        return self._include_temporal_env
    
    @include_temporal_env.setter
    def include_temporal_env(self, value):
        self._include_temporal_env = bool(value)
        self._recompute_parameters()
    
    @property
    def include_detector_env(self):
        return self._include_temporal_env
    
    @include_detector_env.setter
    def include_detector_env(self, value):
        self._include_detector_env = bool(value)
        self._recompute_parameters()

    @property
    def include_spatial_env(self):
        return self._include_temporal_env
    
    @include_spatial_env.setter
    def include_spatial_env(self, value):
        self._include_spatial_env = bool(value)
        self._notify_callbacks()
   

    def _common_spatial_envelope_term(self, freq, defocus_Å):
        """
        Compute the spatial envelope term for given frequency and defocus.
        """
        return np.exp(
            - (math.pi * self.electron_source_angle / self.wavelength)**2 *
            (self.cs_ang * self.wavelength**3 * freq**3
             + defocus_Å * self.wavelength * freq)**2
        )
    

class CTF1D(CTFBase):
    """
    A subclass for the 1D CTF scenario, where defocus is a single float (µm).
    """
    def __init__(self,
                 defocus_um: float = 1.0,
                 amplitude_contrast: float = 0.1,
                 phase_shift_deg: float = 0.0,
                 microscope_param: Microscope = Microscope(),
                 detector_param: Detector = Detector(),
                 include_temporal_env: bool = True,
                 include_spatial_env: bool = True,
                 include_detector_env: bool = True):
        super().__init__(
            amplitude_contrast,
            phase_shift_deg,
            microscope_param,
            detector_param
        )
        self._defocus_um = defocus_um  # Store defocus in µm
        self.df_angstrom = self._defocus_um * 1e4  # Convert to Å
        super().add_callback(self._on_dependency_update)
        self._setup_ctf_1d()  # Initial computation

    @property
    def defocus_um(self):
        return self._defocus_um

    @defocus_um.setter
    def defocus_um(self, value):
        self._defocus_um = value
        self.df_angstrom = self._defocus_um * 1e4  # Update derived value
        self._setup_ctf_1d()  # Recompute attributes

    def _setup_ctf_1d(self):
        """
        Create the lambda functions for ctf_1d, envelope, etc.
        """
        df_part = math.pi * self.df_angstrom * self.wavelength

        def phase_function(x):
            return df_part * x**2 - self.cs_part * x**4

        self.ctf_1d = lambda x: -np.sin(
            phase_function(x) + self.amplitude_contrast_phase + self.phase_shift_rad
        )

        def spatial_envelope_1d(x):
            return self._common_spatial_envelope_term(x, self.df_angstrom)

        if self._include_spatial_env:
            self.Es_1d = lambda x: spatial_envelope_1d(x)
        else:
            self.Es_1d = lambda x: np.ones_like(x)
        self.Etotal_1d = lambda x: self.Et(x) * self.Es_1d(x) * self.Ed(x)
        self.dampened_ctf_1d = lambda x: self.ctf_1d(x) * self.Etotal_1d(x)

    def _on_dependency_update(self, _):
        """
        Automatically recompute CTF1D when dependencies update.
        """
        # Recompute shared parameters
        super()._on_dependency_update(_)
        # Recompute 1D-specific parameters
        self._setup_ctf_1d()


class CTF2D(CTFBase):
    """
    A subclass for the 2D CTF scenario, where defocus is (du, dv, da).
    """
    def __init__(self,
                 defocus_tuple: tuple[float, float, float] = (1.0, 1.0, 0.0),  # (du, dv, angle) in µm, deg
                 amplitude_contrast: float = 0.1,
                 phase_shift_deg: float = 0.0,
                 microscope_param: Microscope = Microscope(),
                 detector_param: Detector = Detector(), 
                 include_temporal_env: bool = True,
                 include_spatial_env: bool = True,
                 include_detector_env: bool = True):
        super().__init__(
            amplitude_contrast,
            phase_shift_deg,
            microscope_param,
            detector_param
        )
        self._du_um, self._dv_um, self._da_deg = defocus_tuple
        self._df = (self._du_um + self._dv_um) / 2.0
        self._df_diff = (self._du_um - self._dv_um)
        self.da_rad = math.radians(self._da_deg)
        # define self.
        self._convert_unit_for_defocus()
        super().add_callback(self._on_dependency_update)
        self._setup_ctf_2d()  # Initial computation

    @property
    def df(self):
        return self._df
    
    @df.setter
    def df(self, value):
        self._df = value
        self._du_um = value + self._df_diff / 2.0
        self._dv_um = value - self._df_diff / 2.0
        self._convert_unit_for_defocus()
        self._setup_ctf_2d()  # Recompute attributes

    @property
    def df_diff(self):
        return self._df_diff
    
    @df_diff.setter
    def df_diff(self, value):
        self._df_diff = value
        self._du_um = self._df + value / 2.0
        self._dv_um = self._df - value / 2.0
        self._convert_unit_for_defocus()
        self._setup_ctf_2d()

    @property
    def df_az(self):
        return self._da_deg
    
    @df_az.setter
    def df_az(self, value):
        self._da_deg = value
        self.da_rad = math.radians(value)
        self._setup_ctf_2d()

    def _convert_unit_for_defocus(self):
        self.du_angstrom = self._du_um * 1e4
        self.dv_angstrom = self._dv_um * 1e4
        

    def _setup_ctf_2d(self):
        """
        Create the lambda functions for ctf_2d, envelope, etc.
        """
        delta_df = self.du_angstrom - self.dv_angstrom

        def tilt_angle(x, y):
            return np.arctan2(y, x) - self.da_rad

        def defocus(x, y):
            return 0.5 * (self.du_angstrom + self.dv_angstrom
                          + delta_df * np.cos(2.0 * tilt_angle(x, y)))

        def freq_sq(x, y):
            return x**2 + y**2

        def phase_function(x, y):
            return (math.pi * self.wavelength * defocus(x, y) * freq_sq(x, y)
                    - self.cs_part * freq_sq(x, y)**2)

        self.ctf_2d = lambda x, y: -np.sin(
            phase_function(x, y) + self.amplitude_contrast_phase + self.phase_shift_rad
        )

        def radial_freq(x, y):
            return np.sqrt(freq_sq(x, y))

        def spatial_envelope_2d(x, y):
            return self._common_spatial_envelope_term(
                radial_freq(x, y),
                defocus(x, y)
            )

        if self._include_spatial_env:
            self.Es_2d = lambda x, y: spatial_envelope_2d(x, y)
        else:
            self.Es_2d = lambda x, y: np.ones_like(x + y)
        self.Etotal_2d = lambda x, y: (
            self.Et(radial_freq(x, y)) *
            self.Es_2d(x, y) *
            self.Ed(radial_freq(x, y))
        )
        self.dampened_ctf_2d = lambda x, y: self.ctf_2d(x, y) * self.Etotal_2d(x, y)

    def _on_dependency_update(self, _):
        """
        Automatically recompute CTF2D when dependencies update.
        """
        # Recompute shared parameters
        super()._on_dependency_update(_)
        # Recompute 1D-specific parameters
        self._setup_ctf_2d()

