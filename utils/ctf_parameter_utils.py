import numpy as np

PARAMETER_MAP = {
    "voltage": ("microscope.voltage", [0, 1, 2, 3, 4, 5]),
    "voltage_stability": ("microscope.voltage_stability", [0, 1, 2, 3, 4, 5]),
    "es_angle": ("microscope.electron_source_angle", [0, 1, 2, 3, 4, 5]),
    "es_spread": ("microscope.electron_source_spread", [0, 1, 2, 3, 4, 5]),
    "cc": ("microscope.cc", [0, 1, 2, 3, 4, 5]),
    "cs": ("microscope.cs", [0, 1, 2, 3, 4, 5]),
    "obj_stability": ("microscope.obj_lens_stability", [0, 1, 2, 3, 4, 5]),
    "detector": ("envelope.detector_model", [0, 1, 2, 3, 4, 5]),
    "pixel_size": ("envelope.pixel_size", [0, 1, 2, 3, 4, 5]),
    "df": ("defocus", [0, 1, 2, 3, 4, 5]),
    "df_diff": ("defocus_diff", [1, 3, 4, 5]),
    "df_az": ("defocus_az", [1, 3, 4, 5]),
    "ac": ("amplitude_contrast", [0, 1, 2, 3, 4, 5]),
    "phase": ("phase_shift_deg", [0, 1, 2, 3, 4, 5]),
    "temporal_env": ("envelope.include_temporal", [0, 1, 2, 3, 4, 5]),
    "spatial_env": ("envelope.include_spatial", [0, 1, 2, 3, 4, 5]),
    "detector_env": ("envelope.include_detector", [0, 1, 2, 3, 4, 5]),
    "ice": ("envelope.ice_thickness", [2, 3]),  # 1D & 2D
}


def apply_ctf_parameter(key: str, value: float, models: list) -> bool:
    """
    Apply the given CTF parameter (key,value) to all models in the list.
    Returns True if pixel_size was updated (so that controller can resample freq grid).

    Note: 1D’s defocus is stored as defocus_um rather than df, so we handle that mapping here.
    """
    if key not in PARAMETER_MAP:
        return False

    attr_path, idx_list = PARAMETER_MAP[key]
    for i in idx_list:
        target = models[i]
        parts = attr_path.split(".")
        if len(parts) == 2:
            parent_attr, child_attr = parts
            setattr(getattr(target, parent_attr), child_attr, value)
        else:
            setattr(target, attr_path, value)

    # Return whether pixel_size changed
    return key == "pixel_size"


def get_ctf_wrap_func(radio_group, radio_ctf, radio_abs_ctf, radio_ctf_squared):
    """
    Inspect the three radio buttons in `radio_group` and return:
      - `lambda x: x`           if “CTF” is checked
      - `lambda x: np.abs(x)`   if “|CTF|” is checked
      - `lambda x: x**2`        if “CTF²” is checked
    """
    checked = radio_group.checkedButton()
    if checked == radio_ctf:
        return lambda x: x
    elif checked == radio_abs_ctf:
        return lambda x: np.abs(x)
    elif checked == radio_ctf_squared:
        return lambda x: x**2
    else:
        return lambda x: x
