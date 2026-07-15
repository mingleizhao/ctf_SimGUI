import os
import sys


def resource_path(relative_path: str) -> str:
    """
    Resolve a path to a bundled data file, working both in a normal Python run
    and inside a PyInstaller bundle (onefile or onedir).

    When frozen, PyInstaller extracts `datas` under `sys._MEIPASS`; otherwise we
    resolve relative to the project root (the parent of this file's directory).

    Args:
        relative_path (str): Path relative to the project root, e.g.
            "info/1d_info.html" or "sample_images/sample_image.png".

    Returns:
        str: An absolute path to the resource.
    """
    if getattr(sys, "frozen", False):
        base_dir = sys._MEIPASS  # pylint: disable=no-member,protected-access
    else:
        # utils/ -> project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, relative_path)
