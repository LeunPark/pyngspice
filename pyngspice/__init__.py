import os
import subprocess
from pathlib import Path
from platform import system
from .utils import get_pkg_config

os_system = system()


def get_ngspice_path() -> Path:
    if os_system == 'Windows':
        possible_paths = [Path(__file__).parent / 'Spice64', Path('C:/Spice64')]
        for path in possible_paths:
            if path.is_dir():
                return path
        raise FileNotFoundError(
            "Ngspice not found on system."
        )

    if os_system == 'Darwin':
        try:
            lib_dirs = get_pkg_config('ngspice').get('library_dirs', [])
            if '/usr/local/lib' in lib_dirs and Path('/usr/local/share/ngspice').is_dir():
                return Path('/usr/local/')
        except ImportError:
            pass

        try:
            return Path(subprocess.check_output(
                ['brew', '--prefix', 'ngspice'],
                stderr=subprocess.DEVNULL, text=True
            ).strip())
        except subprocess.CalledProcessError:
            raise FileNotFoundError(
                "Ngspice not found on system. Run `brew install ngspice`."
            )

    if os_system == 'Linux':
        possible_paths = [Path('/usr'), Path('/usr/local')]
        for path in possible_paths:
            if path.joinpath('share', 'ngspice').is_dir():
                return path

        lib_dirs = get_pkg_config('ngspice').get('library_dirs', [])
        for path in lib_dirs:
            if (Path(path).parent / 'share' / 'ngspice').is_dir():
                return Path(path)

    raise OSError(
        "Unsupported operating system."
    )


NGSPICE_PATH = get_ngspice_path()
os.environ.setdefault('SPICE_LIB_DIR', str(NGSPICE_PATH / 'share' / 'ngspice'))

if os_system == 'Windows':
    import ctypes
    ctypes.windll.LoadLibrary(str(NGSPICE_PATH / 'dll-vs' / 'ngspice.dll'))

from ._pyngspice import (  # noqa: E402
    Shared
)

__all__ = ["Shared"]
__version__ = "0.1.0"
