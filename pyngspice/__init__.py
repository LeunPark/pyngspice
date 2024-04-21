import os
import subprocess
from pathlib import Path
from platform import system as _system
from .utils import get_pkg_config

NGSPICE_PATH = None

system = _system()
if system == 'Windows':
    possible_paths = [Path(__file__).parent.joinpath('Spice64'), Path('C:/Spice64')]
    for path in possible_paths:
        if path.is_dir():
            NGSPICE_PATH = path
            break
    else:
        # 없으면 다운로드
        # https://jaist.dl.sourceforge.net/project/ngspice/ng-spice-rework/42/ngspice-42_dll_64.7z
        raise ImportError(
            "Ngspice not found on system. Post-installation proceed."
        )

    import ctypes
    ctypes.windll.LoadLibrary(str(NGSPICE_PATH / 'dll-vs' / 'ngspice.dll'))
elif system == 'Darwin':
    def _get_ngspice_path():
        try:
            lib_dirs = get_pkg_config('ngspice').get('library_dirs', [])
            if '/usr/local/lib' in lib_dirs and Path('/usr/local/share/ngspice').is_dir():
                return '/usr/local/'
        except ImportError:
            pass

        try:
            return subprocess.check_output(
                ['brew', '--prefix', 'ngspice'],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
        except subprocess.CalledProcessError:
            raise ImportError(
                "Ngspice not found on system. Run `brew install ngspice`."
            )

    NGSPICE_PATH = Path(_get_ngspice_path())
elif system == 'Linux':
    possible_paths = [Path('/usr'), Path('/usr/local')]
    for path in possible_paths:
        if path.joinpath('share', 'ngspice').is_dir():
            NGSPICE_PATH = path
            break
    else:
        lib_dirs = get_pkg_config('ngspice').get('library_dirs', [])
        for path in lib_dirs:
            if Path(path).parent.joinpath('share', 'ngspice').is_dir():
                NGSPICE_PATH = Path(path)
                break

if NGSPICE_PATH and 'SPICE_LIB_DIR' not in os.environ:
    os.environ['SPICE_LIB_DIR'] = str(NGSPICE_PATH / 'share' / 'ngspice')


from ._pyngspice import (
    Shared
)

__all__ = ["Shared"]
__version__ = "0.1.0"
