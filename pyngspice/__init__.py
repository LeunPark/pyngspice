import os
import subprocess
from pathlib import Path
from platform import system as _system

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
    try:
        output = subprocess.check_output(
            ['brew', '--prefix', 'ngspice'],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        raise ImportError(
            "Ngspice not found on system. Run `brew install ngspice`."
        )

    NGSPICE_PATH = Path(output)

if NGSPICE_PATH and 'SPICE_LIB_DIR' not in os.environ:
    os.environ['SPICE_LIB_DIR'] = str(NGSPICE_PATH.joinpath('share', 'ngspice'))


from ._pyngspice import (
    # Shared as _Shared,
    Shared
)

__all__ = ["Shared"]
