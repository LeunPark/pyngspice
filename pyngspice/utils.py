import os
import subprocess
from platform import system
from typing import Dict, List

os.environ['PKG_CONFIG_PATH'] = '/usr/local/lib/pkgconfig:' + os.getenv('PKG_CONFIG_PATH', '')

_install_cmds = {
    'Darwin': 'brew install ngspice',
    'Linux': 'apt-get install libngspice0-dev',
}


def get_pkg_config(*packages, **kw) -> Dict[str, List[str]]:
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    try:
        output = subprocess.check_output(
            ["pkg-config", "--libs", "--cflags"] + list(packages),
            stderr=subprocess.DEVNULL,
        ).decode()
    except FileNotFoundError:
        raise FileNotFoundError(
            "pkg-config is not installed. Please install it."
        )
    except subprocess.CalledProcessError:
        raise ImportError(
            f"Ngspice not found on system. Run `{_install_cmds[system()]}`."
        )

    for token in output.split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return kw
