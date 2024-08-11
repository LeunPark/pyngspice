import os
import subprocess
from platform import system
from setuptools import setup, Extension

os.environ['PKG_CONFIG_PATH'] = '/usr/local/lib/pkgconfig:' + os.getenv('PKG_CONFIG_PATH', '')


class get_numpy_include:
    def __str__(self):
        # to implement lazy import effect
        from numpy import get_include
        return get_include()


def get_pkg_config(*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    try:
        output = subprocess.check_output(
            ["pkg-config", "--libs", "--cflags"] + list(packages),
            stderr=subprocess.DEVNULL,
        ).decode()
    except FileNotFoundError:
        raise ImportError(
            "pkg-config is not installed. Please install it."
        )

    for token in output.split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return kw


os_system = system()
if os_system in ('Darwin', 'Linux'):
    try:
        pkg_config = get_pkg_config('ngspice')
    except subprocess.CalledProcessError:
        install_cmd = 'brew install ngspice' if os_system == 'Darwin' else 'apt install ngspice libngspice0-dev'
        raise ImportError(
            f"Ngspice not found on system. Run `{install_cmd}`."
        )
elif os_system == 'Windows':
    pkg_config = {
        'include_dirs': ['C:/Spice64/include'],
        'library_dirs': ['C:/Spice64/lib/lib-vs'],
        'libraries': ['ngspice'],
    }
else:
    pkg_config = {}

pkg_config.setdefault('include_dirs', []).append(get_numpy_include())

setup(
    # setup_requires=['numpy'],
    ext_modules=[
        Extension(
            "pyngspice._pyngspice",
            sources=["pyngspice/_pyngspice.c"],
            **pkg_config,
        )
    ],
    packages=["pyngspice"],
)
