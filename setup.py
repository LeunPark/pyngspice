import subprocess
from platform import system as _system
from setuptools import setup, Extension


class get_numpy_include:
    def __str__(self):
        # to implement lazy import effect
        from numpy import get_include
        return get_include()


def get_pkg_config(*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    # try:
    output = subprocess.check_output(
        ["pkg-config", "--libs"] + list(packages),  # TODO: , "--cflags"
        stderr=subprocess.DEVNULL,
    ).decode()
    # except subprocess.CalledProcessError:
    #     return kw

    for token in output.split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return kw


system = _system()
install_cmds = {
    'Darwin': 'brew install ngspice',
    'Linux': 'apt-get install libngspice0-dev',
}
if system in ('Darwin', 'Linux'):
    try:
        pkg_config = get_pkg_config('ngspice')
    except subprocess.CalledProcessError:
        raise ImportError(
            f"Ngspice not found on system. Run `{install_cmds[system]}`."
        )
else:  # TODO: On Windows
    pkg_config = {}
pkg_config.setdefault('include_dirs', []).append(get_numpy_include())

setup(
    name="pyngspice",
    version="0.0.1",
    author="Leun Park",
    setup_requires=['numpy'],
    ext_modules=[
        Extension(
            "pyngspice._pyngspice",
            sources=["pyngspice/_pyngspice.c"],
            **pkg_config,
        )
    ],
    packages=["pyngspice"],
    python_requires=">=3.5",
)
