#!/bin/bash

if ! sudo -v 2>/dev/null; then
    echo "sudo privileges are required. The script will now exit."
    exit 1
fi

OS=$(uname -s)
CC="gcc"
CXX="g++"
LDFLAGS=""

case "$OS" in
  Linux)
    LDFLAGS+="-s"
    ;;
  Darwin)
    if gcc_path=$(brew --prefix gcc 2>/dev/null) && [ -d "$gcc_path" ]; then
      gcc_executable=$(find "$gcc_path"/bin -name 'gcc-[0-9]*' -print -quit)
      if [ -n "$gcc_executable" ]; then
        version=$(basename "$gcc_executable" | grep -o -E '[0-9]+(\.[0-9]+)*')
        CC+="-$version"
        CXX+="-$version"
      else
        echo "[ERROR] No GCC executable found in the Homebrew GCC path."
        exit 1
      fi
    else
      echo "[ERROR] GCC is not installed via Homebrew."
      exit 1
    fi
    ;;
esac

# Download and extract ngspice
curl -L -o ngspice-42.tar.gz https://sourceforge.net/projects/ngspice/files/ng-spice-rework/42/ngspice-42.tar.gz
tar -zxvf ngspice-42.tar.gz
cd ngspice-42

if [ ! -d "release" ]; then
  mkdir release
  if [ $? -ne 0 ]; then  echo "[ERROR] mkdir release failed"; exit 1 ; fi
fi
cd release

CONFIGURE_FLAGS="--with-ngshared --enable-xspice --disable-debug --enable-openmp --enable-klu"
CFLAGS="-m64 -O3 -I/usr/local/opt/ncurses/include -I/usr/local/include"
LDFLAGS="-m64 -L/usr/local/opt/ncurses/lib -L/usr/local/lib $LDFLAGS"
../configure $CONFIGURE_FLAGS CC="$CC" CXX="$CXX" CFLAGS="$CFLAGS" LDFLAGS="$LDFLAGS"
if [ $? -ne 0 ]; then  echo "[ERROR] Configure failed"; exit 1 ; fi

make clean -j8
if [ $? -ne 0 ]; then  echo "[ERROR] Make clean failed"; exit 1 ; fi

echo "Compiling (see make.log)"
make -j8 | tee make.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then  echo "[ERROR] Make failed"; exit 1 ; fi

echo "installing (see make_install.log)"
sudo make install | tee make_install.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then  echo "[ERROR] Make install failed"; exit 1 ; fi

if [ "$OS" == "Linux" ]; then
  sudo ldconfig
fi

echo
echo "Installation success!"
exit 0
