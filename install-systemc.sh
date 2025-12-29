#!/bin/bash

# SystemC installation script

echo "=========================================="
echo "SystemC Installation Script"
echo "=========================================="

SYSTEMC_VERSION="2.3.3"
SYSTEMC_DIR="/opt/systemc"

START_DIR=$(pwd)

mkdir -p ${SYSTEMC_DIR}
cd ${SYSTEMC_DIR} || {
    echo "Failed to change directory to ${SYSTEMC_DIR}"
    exit 1
}

wget http://accellera.org/images/downloads/standards/systemc/systemc-2.3.3.tar.gz \
    -O systemc-${SYSTEMC_VERSION}.tar.gz
tar -xzf systemc-${SYSTEMC_VERSION}.tar.gz

cd systemc-${SYSTEMC_VERSION} || {
    echo "Failed to change directory to systemc-${SYSTEMC_VERSION}"
    exit 1
}
mkdir build
cd build || {
    echo "Failed to change directory to build"
    exit 1
}

SYSTEMC_HOME=${SYSTEMC_DIR}/systemc-${SYSTEMC_VERSION}-install

cmake \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${SYSTEMC_HOME} \
    -DINSTALL_TO_LIB_TARGET_ARCH_DIR=ON \
    -DINSTALL_LIB_TARGET_ARCH_SYMLINK=ON \
    ..

make
make install

cd "${START_DIR}" || {
    echo "Failed to return to starting directory ${START_DIR}"
    exit 1
}

echo ""
echo "=========================================="
echo "SystemC ${SYSTEMC_VERSION} installed successfully!"
echo "Installed at: ${SYSTEMC_DIR}/systemc-${SYSTEMC_VERSION}-install"
echo "=========================================="
echo ""

{
    echo "# SystemC install path"
    echo "export SYSTEMC_HOME=${SYSTEMC_HOME}"
    echo "if [ -z \"$LD_LIBRARY_PATH\" ]; then"
    echo "    export LD_LIBRARY_PATH=${SYSTEMC_HOME}/lib-linux64"
    echo "else"
    echo "    export LD_LIBRARY_PATH=${SYSTEMC_HOME}/lib-linux64:$LD_LIBRARY_PATH"
    echo "fi"
} >>~/.bashrc
