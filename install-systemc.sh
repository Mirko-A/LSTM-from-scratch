#!/bin/bash

# SystemC installation script

set -e

echo "=========================================="
echo "SystemC Installation Script"
echo "=========================================="

SYSTEMC_VERSION="2.3.3"
SYSTEMC_DIR="/opt/systemc"

START_DIR=$(pwd)

mkdir -p ${SYSTEMC_DIR}
cd ${SYSTEMC_DIR}

wget http://accellera.org/images/downloads/standards/systemc/systemc-2.3.3.tar.gz \
    -O systemc-${SYSTEMC_VERSION}.tar.gz
tar -xzf systemc-${SYSTEMC_VERSION}.tar.gz

cd systemc-${SYSTEMC_VERSION}
mkdir build
cd build

SYSTEMC_HOME=${SYSTEMC_DIR}/systemc-${SYSTEMC_VERSION}-install

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${SYSTEMC_HOME} \
    -DINSTALL_TO_LIB_TARGET_ARCH_DIR=ON \
    ..

make
make install

cd "${START_DIR}"

echo ""
echo "=========================================="
echo "SystemC ${SYSTEMC_VERSION} installed successfully!"
echo "Installed at: ${SYSTEMC_DIR}/systemc-${SYSTEMC_VERSION}-install"
echo "=========================================="
echo ""

export SYSTEMC_HOME=${SYSTEMC_HOME}
if [ -z "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH=${SYSTEMC_HOME}/lib-linux64
else
    export LD_LIBRARY_PATH=${SYSTEMC_HOME}/lib-linux64:${LD_LIBRARY_PATH}
fi
