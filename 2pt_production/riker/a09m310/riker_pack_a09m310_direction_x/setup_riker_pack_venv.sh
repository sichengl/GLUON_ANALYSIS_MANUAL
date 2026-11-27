#!/bin/bash
#One-time setup on a Riker LOGIN node: a Python environment with numpy and h5py, all the packers need (the Frontier environments do not run on Riker).
#It makes /ccs/home/$USER/riker_pack_venv (home is shared with Frontier, but this environment is only for Riker) and tests the imports.
#Run on Riker from this folder:    bash setup_riker_pack_venv.sh
set -e
VENV=/ccs/home/$USER/riker_pack_venv
if [ ! -x $VENV/bin/python ]; then
    echo "making $VENV with $(command -v python3) ($(python3 --version))"
    python3 -m venv $VENV
    $VENV/bin/python -m pip install --upgrade pip
    $VENV/bin/python -m pip install numpy h5py
fi
$VENV/bin/python -c "import sys, numpy, h5py; print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| h5py', h5py.__version__)"
