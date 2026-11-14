#!/bin/bash
# Mock run of the smearing optimization on a laptop: the REAL worker script of one run folder, on several "nodes" at once (one mpiexec each), with
# numpy stand-ins for cupy and PyQUDA (stubs/) and the free Wilson inverse as the solver. Checks results, not speed.
#   bash run_mock.sh <empty scratch directory> [ranks per node: 4] [run folder: delta_test] [nodes: 3] [node that is NOT started: none]
# The last argument tests the lost-node rule of the many-node (Frontier) folders: that node never delivers, and the run must go on without it.
# Then: python3 check_mock_output.py <scratch directory>/run
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OPT=$HERE/../..                                  # smearing_parameter_opt
TOOLS2PT=$OPT/../2pt_production/tools
WORK=$1
NRANK=${2:-4}
FOLDER=${3:-delta_test}
NWORKERS=${4:-3}
NOT_STARTED=${5:-none}
SCRIPT=$(basename "$OPT/$FOLDER"/opt_worker_*.py)
SETUP=$(basename "$OPT/$FOLDER"/opt_setup_*.py)
mkdir -p "$WORK/code" "$WORK/logs"
# copies of exactly what runs, as the job script makes them
cp "$OPT/$FOLDER/$SCRIPT" "$TOOLS2PT/coulomb_smearing.py" "$TOOLS2PT/pt2_comm_tools.py" \
   "$OPT/tools_smearing_opt/opt_kernel.py" "$OPT/tools_smearing_opt/opt_contract.py" "$OPT/tools_smearing_opt/opt_check.py" "$WORK/code/"
python3 "$HERE/make_mock_setup.py" "$OPT/$FOLDER/$SETUP" "$WORK/code/$SETUP" "$WORK/run" "$NRANK" "$NWORKERS"
cd "$WORK/code"
export PYTHONPATH=$HERE/stubs:$WORK/code
for ((w = 0; w < NWORKERS; w++)); do
    [ "$w" = "$NOT_STARTED" ] && continue
    mpiexec -n "$NRANK" python3 -m mpi4py "$SCRIPT" --worker $w > "$WORK/logs/worker_$w.out" 2> "$WORK/logs/worker_$w.err" &
done
wait
echo "mock run of $FOLDER finished: status in $WORK/run/status.log, logs in $WORK/logs"
