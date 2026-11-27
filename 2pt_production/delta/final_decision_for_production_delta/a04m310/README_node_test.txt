a04m310 (l64192f211b700m00316m0158m188, 64^3 x 192) node tests on Delta gpuA100x8 (8 A100-40GB per node, NVLink)

folder / method                 GPUs  local t  code             mrhs  nt_chunk  expected GB/GPU
1node                              -       -   -                   -      -        not possible: see 1node/README.txt
2node/method_1_hostpark           16      12   hostpark            1      1        ~40  (at the limit)
2node/method_2_hostpark_phases    16      12   hostpark_phases     1      1        ~35
3node/method_1_lean               24       8   lean                1      1        ~38  (at the limit)
3node/method_2_hostpark           24       8   hostpark            1      1        ~29
4node/method_1_mrhs1              32       6   asis                1      1        ~33
4node/method_2_lean               32       6   lean                1      1        ~30
mass -0.022 and c_SW 1.029 in the a04 setups are TEST VALUES (memory and time do not depend on them): set them after the a04 tuning.

How to test one method (Delta, from its folder):
    mkdir -p logs && BOOST_DIRECTION=x sbatch <the .sh in that folder>          (cfg_list[0..4] = cfgs 1002 .. 1026; others: ICFG=5 BOOST_DIRECTION=x sbatch ...)
    The job measures five cfgs with 24 sources each (3 t_src x 2 x 2 x 2, test statistics; production 3072) in at most 2 h (#SBATCH --time);
    every t_src (8 sources) that finishes is saved and a later job continues from there.
What to look at after the test:
    grep -m 3 "AFTER INVERT" logs/<name>_<jobid>.out                 GPU memory after the solves (rank 0): total, cupy pool, cupy live
    grep -e "UNTIL CONTRACTION" logs/<name>_<jobid>.out | tail -n 3      seconds per source (the first one includes the startup)
    grep -e "Device memory used" -e "out of memory" -e "Traceback" logs/<name>_<jobid>.*   QUDA's peak, or the failure
    /projects/biqz/sliu54/2pt_production/testruns/<name>_<jobid>/gpu_mem_<jobid>.csv   the 8 GPUs of the first node every 2 s
    The output of every method goes to its own folder (current_dir of its setup), so methods can be compared with each other on the same cfg.

The code variants (each file in a method folder is a generated copy; every changed line is marked NODE TEST):
    asis             the production 2pt script (a09m310/delta_2pt_production_a09m310_stream_e_direction_x, md5 93906d35) unchanged
    lean             asis + gauge copy freed after loadGauge, +k source made after the -k solve, sink phases made after the solves,
                     sink-smearing FFT nt_chunk time slices at a time (coulomb_smearing_boost_direction_tchunk.py)
    hostpark         lean + every solved propagator waits in host memory until its sink smearing, which works in place on a GPU copy
    hostpark_phases  hostpark + the 99 sink phases stay in host memory; only the current time chunk is copied to the GPU
All variants give the same correlators as the production script (CPU mock with PyQUDA stand-ins, 2026-11-26).
The GPU grid is split along t only: the sink smearing is a 3D FFT and needs unsplit x, y, z, and PyQUDA needs an even local extent in every direction.

Expected peak GPU memory per GPU is from a per-site model (QUDA fixed part ~4.5 kB/site estimated, not measured). The same model gives 26.6 GB for the
a09 Delta production against 36.2 GB measured, so up to ~10 GB more is possible; the tests are what decides.
Physics choices to confirm before the production (same in every method): smearing widths = a09's in fm, tsep_max = a09's 20 in fm,
the source grid (test: 24 per cfg; production: 3072 on a09's grid in fm), and for a04 the mass and c_SW (test values).
