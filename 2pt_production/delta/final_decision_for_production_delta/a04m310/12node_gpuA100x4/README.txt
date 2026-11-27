a04m310 (l64192f211b700m00316m0158m188, 64^3 x 192) CROSS-CHECK on 12 Delta gpuA100x4 nodes (4 A100-40GB per node): 48 GPUs along t, 4 time slices each

Purpose: show that the memory-saving code variants give the same numbers as the production code. All four methods run the same cfgs and sources
on the same grid with the same setup (mrhs 1, nt_chunk 1, multigrid [[4, 4, 4, 2], [2, 2, 2, 1]]) and ONE shared QUDA tuning cache
(quda_resource_path = /u/sliu54/QUDA_resources/.cache/quda_a04m310_48gpu_12node_gpuA100x4), so only the code differs.

folder                      code             copied from (byte for byte)
method_1_mrhs1              asis (production) ../4node/method_1_mrhs1
method_2_lean               lean             ../4node/method_2_lean
method_3_hostpark           hostpark         ../3node/method_2_hostpark
method_4_hostpark_phases    hostpark_phases  ../2node/method_2_hostpark_phases
The setups are ../4node/method_1_mrhs1's with grid [1, 1, 1, 48], the multigrid blocks, the names and the shared tuning cache changed.
The job scripts are the 4node/3node/2node ones with partition gpuA100x4, 12 nodes, 4 ranks and 4 GPUs per node, --mem=208G and the
production srun CPU masks (rank i on the 16 cores next to GPU i); the expected peak with the production code is ~28-41 GB per GPU (estimate).

Submit (from this folder): method_1 first, the other three after it ends, so that they reuse its tuning cache instead of tuning again:
    cd method_1_mrhs1 && mkdir -p logs && M1=$(BOOST_DIRECTION=x sbatch --parsable 2pt_delta_a04m310_12node_gpuA100x4_method_1_mrhs1.sh) && cd .. && echo $M1
    for m in method_2_lean method_3_hostpark method_4_hostpark_phases; do (cd $m && mkdir -p logs && BOOST_DIRECTION=x sbatch --parsable --dependency=afterany:$M1 2pt_delta_a04m310_12node_gpuA100x4_$m.sh); done
If method_1 runs out of GPU memory (grep "out of memory" in its logs), scancel the other three: the production code then needs 96 GPUs (24 nodes).

Compare (after the jobs, read only): python compare_methods_12node_gpuA100x4.py
    For every file of method_1 (finished cfgs: .h5; saved t_src: parts/*.npy) it prints, per method, how many arrays are bit for bit identical
    and the largest max|a - b| / max|a|. Expected: identical, or ~1e-15 for lean and later (the sink-smearing FFT runs per time slice).
