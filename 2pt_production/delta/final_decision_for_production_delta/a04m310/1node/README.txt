No code here. On one gpuA100x8 node (8 GPUs, 64^3 x 24 sites each) QUDA's own fields alone are ~29 GB per GPU (gauge and clover in three precisions,
multigrid null vectors and coarse operators: ~4.5 kB/site, estimate), one 12-column complex128 propagator is 14.5 GB per GPU, and a solve with
one column at a time adds ~10 GB of solver workspace: ~54 GB before any smearing or contraction. Even the most host-memory-heavy variant
(hostpark_phases, mrhs 1) is ~68 GB per GPU against 40 GB. The smallest t-only layouts that can fit are 2node (16 GPUs) with hostpark or 3node.
