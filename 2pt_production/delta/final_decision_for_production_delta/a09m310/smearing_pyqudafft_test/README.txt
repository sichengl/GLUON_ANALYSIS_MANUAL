Test of 2pt_production/tools/coulomb_smearing_pyqudafft.py (Coulomb-gauge boosted smearing with PyQUDA's distributed FFT, any process grid)
against the Delta x results of cfg 2682 (stream e) from the 10-cfg x test of 2026-11-17 (job 24085194, production smearing, production setup,
production QUDA tuning cache): /work/hdd/biqz/sliu54/2pt_production/test_directionxyz/coulomb_..._smear_direction_x_GEVP_sepq_3072src_phyp_fb_tsep20/cfg2682

No code is copied into this folder. test_2pt_pyqudafft.sh runs the production files of ../delta_2pt_production_a09m310_stream_e_direction_x and
changes only their copies in the run folder: the main script imports coulomb_smearing_pyqudafft, the setup writes to
/work/hdd/biqz/sliu54/2pt_production/test_smearing_pyqudafft/grid<GRID>, and with GRID another grid_size and QUDA tuning cache.
    sbatch test_2pt_pyqudafft.sh                  grid [1, 1, 1, 4] (production), expected IDENTICAL
    GRID=2,2,1,1 sbatch test_2pt_pyqudafft.sh     x and y split, expected SAME (1e-9 to 1e-7: the solver and gauge fixing run on another grid)
8 h limit, TIMEOUT is the expected end; each saved t_src (128 sources) is compared. Then, on a login node:
    python compare_2pt_pyqudafft_with_x_test.py --grid 1114
    python compare_2pt_pyqudafft_with_x_test.py --grid 2211

Checked on the Mac (2026-11-27): the real pyquda_comm field classes and pyquda_utils.fft of PyQUDA 0.10.54 under MPI (numpy backend), grids
[1,1,1,1], [1,1,1,4], [2,1,1,1], [1,2,1,1], [1,1,2,1], [2,2,1,1], [1,2,2,1], [2,1,1,2], [2,2,2,1], [2,2,2,2] on an 8 x 12 x 4 x 8 lattice:
sink smearing, source and theta equal to the production math on the whole lattice (max difference 0), and bit for bit equal to the production
module on the t-only grids; the production local FFT and a local roll in theta (negative controls) are wrong on every space-split grid.
Not checked there: cupy on a GPU, the PyQUDA version on Delta (the job prints it and stops early if fft/ifft/shift/location are missing), the time.
