#The two functions the Frontier final 2pt script (2pt_coulomb_boosted_sepq_3072srcs_frontier_final.py) adds to the Delta script.
import ctypes
import cupy as cp
from mpi4py import MPI
from pyquda_utils import core
from coulomb_smearing import boosted_gaussian_kernel, to_evenodd


def use_run_communicator(gcds):
    #One srun can hold many independent runs: consecutive ranks, gcds at a time, form one run (with srun --gpu-bind=closest, node-local ranks 2j and
    #2j+1 are the two GCDs of one MI250X). PyQUDA and QUDA communicate over MPI_COMM_WORLD, so with more than one run they get the communicator of
    #the run instead: PyQUDA reads it from the module variables of pyquda_comm, QUDA takes it through setMPICommHandleQuda (quda.h), which PyQUDA
    #does not wrap, so it is called on the loaded libquda. Call before init_quda. Returns the index of this run and the number of runs.
    world = MPI.COMM_WORLD
    run_id, n_runs = world.Get_rank() // gcds, world.Get_size() // gcds
    if n_runs > 1:
        import pyquda_comm
        comm = world.Split(run_id, world.Get_rank())
        shared = comm.Split_type(MPI.COMM_TYPE_SHARED)
        pyquda_comm._MPI_COMM, pyquda_comm._MPI_SIZE, pyquda_comm._MPI_RANK = comm, comm.Get_size(), comm.Get_rank()
        pyquda_comm._SHARED_COMM, pyquda_comm._SHARED_SIZE, pyquda_comm._SHARED_RANK = shared, shared.Get_size(), shared.Get_rank()
        pyquda_comm._SHARED_SIZE_MAX = comm.allreduce(shared.Get_size(), MPI.MAX)
        pyquda_comm._SHARED_SIZE_MIN = comm.allreduce(shared.Get_size(), MPI.MIN)
        with open("/proc/self/maps") as f:
            libquda = next(line.split()[-1] for line in f if "libquda" in line)
        ctypes.CDLL(libquda).setMPICommHandleQuda(ctypes.c_void_p(MPI._addressof(comm)))
    return run_id, n_runs


def invert_coulomb_boosted_source(dirac, latt_info, kvec, src_pos, rho_T, rho_z, mrhs, host_columns):
    #core.invertPropagator(dirac, coulomb_boosted_source(latt_info, kvec, src_pos, rho_T, rho_z), mrhs) with less GPU memory. The source is the
    #smearing profile eta(x) = K(x - x_src) on the t_src slice times the unit spin and color matrices (coulomb_smearing.source_lexico), so its column
    #(spin, color) is eta at [..., spin, color] and zero elsewhere. The columns are built from eta mrhs at a time, solved with the call
    #invertPropagator makes, and solution column spin * 3 + color is copied to host_columns: the full source and solution are never on the GPU.
    Lx, Ly, Lz, Lt = latt_info.size
    x0, y0, z0, t0 = src_pos
    eta = cp.zeros((Lt, Lz, Ly, Lx), "<c16")
    t_local = t0 - latt_info.gt * latt_info.Lt
    if 0 <= t_local < Lt:
        eta[t_local] = cp.roll(boosted_gaussian_kernel(cp, [Lx, Ly, Lz], kvec, rho_T, rho_z), shift=(z0, y0, x0), axis=(0, 1, 2))
    eta = to_evenodd(eta)
    for s in range(0, 12, mrhs):
        b = core.MultiLatticeFermion(latt_info, min(mrhs, 12 - s))
        for i in range(b.L5):
            b.data[i][..., (s + i) // 3, (s + i) % 3] = eta
        x = dirac.invertMultiSrc(b)
        for i in range(b.L5):
            x.data[i].get(out=host_columns[s + i])
