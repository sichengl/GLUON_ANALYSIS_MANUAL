"""Laptop stand-in for pyquda_utils.core: real mpi4py, numpy arrays, the free Wilson inverse of mock_free.py as the solver."""
from math import prod

import numpy as np
from mpi4py import MPI

from mock_free import free_wilson_inverse, mock_mass
from pyquda.field import LatticePropagator

GRID = None
LATT = None


def init(grid_size, resource_path=None):
    global GRID
    GRID = list(grid_size)


class _Logger:
    def info(self, text):
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(text, flush=True)


def getLogger():
    return _Logger()


class LatticeInfo:
    def __init__(self, latt_size, t_boundary=1, anisotropy=1.0):
        global LATT
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.grid_size = GRID
        assert GRID[:3] == [1, 1, 1] and self.mpi_comm.Get_size() == GRID[3], "run with mpiexec -n <grid_size[3]>"
        self.gx = self.gy = self.gz = 0
        self.gt = self.mpi_rank  # rank i holds the i-th block of time slices
        self.global_size = list(latt_size)
        self.GLx, self.GLy, self.GLz, self.GLt = latt_size
        self.size = [latt_size[0], latt_size[1], latt_size[2], latt_size[3] // GRID[3]]
        self.Lx, self.Ly, self.Lz, self.Lt = self.size
        self.volume = prod(self.size)
        self.global_volume = prod(latt_size)
        self.t_boundary = t_boundary
        LATT = self

    # the loops of lexico() and evenodd() in pyquda/field.py of the PyQUDA source
    def lexico(self, data, multi, backend="numpy"):
        Lx, Ly, Lz, Lt = self.size
        out = np.zeros((Lt, Lz, Ly, Lx) + data.shape[5:], data.dtype)
        for t in range(Lt):
            for z in range(Lz):
                for y in range(Ly):
                    eo = (t + z + y) % 2
                    out[t, z, y, eo::2] = data[0, t, z, y]
                    out[t, z, y, 1 - eo :: 2] = data[1, t, z, y]
        return out

    def evenodd(self, data, multi, backend="numpy"):
        Lx, Ly, Lz, Lt = self.size
        out = np.zeros((2, Lt, Lz, Ly, Lx // 2) + data.shape[4:], data.dtype)
        for t in range(Lt):
            for z in range(Lz):
                for y in range(Ly):
                    eo = (t + z + y) % 2
                    out[0, t, z, y] = data[t, z, y, eo::2]
                    out[1, t, z, y] = data[t, z, y, 1 - eo :: 2]
        return out


class MockDirac:
    def __init__(self, latt_info):
        self.latt_info = latt_info
        self.cfg = None

    def loadGauge(self, gauge):
        self.cfg = gauge.cfg

    def freeGauge(self):
        pass


def getDirac(latt_info, mass, tol, maxiter, xi_0=1.0, clover_coeff_t=0.0, clover_coeff_r=1.0, multigrid=None):
    return MockDirac(latt_info)


def invertPropagator(dirac, source_propag, mrhs=1, restart=0):
    """All ranks send their time slices to rank 0, which inverts on the whole lattice and sends the slices back."""
    latt_info = dirac.latt_info
    comm = latt_info.mpi_comm
    parts = comm.gather(latt_info.lexico(source_propag.data, False), root=0)
    pieces = None
    if latt_info.mpi_rank == 0:
        solution = free_wilson_inverse(np.concatenate(parts, axis=0), mock_mass(dirac.cfg))
        pieces = np.split(solution, comm.Get_size(), axis=0)
    piece = comm.scatter(pieces, root=0)
    return LatticePropagator(latt_info, latt_info.evenodd(piece, False))
