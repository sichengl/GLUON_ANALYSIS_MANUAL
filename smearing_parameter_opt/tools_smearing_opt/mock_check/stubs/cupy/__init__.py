"""Laptop stand-in for cupy: numpy under the name cupy, with the few extra names the scripts use. For mock checks only."""
import numpy as _np
from numpy import *  # noqa: F401,F403

from . import cuda  # noqa: F401


def asnumpy(a):
    return _np.asarray(a)


def get_array_module(*args):
    return _np


class _Pool:
    def free_all_blocks(self):
        pass

    def total_bytes(self):
        return 0

    def used_bytes(self):
        return 0


def get_default_memory_pool():
    return _Pool()


class _PlanCache:
    def clear(self):
        pass


class _Config:
    def get_plan_cache(self):
        return _PlanCache()


class _FFT:
    fftn = staticmethod(_np.fft.fftn)
    ifftn = staticmethod(_np.fft.ifftn)
    config = _Config()


fft = _FFT()
