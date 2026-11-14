import numpy as np

from . import core


class MockGauge:
    """Unit links in the even-odd layout [direction][2, Lt, Lz, Ly, Lx/2, color, color]: already in Coulomb gauge."""

    def __init__(self, cfg):
        self.cfg = cfg
        Lx, Ly, Lz, Lt = core.LATT.size
        self.data = np.zeros((4, 2, Lt, Lz, Ly, Lx // 2, 3, 3), "<c16") + np.eye(3)

    def fixingOVR(self, **kwargs):
        pass

    def copy(self):
        return MockGauge(self.cfg)

    def hypSmear(self, *args):
        pass

    def plaquette(self):
        return [1.0, 1.0, 1.0]


def readMILCGauge(filename, checksum=True, reunitarize_sigma=1e-6):
    return MockGauge(int(filename.rsplit(".", 1)[1]))
