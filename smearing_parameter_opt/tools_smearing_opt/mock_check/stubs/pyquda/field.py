class LatticePropagator:
    """Laptop stand-in: only holds .data in the even-odd layout [2, Lt, Lz, Ly, Lx/2, spin, spin, color, color]."""

    def __init__(self, latt_info, data=None):
        self.latt_info = latt_info
        self.data = data
