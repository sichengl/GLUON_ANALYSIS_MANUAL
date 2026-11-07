"""Convert lattice momentum integers to GeV: |p| = (2 pi / L) * |n| * (hbarc / a)

  python mom_to_gev.py            table for pz = 0..6
  python mom_to_gev.py 3          n = (0,0,3)
  python mom_to_gev.py 1 0 3      n = (1,0,3)
"""
import math
import sys

a_fm = 0.09     # lattice spacing in fm
L = 32          # spatial lattice size
hbarc = 0.1973  # GeV fm


def mom_to_gev(nx, ny, nz):
    return 2 * math.pi / L * math.sqrt(nx**2 + ny**2 + nz**2) * hbarc / a_fm


args = [int(x) for x in sys.argv[1:]]
if len(args) == 0:
    moms = [(0, 0, pz) for pz in range(7)]
elif len(args) == 1:
    moms = [(0, 0, args[0])]
elif len(args) == 3:
    moms = [tuple(args)]
else:
    sys.exit(__doc__)

for n in moms:
    print(n, f"{mom_to_gev(*n):.4f} GeV")
