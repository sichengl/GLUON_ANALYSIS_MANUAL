No code here. 4 gpuA100x8 nodes are 32 GPUs; with the t-only grid each GPU would hold 144/32 = 4.5 time slices. PyQUDA needs a whole, even
local extent in every direction (pyquda_comm/__init__.py: "sublattice size must be even in all directions"), so the possible t-only GPU counts for
Lt = 144 are 8, 12, 24, 36, 72 and 144. 24 GPUs is 3node (6 local slices). Splitting x, y or z as well would need a distributed FFT for the sink
smearing (coulomb_smearing_boost_direction.py stops unless the grid is [1, 1, 1, Gt]), which changes the structure of the code.
