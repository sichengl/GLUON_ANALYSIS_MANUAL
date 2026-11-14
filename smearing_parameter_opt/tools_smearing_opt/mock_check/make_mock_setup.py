"""
Builds the setup of a mock run from a REAL setup file by replacing single lines (lattice 8^3 x 16, a few nodes, 16 sources,
short times, scratch paths). The worker script, the tools and all other setup lines run unchanged. Works for every setup of
smearing_parameter_opt (Delta and Frontier folders).
Usage: python3 make_mock_setup.py <real setup> <mock setup to write> <opt_dir of the mock run> <ranks per node> <number of nodes>
"""
import os
import re
import sys

real, mock, opt_dir, n_rank, n_workers = sys.argv[1:6]
text = open(real).read()
one_pz = re.search(r"^opt_pz_target = \[6\]", text, re.M) is not None

rules = [  # (start of the line to replace, new line, must be there)
    (r"delta_params\.GLs = ", "delta_params.GLs = 8", True),
    (r"delta_params\.GLt = ", "delta_params.GLt = 16", True),
    (r"delta_params\.t_base = ", "delta_params.t_base = np.arange(0, delta_params.GLt, 8)", True),
    (r"delta_params\.x_base = ", "delta_params.x_base = np.arange(0, delta_params.GLs, 4)", True),
    (r"delta_params\.y_base = ", "delta_params.y_base = np.arange(0, delta_params.GLs, 4)", True),
    (r"delta_params\.z_base = ", "delta_params.z_base = np.arange(0, delta_params.GLs, 4)", True),
    (r"delta_params\.grid_size = ", f"delta_params.grid_size = [1, 1, 1, {n_rank}]", True),
    (r"delta_params\.nt_chunk = ", "delta_params.nt_chunk = 3", True),
    (r"delta_params\.mom_max = ", "delta_params.mom_max = 2", True),
    (r"opt_cfg_index = ", f"opt_cfg_index = list(range({n_workers}))", True),
    # narrower Gaussians for the 8^3 lattice; with several Gaussians three more are switched on so that their shape derivatives are tested too
    (r"#======== what is measured ========", "opt_start_table[:, 1:3] *= 0.3\nif len(opt_start_table) > 3: opt_start_table[1:4, 0] = [0.3, -0.2, 0.1]\n#======== what is measured ========", True),
    (r"opt_pz_list = ", "opt_pz_list = [0, 1, 2]", True),
    (r"opt_pz_target = ", "opt_pz_target = [2]" if one_pz else "opt_pz_target = [1, 2]", True),
    (r"opt_pz_weight = ", "opt_pz_weight = [1.0]" if one_pz else "opt_pz_weight = [0.5, 0.5]", True),
    (r"opt_t_score = ", "opt_t_score = 2", True),
    (r"opt_t_noise = ", "opt_t_noise = 4", True),
    (r"opt_t_max = ", "opt_t_max = 5", True),
    (r"opt_max_steps = ", f"opt_max_steps = {os.environ.get('MOCK_MAX_STEPS', 6)}", True),
    (r"opt_wait_minutes = ", "opt_wait_minutes = 3", True),
    (r"opt_start_wait_minutes = ", "opt_start_wait_minutes = 3", True),
    (r"opt_poll_seconds = ", "opt_poll_seconds = 0.2", True),
    (r"opt_max_lost = ", "opt_max_lost = 1", False),                      # only the many-node setups have the lost-node rule
    (r"opt_straggler_fraction = ", "opt_straggler_fraction = 0.5", False),
    (r"opt_straggler_minutes = ", "opt_straggler_minutes = 0.1", False),
    (r"opt_dir = ", f'opt_dir = "{opt_dir}"', True),
    (r"quda_resource_path = ", 'quda_resource_path = "mock"', True),
    (r"gauge_file_prefix = ", 'gauge_file_prefix = "mock/gauge"', True),
]
for start, new, needed in rules:
    text, n = re.subn(rf"^{start}.*$", lambda match: new, text, flags=re.M)
    assert n == 1 or (n == 0 and not needed), f"line starting with {start!r} found {n} times in {real}"
open(mock, "w").write(text)
