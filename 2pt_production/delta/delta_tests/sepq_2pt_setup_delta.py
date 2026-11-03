import numpy as np


#======== parameter definitions ========
Ls = 32
Lt = 96
cfg_list = np.arange(204, 204 + 200*6, 6)
cfg_measure_spacing = 1
src_phase_sign = -1
mom_min = 0
mom_max = 6
rho_T = 3.25
shape_list = [[rho_T, rho_T], [rho_T, rho_T / 2],[rho_T, rho_T / 3]]
shape_names = ["iso", "aniso2", "aniso3"]      # rho_z = rho_T, rho_T/2, rho_T/3
mom_frac_list = [0.6,0.3]
k_list = [np.array([0, 0, frac * mom_max]) for frac in mom_frac_list]
smear_list = [[(i_shape,shape),(i_frac,frac)] for i_shape, shape in enumerate(shape_list) for i_frac, frac in enumerate(mom_frac_list)]
frac_str = [("%.3f" % frac).rstrip("0").rstrip(".").replace(".", "p") for frac in mom_frac_list]
smear_tag = f"coulomb_rhoT{rho_T}_{'-'.join(shape_names)}_frac{'-'.join(frac_str)}"
gamma_ids   = [15, 7]
gamma_names = ["G5", "G45"]


#======== forward and backward time slices of the *_tsrcsave_fb output (taken after the roll, so index 0 is t_src) ========
tsep_max = 20
forward_index  = np.arange(tsep_max)             # 0, 1, 2, ..., 19:   C(t_src + tsep)
backward_index = (-np.arange(tsep_max)) % Lt     # 0, 95, 94, ..., 77: C(t_src - tsep)
fb_index = {"pt2_forward": forward_index, "pt2_backward": backward_index}      # dataset name in the .h5 file: its time slices after the roll


#======== list definitions ========
q_forward_list    = [[ 0, 0, 0]]
q_sym_0xi_list     = [[ 2, 0, 0],[-2, 0, 0],[ 0, 2, 0],[ 0,-2, 0],[ 2, 2, 0],[ 2,-2, 0],[-2, 2, 0],[-2,-2, 0]]
q_sym_non0xi_list  = [[ 0, 0, 2],[ 2, 0, 2],[-2, 0, 2],[ 0, 2, 2],[ 0,-2, 2],[ 2, 2, 2],[ 2,-2, 2],[-2, 2, 2],[-2,-2, 2],
                     [ 0, 0,-2],[ 2, 0,-2],[-2, 0,-2],[ 0, 2,-2],[ 0,-2,-2],[ 2, 2,-2],[ 2,-2,-2],[-2, 2,-2],[-2,-2,-2]]
q_asy_0xi_list     = [[ 1, 0, 0],[-1, 0, 0],[ 0, 1, 0],[ 0,-1, 0],[ 1, 1, 0],[ 1,-1, 0],[-1, 1, 0],[-1,-1, 0]]
q_asy_non0xi_list  = [[ 0, 0, 1],[ 1, 0, 1],[-1, 0, 1],[ 0, 1, 1],[ 0,-1, 1],[ 1, 1, 1],[ 1,-1, 1],[-1, 1, 1],[-1,-1, 1],
                     [ 0, 0,-1],[ 1, 0,-1],[-1, 0,-1],[ 0, 1,-1],[ 0,-1,-1],[ 1, 1,-1],[ 1,-1,-1],[-1, 1,-1],[-1,-1,-1],
                     [ 0, 0, 2],[ 1, 0, 2],[-1, 0, 2],[ 0, 1, 2],[ 0,-1, 2],[ 1, 1, 2],[ 1,-1, 2],[-1, 1, 2],[-1,-1, 2],
                     [ 0, 0,-2],[ 1, 0,-2],[-1, 0,-2],[ 0, 1,-2],[ 0,-1,-2],[ 1, 1,-2],[ 1,-1,-2],[-1, 1,-2],[-1,-1,-2]]
q_plain_list       = [[ 0, 0, 0]]

pz_list = list(range(mom_min, mom_max + 1)) 
P_list = np.array([[0, 0, pz] for pz in pz_list])
pf_forward_table    = P_list + np.array(q_forward_list)[:, None]
pf_sym_0xi_table    = P_list + np.array(q_sym_0xi_list)[:, None] // 2
pf_sym_non0xi_table = P_list + np.array(q_sym_non0xi_list)[:, None] // 2
pf_asy_0xi_table    = P_list + np.array(q_asy_0xi_list)[:, None]
pf_asy_non0xi_table = P_list + np.array(q_asy_non0xi_list)[:, None]

pf_plain_list = [[px, py, pz] for px in [0,-1,1] for py in [0,-1,1] for pz in range(mom_min - 2, mom_max + 3)]

pf_forward_index_list    = [[pf_plain_list.index(pf) for pf in row] for row in pf_forward_table.tolist()]
pf_sym_0xi_index_list    = [[pf_plain_list.index(pf) for pf in row] for row in pf_sym_0xi_table.tolist()]
pf_sym_non0xi_index_list = [[pf_plain_list.index(pf) for pf in row] for row in pf_sym_non0xi_table.tolist()]
pf_asy_0xi_index_list    = [[pf_plain_list.index(pf) for pf in row] for row in pf_asy_0xi_table.tolist()]
pf_asy_non0xi_index_list = [[pf_plain_list.index(pf) for pf in row] for row in pf_asy_non0xi_table.tolist()]


#======== source definitions ========
t_base = np.arange(0, Lt, 32)
x_base = np.arange(0, Ls,  16)
y_base = np.arange(0, Ls,  16)
z_base = np.arange(0, Ls,  16)
n_spatial_src = len(x_base) * len(y_base) * len(z_base)


def make_sources(ncfg):
    t_src = (t_base[None, :] + 5*ncfg[:, None]) % Lt
    x_src = (x_base[None, :] + 3*ncfg[:, None]) % Ls
    y_src = (y_base[None, :] + 3*ncfg[:, None]) % Ls
    z_src = (z_base[None, :] + 3*ncfg[:, None]) % Ls
    return t_src, x_src, y_src, z_src


def make_run_parameters(n, t_src, x_src, y_src, z_src):
    return {
        "Ls": Ls,
        "Lt": Lt,
        "cfgs_to_meas": n,
        "shape_list": shape_list,
        "mom_frac_list": mom_frac_list,
        "k_list": k_list,
        "mom_min": mom_min,
        "mom_max": mom_max,
        "x_src_list_shifted": x_src,
        "y_src_list_shifted": y_src,
        "z_src_list_shifted": z_src,
        "t_src_list_shifted": t_src,
        "q_forward_list": q_forward_list,
        "q_sym_0xi_list": q_sym_0xi_list,
        "q_sym_non0xi_list": q_sym_non0xi_list,
        "q_asy_0xi_list": q_asy_0xi_list,
        "q_asy_non0xi_list": q_asy_non0xi_list,
        "pz_list": pz_list,
        "pf_plain_list": pf_plain_list,
    }


#======== gamma and epsilon tensor definitions (call after core.init) ========
def make_gamma_eps():
    import cupy as cp
    from pyquda_utils import gamma
    gamma_list = [gamma.gamma(i) for i in gamma_ids]
    G4 = gamma.gamma(8)
    G5 = gamma.gamma(15)
    charge = gamma.gamma(10)
    parity_p = (gamma.gamma(0) + gamma.gamma(8)) / 2
    parity_m = (gamma.gamma(0) - gamma.gamma(8)) / 2
    eps_color = cp.zeros((3, 3, 3), dtype=cp.complex128)
    eps_color[0,1,2] = eps_color[1,2,0] = eps_color[2,0,1] = +1
    eps_color[0,2,1] = eps_color[2,1,0] = eps_color[1,0,2] = -1
    return gamma_list, G4, G5, charge, parity_p, parity_m, eps_color


#======== container definitions (call after core.init) ========
def make_containers(latt_info):
    import cupy as cp
    n_shape, n_frac, n_gamma, n_t = len(shape_list), len(mom_frac_list), len(gamma_ids), len(t_base)
    pion   = lambda n_q, n_p: cp.zeros((n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma,       n_t, n_q, n_p, latt_info.Lt), "<c16")
    proton = lambda n_q, n_p: cp.zeros((n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma, 4, 4, n_t, n_q, n_p, latt_info.Lt), "<c16")
    return {
        "pion_forward":            pion(  len(q_forward_list),    len(pz_list)),
        "proton_forward_dirac":    proton(len(q_forward_list),    len(pz_list)),
        "pion_sym_0xi":            pion(  len(q_sym_0xi_list),    len(pz_list)),
        "proton_sym_0xi_dirac":    proton(len(q_sym_0xi_list),    len(pz_list)),
        "pion_sym_non0xi":         pion(  len(q_sym_non0xi_list), len(pz_list)),
        "proton_sym_non0xi_dirac": proton(len(q_sym_non0xi_list), len(pz_list)),
        "pion_asy_0xi":            pion(  len(q_asy_0xi_list),    len(pz_list)),
        "proton_asy_0xi_dirac":    proton(len(q_asy_0xi_list),    len(pz_list)),
        "pion_asy_non0xi":         pion(  len(q_asy_non0xi_list), len(pz_list)),
        "proton_asy_non0xi_dirac": proton(len(q_asy_non0xi_list), len(pz_list)),
        "pion_plain":              pion(  len(q_plain_list),      len(pf_plain_list)),
        "proton_plain_dirac":      proton(len(q_plain_list),      len(pf_plain_list)),
    }


#containers without the t_src index: they hold only the t_src being computed, so the GPU never holds more than one t_src of results
def make_containers_one_tsrc(latt_info):
    import cupy as cp
    n_shape, n_frac, n_gamma = len(shape_list), len(mom_frac_list), len(gamma_ids)
    pion   = lambda n_q, n_p: cp.zeros((n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma,       n_q, n_p, latt_info.Lt), "<c16")
    proton = lambda n_q, n_p: cp.zeros((n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma, 4, 4, n_q, n_p, latt_info.Lt), "<c16")
    return {
        "pion_forward":            pion(  len(q_forward_list),    len(pz_list)),
        "proton_forward_dirac":    proton(len(q_forward_list),    len(pz_list)),
        "pion_sym_0xi":            pion(  len(q_sym_0xi_list),    len(pz_list)),
        "proton_sym_0xi_dirac":    proton(len(q_sym_0xi_list),    len(pz_list)),
        "pion_sym_non0xi":         pion(  len(q_sym_non0xi_list), len(pz_list)),
        "proton_sym_non0xi_dirac": proton(len(q_sym_non0xi_list), len(pz_list)),
        "pion_asy_0xi":            pion(  len(q_asy_0xi_list),    len(pz_list)),
        "proton_asy_0xi_dirac":    proton(len(q_asy_0xi_list),    len(pz_list)),
        "pion_asy_non0xi":         pion(  len(q_asy_non0xi_list), len(pz_list)),
        "proton_asy_non0xi_dirac": proton(len(q_asy_non0xi_list), len(pz_list)),
        "pion_plain":              pion(  len(q_plain_list),      len(pf_plain_list)),
        "proton_plain_dirac":      proton(len(q_plain_list),      len(pf_plain_list)),
    }


#forward and backward 2pt of one t_src: numpy arrays on the host (no core.init needed), tsep_max time slices instead of Lt.
#Two containers per particle and case: fb_dic["pt2_forward"][key] and fb_dic["pt2_backward"][key], key as in make_containers_one_tsrc
def make_containers_one_tsrc_fb():
    n_shape, n_frac, n_gamma = len(shape_list), len(mom_frac_list), len(gamma_ids)
    pion   = lambda n_q, n_p: np.zeros((n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma,       n_q, n_p, tsep_max), "<c16")
    proton = lambda n_q, n_p: np.zeros((n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma, 4, 4, n_q, n_p, tsep_max), "<c16")
    return {
        pt2: {
            "pion_forward":            pion(  len(q_forward_list),    len(pz_list)),
            "proton_forward_dirac":    proton(len(q_forward_list),    len(pz_list)),
            "pion_sym_0xi":            pion(  len(q_sym_0xi_list),    len(pz_list)),
            "proton_sym_0xi_dirac":    proton(len(q_sym_0xi_list),    len(pz_list)),
            "pion_sym_non0xi":         pion(  len(q_sym_non0xi_list), len(pz_list)),
            "proton_sym_non0xi_dirac": proton(len(q_sym_non0xi_list), len(pz_list)),
            "pion_asy_0xi":            pion(  len(q_asy_0xi_list),    len(pz_list)),
            "proton_asy_0xi_dirac":    proton(len(q_asy_0xi_list),    len(pz_list)),
            "pion_asy_non0xi":         pion(  len(q_asy_non0xi_list), len(pz_list)),
            "proton_asy_non0xi_dirac": proton(len(q_asy_non0xi_list), len(pz_list)),
            "pion_plain":              pion(  len(q_plain_list),      len(pf_plain_list)),
            "proton_plain_dirac":      proton(len(q_plain_list),      len(pf_plain_list)),
        }
        for pt2 in fb_index
    }


#======== path definitions ========
current_dir = "/projects/biqz/sliu54/2pt_production"
quda_resource_path = "/u/sliu54/QUDA_resources/.cache/quda"
gauge_file_prefix = "/work/hdd/biqz//MILC/l3296f211b630m0074m037m440d/l3296f211b630m0074m037m440d"
transfer_note = ("P_f = P_i + q (physical momenta, equal to the labels): source weight exp(-i 2pi/Ls q.x_src), gluon operator "
                    "Fourier transformed with exp(+i 2pi/Ls q.z) (FF q_phase_sign = +1). The initial-state 2pt of a 3pt at sink "
                    "label pf and transfer q is at label pf - q.")