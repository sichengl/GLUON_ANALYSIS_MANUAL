#Functions shared by the 2pt production scripts, written once here instead of once per setup file.
#They do not depend on where the 2pt script runs: everything machine-specific (sources, cfg list, paths) stays in the setup file.
#
#TwoPtParams holds the parameters of a setup file and its methods are the shared functions. Build it empty, then set the parameters one by one:
#    from pt2_comm_tools import TwoPtParams, release_cupy_cache, log_gpu_memory
#    fermilab_params = TwoPtParams()
#    fermilab_params.GLs = 32
#    fermilab_params.GLt = 96
#    fermilab_params.t_base = np.arange(0, fermilab_params.GLt, 6)
#    ...
#and call the functions on it, passing what they need at run time:
#    t_src, x_src, y_src, z_src = fermilab_params.make_sources(measurement_list)
#    fermilab_params.init_quda(quda_resource_path)
#    latt_info = fermilab_params.make_latt_info()
#    dirac = fermilab_params.make_dirac(latt_info)
#    corr_dic = fermilab_params.make_containers_one_tsrc(latt_info)
#cupy and pyquda_utils are imported inside the functions: importing pyquda calls MPI_Init, and the job scripts import the setup outside srun.
import math
import numpy as np


#======== GPU memory (QUDA allocates with cudaMalloc and cannot use the blocks cupy keeps cached) ========
def release_cupy_cache():
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    cp.fft.config.get_plan_cache().clear()

def log_gpu_memory(tag):
    import cupy as cp
    from pyquda_utils import core
    free, total = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    core.getLogger().info(f"{tag}: GPU used {(total - free)/1e9:.1f} GB, cupy pool {pool.total_bytes()/1e9:.1f} GB, cupy live {pool.used_bytes()/1e9:.1f} GB")


#======== gamma matrices: PyQUDA's convention ========
#DeGrand-Rossi basis as in Chroma: Gamma(n) = g1^n0 g2^n1 g3^n2 g4^n3 with n = 8 n3 + 4 n2 + 2 n1 + n0,
#g5 = Gamma(15) = g1 g2 g3 g4 = diag(1, 1, -1, -1), C = Gamma(10) = g2 g4, C g5 = Gamma(5) = g1 g3.
#GAMMA_NAMES[n] is exactly Gamma(n), sign included, written with G5 where that is shorter:
#Gamma(7) = g1 g2 g3 = G5 G4 = -G4 G5, Gamma(11) = g1 g2 g4 = G3 G5, Gamma(13) = g1 g3 g4 = G5 G2 = -G2 G5, Gamma(14) = g2 g3 g4 = G1 G5
GAMMA_NAMES = ["1",  "G1",   "G2",   "G1G2", "G3",   "G1G3", "G2G3", "G5G4",
               "G4", "G1G4", "G2G4", "G3G5", "G3G4", "G5G2", "G1G5", "G5"]


#======== color epsilon tensor (call after core.init, so that it is created on this rank's GPU) ========
def make_eps():
    import cupy as cp
    eps_color = cp.zeros((3, 3, 3), dtype=cp.complex128)
    eps_color[0,1,2] = eps_color[1,2,0] = eps_color[2,0,1] = +1
    eps_color[0,2,1] = eps_color[2,1,0] = eps_color[1,0,2] = -1
    return eps_color


#======== parameters of one setup file, and the functions that use them ========
class TwoPtParams:

    def __init__(self):
        #the parameters of a 2pt run: the methods below use some, the 2pt scripts read the others directly.
        #The setup file sets them one by one (those with a value here have a default)
        #lattice, cfgs and sources
        self.GLs = None                 #global lattice size, as PyQUDA names it: GLs = GLx = GLy = GLz in space, GLt in time
        self.GLt = None                 #(latt_info.Lt is the local time extent of one rank, GLt / Gt)
        self.cfg_first = None           #number of the first cfg of the ensemble
        self.cfg_step = None            #step between cfg numbers: cfg = cfg_first + cfg_step * (index of the cfg)
        self.t_base = None
        self.x_base = None
        self.y_base = None
        self.z_base = None
        self.temporal_shift = 5         #the sources move by these many lattice units from one cfg to the next: t by temporal_shift,
        self.spatial_shift = 3          #x, y and z each by spatial_shift
        #QUDA: the arguments of core.init, core.LatticeInfo and core.getDirac, with PyQUDA's names
        self.grid_size = None           #number of GPUs (MPI processes) along x, y, z, t, e.g. [1, 1, 1, 4]: local size = global size / grid_size
        self.t_boundary = -1            #fermion boundary condition in t: -1 antiperiodic (PyQUDA flips the sign of the t links that wrap around), +1 periodic
        self.anisotropy = 1.0           #renormalized anisotropy xi = a_s / a_t: 1.0 is isotropic
        self.mass = None                #bare quark mass m0: kappa = 1 / (2 (m0 + 1 + 3 / xi)), = 1 / (2 (m0 + 4)) at xi = 1
        self.tol = None                 #the solver stops when the relative residual |r| / |b| is below tol
        self.maxiter = None             #maximum number of solver iterations
        self.xi_0 = 1.0                 #bare anisotropy: enters the clover coefficient only when anisotropy != 1
        self.clover_coeff_t = None      #clover coefficients: at anisotropy 1, csw = clover_coeff_t and clover_coeff_r is not used
        self.clover_coeff_r = None
        self.multigrid = None           #multigrid block sizes, e.g. [[4, 4, 4, 4], [2, 2, 2, 2]]: 3 levels, blocking 4^4 sites, then 2^4
        self.mrhs = 12                  #right-hand sides solved together by invertPropagator: 12 = all spin-color columns at once
        #gauge field preparation: the arguments of gauge.fixingOVR and gauge.hypSmear, with PyQUDA's names
        self.gauge_fix_dir = 3                  #3 Coulomb gauge (the boosted smearing needs it), 4 Landau
        self.gauge_fix_Nsteps = None            #maximum number of overrelaxation steps
        self.gauge_fix_verbose_interval = None  #print the gauge fixing status every this many steps
        self.gauge_fix_relax_boost = None       #overrelaxation parameter, usually 1.5 or 1.7
        self.gauge_fix_tolerance = None         #stop when the gauge fixing quality is below this (0: always run Nsteps)
        self.gauge_fix_reunit_interval = None   #reunitarize the links every this many steps
        self.gauge_fix_stopWtheta = None        #quantity compared with tolerance: 1 theta, 0 the MILC criterion
        self.hyp_n_steps = None                 #number of HYP smearing steps
        self.hyp_alpha1 = None                  #the three HYP smearing parameters
        self.hyp_alpha2 = None
        self.hyp_alpha3 = None
        self.hyp_dir_ignore = -1                #direction left unsmeared: -1 smears all four
        self.hyp_compute_plaquette = True       #print the plaquette after smearing (diagnostic only)
        self.hyp_compute_qcharge = True         #print the topological charge after smearing (diagnostic only)
        #contraction and job
        self.nt_chunk = None            #local time slices per site-local contraction, to save GPU memory: must be set where the script chunks (delta)
        self.icfg = 0                   #default of --icfg: position in cfg_list of the job's first cfg
        self.n_cfg = 20                 #default of --n: number of cfgs the job measures
        #smearing and gammas
        self.shape_list = None
        self.mom_frac_list = None
        self.k_list = None
        self.mom_min = None
        self.mom_max = None
        self.gamma_ids = None
        #momentum transfers and sink momenta
        self.q_forward_list = None
        self.q_sym_0xi_list = None
        self.q_sym_non0xi_list = None
        self.q_asy_0xi_list = None
        self.q_asy_non0xi_list = None
        self.q_plain_list = None
        self.pz_list = None
        self.pf_plain_list = None
        #forward and backward time slices of the *_fb output
        self.tsep_max = None
        self.fb_index = None


    #======== sources and run parameters ========
    def make_sources(self, cfgs):
        #cfgs: cfg numbers; the sources of a cfg are shifted according to its index (cfg - cfg_first) / cfg_step
        ncfg, rest = np.divmod(np.asarray(cfgs) - self.cfg_first, self.cfg_step)
        assert np.all(rest == 0), f"every cfg must be cfg_first + cfg_step * n (cfg_first {self.cfg_first}, cfg_step {self.cfg_step})"
        #a shift coprime to the lattice size is coprime to the source spacing too, so the sources go through as many different sets as possible
        assert math.gcd(self.temporal_shift, self.GLt) == 1, f"temporal_shift {self.temporal_shift} is not coprime to GLt {self.GLt}"
        assert math.gcd(self.spatial_shift, self.GLs) == 1, f"spatial_shift {self.spatial_shift} is not coprime to GLs {self.GLs}"
        t_src = (self.t_base[None, :] + self.temporal_shift*ncfg[:, None]) % self.GLt
        x_src = (self.x_base[None, :] + self.spatial_shift*ncfg[:, None]) % self.GLs
        y_src = (self.y_base[None, :] + self.spatial_shift*ncfg[:, None]) % self.GLs
        z_src = (self.z_base[None, :] + self.spatial_shift*ncfg[:, None]) % self.GLs
        return t_src, x_src, y_src, z_src


    def make_run_parameters(self, n, t_src, x_src, y_src, z_src):
        return {
            "GLs": self.GLs,
            "GLt": self.GLt,
            "cfgs_to_meas": n,
            "shape_list": self.shape_list,
            "mom_frac_list": self.mom_frac_list,
            "k_list": self.k_list,
            "mom_min": self.mom_min,
            "mom_max": self.mom_max,
            "x_src_list_shifted": x_src,
            "y_src_list_shifted": y_src,
            "z_src_list_shifted": z_src,
            "t_src_list_shifted": t_src,
            "q_forward_list": self.q_forward_list,
            "q_sym_0xi_list": self.q_sym_0xi_list,
            "q_sym_non0xi_list": self.q_sym_non0xi_list,
            "q_asy_0xi_list": self.q_asy_0xi_list,
            "q_asy_non0xi_list": self.q_asy_non0xi_list,
            "pz_list": self.pz_list,
            "pf_plain_list": self.pf_plain_list,
        }


    #======== QUDA: initialization, lattice and Dirac operator ========
    def init_quda(self, resource_path):
        from pyquda_utils import core
        core.init(self.grid_size, resource_path=resource_path)

    def make_latt_info(self):
        from pyquda_utils import core
        return core.LatticeInfo([self.GLs, self.GLs, self.GLs, self.GLt], self.t_boundary, self.anisotropy)

    def make_dirac(self, latt_info):
        from pyquda_utils import core
        return core.getDirac(latt_info, self.mass, self.tol, self.maxiter, self.xi_0, self.clover_coeff_t, self.clover_coeff_r, self.multigrid)


    #======== gamma matrices of the interpolators (call after core.init) ========
    #Gamma(n) from PyQUDA for every n in gamma_ids, and its name from GAMMA_NAMES: gamma_ids = [15, 7, 11] gives G5, G5G4, G3G5
    def make_gamma_list(self):
        from pyquda_utils import gamma
        gamma_list = [gamma.gamma(n) for n in self.gamma_ids]
        gamma_names = [GAMMA_NAMES[n] for n in self.gamma_ids]
        return gamma_list, gamma_names


    #======== container definitions (call after core.init): the time axis is latt_info.Lt, the local time slices of this rank ========
    def make_containers(self, latt_info):
        import cupy as cp
        n_shape, n_frac, n_gamma, n_t = len(self.shape_list), len(self.mom_frac_list), len(self.gamma_ids), len(self.t_base)
        pion   = lambda n_q, n_p: cp.zeros((n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma,       n_t, n_q, n_p, latt_info.Lt), "<c16")
        proton = lambda n_q, n_p: cp.zeros((n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma, 4, 4, n_t, n_q, n_p, latt_info.Lt), "<c16")
        return {
            "pion_forward":            pion(  len(self.q_forward_list),    len(self.pz_list)),
            "proton_forward_dirac":    proton(len(self.q_forward_list),    len(self.pz_list)),
            "pion_sym_0xi":            pion(  len(self.q_sym_0xi_list),    len(self.pz_list)),
            "proton_sym_0xi_dirac":    proton(len(self.q_sym_0xi_list),    len(self.pz_list)),
            "pion_sym_non0xi":         pion(  len(self.q_sym_non0xi_list), len(self.pz_list)),
            "proton_sym_non0xi_dirac": proton(len(self.q_sym_non0xi_list), len(self.pz_list)),
            "pion_asy_0xi":            pion(  len(self.q_asy_0xi_list),    len(self.pz_list)),
            "proton_asy_0xi_dirac":    proton(len(self.q_asy_0xi_list),    len(self.pz_list)),
            "pion_asy_non0xi":         pion(  len(self.q_asy_non0xi_list), len(self.pz_list)),
            "proton_asy_non0xi_dirac": proton(len(self.q_asy_non0xi_list), len(self.pz_list)),
            "pion_plain":              pion(  len(self.q_plain_list),      len(self.pf_plain_list)),
            "proton_plain_dirac":      proton(len(self.q_plain_list),      len(self.pf_plain_list)),
        }


    #containers without the t_src index: they hold only the t_src being computed, so the GPU never holds more than one t_src of results
    def make_containers_one_tsrc(self, latt_info):
        import cupy as cp
        n_shape, n_frac, n_gamma = len(self.shape_list), len(self.mom_frac_list), len(self.gamma_ids)
        pion   = lambda n_q, n_p: cp.zeros((n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma,       n_q, n_p, latt_info.Lt), "<c16")
        proton = lambda n_q, n_p: cp.zeros((n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma, 4, 4, n_q, n_p, latt_info.Lt), "<c16")
        return {
            "pion_forward":            pion(  len(self.q_forward_list),    len(self.pz_list)),
            "proton_forward_dirac":    proton(len(self.q_forward_list),    len(self.pz_list)),
            "pion_sym_0xi":            pion(  len(self.q_sym_0xi_list),    len(self.pz_list)),
            "proton_sym_0xi_dirac":    proton(len(self.q_sym_0xi_list),    len(self.pz_list)),
            "pion_sym_non0xi":         pion(  len(self.q_sym_non0xi_list), len(self.pz_list)),
            "proton_sym_non0xi_dirac": proton(len(self.q_sym_non0xi_list), len(self.pz_list)),
            "pion_asy_0xi":            pion(  len(self.q_asy_0xi_list),    len(self.pz_list)),
            "proton_asy_0xi_dirac":    proton(len(self.q_asy_0xi_list),    len(self.pz_list)),
            "pion_asy_non0xi":         pion(  len(self.q_asy_non0xi_list), len(self.pz_list)),
            "proton_asy_non0xi_dirac": proton(len(self.q_asy_non0xi_list), len(self.pz_list)),
            "pion_plain":              pion(  len(self.q_plain_list),      len(self.pf_plain_list)),
            "proton_plain_dirac":      proton(len(self.q_plain_list),      len(self.pf_plain_list)),
        }


    #forward and backward 2pt of one t_src: numpy arrays on the host (no core.init needed), tsep_max time slices instead of GLt.
    #Two containers per particle and case: fb_dic["pt2_forward"][key] and fb_dic["pt2_backward"][key], key as in make_containers_one_tsrc
    def make_containers_one_tsrc_fb(self):
        n_shape, n_frac, n_gamma = len(self.shape_list), len(self.mom_frac_list), len(self.gamma_ids)
        pion   = lambda n_q, n_p: np.zeros((n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma,       n_q, n_p, self.tsep_max), "<c16")
        proton = lambda n_q, n_p: np.zeros((n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma, 4, 4, n_q, n_p, self.tsep_max), "<c16")
        return {
            pt2: {
                "pion_forward":            pion(  len(self.q_forward_list),    len(self.pz_list)),
                "proton_forward_dirac":    proton(len(self.q_forward_list),    len(self.pz_list)),
                "pion_sym_0xi":            pion(  len(self.q_sym_0xi_list),    len(self.pz_list)),
                "proton_sym_0xi_dirac":    proton(len(self.q_sym_0xi_list),    len(self.pz_list)),
                "pion_sym_non0xi":         pion(  len(self.q_sym_non0xi_list), len(self.pz_list)),
                "proton_sym_non0xi_dirac": proton(len(self.q_sym_non0xi_list), len(self.pz_list)),
                "pion_asy_0xi":            pion(  len(self.q_asy_0xi_list),    len(self.pz_list)),
                "proton_asy_0xi_dirac":    proton(len(self.q_asy_0xi_list),    len(self.pz_list)),
                "pion_asy_non0xi":         pion(  len(self.q_asy_non0xi_list), len(self.pz_list)),
                "proton_asy_non0xi_dirac": proton(len(self.q_asy_non0xi_list), len(self.pz_list)),
                "pion_plain":              pion(  len(self.q_plain_list),      len(self.pf_plain_list)),
                "proton_plain_dirac":      proton(len(self.q_plain_list),      len(self.pf_plain_list)),
            }
            for pt2 in self.fb_index
        }
