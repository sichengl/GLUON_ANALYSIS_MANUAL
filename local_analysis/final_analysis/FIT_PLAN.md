# Fit plan: ratio fits and sGEVP M_eff fit, 1365 cfg

**Status: approved 2026-11-17 (draft 4 as written).** The work follows this plan exactly. Any later change is written here first, with its date.

Changes: draft 2 added the definitions, the window AIC and the operator AIC. Draft 3 adds a second AIC variant (last tsep only) and says that every variant is plotted, saved and reported, even one that is dropped. After approval, 2026-11-17: the pion sGEVP window grid at pz 6 (section 7); later the same day, the matrix element is forced negative in all three methods, after checking that the data are negative everywhere (section 4, "Sign of M"), and all sGEVP fits are redone (reports v2; v1 kept as `*_v1.pdf`); then the forced-sign sGEVP fit of the mean gets a sixth start, the sign-free fit with A clipped below zero (section 4), and all sGEVP fits are redone again (reports v3; v2 kept as `*_v2.pdf`). Draft 4 adds section 1b (how the raw 2pt and FF data become the 3pt, the ratio and M_eff), AIC over the GEVP times (part D2), and E0 per jackknife sample in S. 2026-11-18, after part D: the final result is the average over the GEVP times (D2) for all three methods, not the average over bases (D1); in the pion sGEVP at pz 6 that average runs over t0 2-5 (section 9, "Decision").

Three fit methods, pion and proton, in separate reports:

1. ratio fit, **separate** excited-state fraction
2. ratio fit, **shared** excited-state fraction
3. **sGEVP M_eff** fit

Each method fits all z together with the full correlation matrix and one shared gap.

## 0. Definitions

- **Same gap:** one dE per pz, shared by all z (w = 0-9). Each method fits it freely.
- **Jackknife error:**
    - The per-cfg data get no special treatment before the jackknife samples are made: no binning, no cfg or sample is left out or reweighted.
    - Every jackknife sample goes through the full procedure: its own fits, and its own AIC weights when AIC is used.
    - The central value is the mean over the jackknife samples. The error is the jackknife formula, err = sqrt[(N-1)/N * sum_j (x_j - mean)^2].
    - Nothing else is added to the error. Where AIC is used, the spread over windows (or bases) is printed for information but not added.
- **AIC weight** (Jay and Neil, arXiv:2008.01069), within one jackknife sample: w ~ exp[-(chi2 + 2k + 2 N_cut) / 2].
    - chi2 is that sample's fit chi2, priors included (lsqfit's chi2).
    - k is the number of fit parameters.
    - N_cut is the number of points (all ten w) of the union of all windows that this window leaves out.
    - The weights are normalized over the windows within each sample.
- **svdcut:** lsqfit's svdcut, applied to the correlation matrix. Eigenvalues below svdcut x (largest) are raised to that value.

## 1. Data (the same for all three methods)

- 1365 cfg: Delta stream e 585 + Frontier stream d 780 (the 781-cfg files without cfg 3006, which has no FF).
- Forward frame, q = 0, pz 0-6, w 0-9, HYP 8.
- Operator O = TXTX + TYTY - 2 XYXY, method-1 vacuum subtraction, both halves (forward and backward).

## 1b. From the raw data to the fitted data

Everything below is what the notebooks do now: `matrix_element_analysis_fit_per_pz.ipynb` (ratio) and `matrix_element_analysis_sgevp.ipynb` (M_eff). For each subset of part C it is redone from the cfgs of that subset.

### The 2pt files

- **Files**, one per data set and hadron: `2pt_coulomb_rho4p5z2p25-rho1p5_frac0p8-0_smear_GEVP_forward_q0_0_0_C3input_fb_tsep20_3072src_ncfg{585, 781}_{pion, proton_projected}.h5`, in `local_analysis/final_production_delta` and `local_analysis/final_production_frontier_streamd`.
- **Datasets** `pt2_forward` and `pt2_backward`: complex, shape (cfg, smear_sink 2, smear_source 2, gamma_sink 3, gamma_source 3, t_src 24, pz 7, tsep 20).
    - Forward: sink at t_src + tsep. Backward: sink at t_src - tsep.
    - tsep = 0 is the source slice.
- **Sources:**
    - 24 t_src per cfg, spacing 4. The offset changes from cfg to cfg (`t_src_list`).
    - At each t_src, 128 spatial positions: x and y at 4 values with spacing 8, z at 8 values with spacing 4 (`x/y/z_src_list`, also shifted per cfg).
    - The pack already holds the **mean over the 128 positions** of each t_src. At q = 0 the source weight exp(-i 2pi q.x_src / 32) is 1.
- **Sink:** momentum projection sum_x exp(-2 pi i p.(x - x_src)/32), with p = (0, 0, pz), pz = 0-6 in units of 2pi/32.
- **Operators:** 2 smearings x 3 gammas = 6, in the order BG5, BG5G4, BG3G5, UG5, UG5G4, UG3G5.
    - B = boosted Gaussian [rho_T 4.5, rho_z 2.25, boost fraction 0.8].
    - U = [1.5, 1.5, 0], Coulomb gauge.
    - Pion: qbar Gamma q. Proton: diquark C Gamma.
- **Proton:** positive-parity projected. Forward uses Tr[(1 + gamma4)/2 M]; backward uses -Tr[(1 - gamma4)/2 M].
- **Pion sign:** the pion correlator is multiplied by c_sign = -1 so the GEVP matrix is positive definite. The proton uses c_sign = +1.
- **Joining the two data sets:** Delta cfgs first, then Frontier cfgs (Frontier cfg numbers + 100000). Frontier cfg 3006 is left out.

### The FF files and the gluon operator

- **Files:** `FF_hyp0-8_w0-9_q0_0_0_ncfg{800, 798}.h5`, one per data set.
    - `FF` has shape (cfg, munu 6, rhosig 6, hyp 9, w 10, t 96), with munu in the order XY, YZ, XZ, XT, YT, ZT.
    - Each 2pt cfg is looked up in the FF file of its own stream by cfg number.
- **Definition:**
    - FF = sum over the spatial volume x of Tr[F_munu(x) W F_rhosig(x - w z) W^dag] at each absolute time slice t, where W is the Wilson line from x to x - w z.
    - F is the clover field strength, F = -(i/8)(Q - Q^dag), made traceless.
    - The gauge field is HYP-smeared in 4D (alpha 0.75, 0.6, 0.3); HYP 8 is used.
- **Operator:** O(cfg, w, t) = FF[XT, XT] + FF[YT, YT] - 2 FF[XY, XY], at HYP 8, for t = 0 ... 95.

### GEVP and the projected 2pt (ratio fits)

- **Matrix:** for each half and pz, C-bar(T) = c_sign x (mean over cfgs and t_src of the 6 x 6 matrix), restricted to the basis of that pz and made Hermitian.
- **GEVP:** with C-bar(t0) = L L^dag, find the eigenvectors of L^-1 C-bar(t_ref) L^-dag. The ground state is the largest eigenvalue.
    - v0 = L^-dag u, normalized so that v0^dag C-bar(t0) v0 = 1. Here t0 = 3, t_ref = 5.
    - There is one v0 per half and pz, made from the ensemble mean and used for every cfg and sample.
    - The time-reflection signs of the backward half (G5G4 against G5 or G3G5) are absorbed by the backward vector.
- **Projection:** c(cfg, t_src, pz, T) = c_sign v0^dag C(cfg, t_src, pz, T) v0, and C2(cfg, pz, T) = its mean over t_src.

### 3pt, method 1 (for each half; s = +1 forward, -1 backward)

- **Connected part:** conn(cfg, w, pz, T, tau) = (1/24) sum over t_src of c(cfg, t_src, pz, T) x O(cfg, w, (t_src + s tau) mod 96).
    - tau is the operator time from the source, 0 ... 19; T is the source-sink separation, 0 ... 19.
- **Operator mean:** O-bar(cfg, w, tau) = (1/24) sum over t_src of O(cfg, w, (t_src + s tau) mod 96), at the same slices as the connected part.
- **Jackknife sample j** (jk_j = mean over the cfgs other than j): C3_j(w, pz, T, tau) = jk_j(conn) - jk_j(C2) x jk_j(O-bar).

### Ratio

- **Halves:** C3 and C2 are averaged over the two halves, sample by sample.
- **Ratio:** R_j(T, tau) = Re C3_j(T, tau) / Re C2_j(T), and NaN for tau > T.
    - At q = 0, p_f = p_i, so the square-root factor of the off-forward ratio is 1.
- **Fit input:** R is then folded and fitted (section 3).

### sGEVP (M_eff fit)

- **Summed operator:** O_sum(cfg, t_src, T, w) = sum over tau = 1 ... T-1 of O(cfg, w, (t_src + s tau) mod 96).
- **Summed 3pt:** K_j(T) = c_sign [jk_j(conn_K) - jk_j(mean over t_src of C_ab) x jk_j(mean over t_src of O_sum)].
    - conn_K = (1/24) sum over t_src of C_ab(cfg, t_src, T) O_sum, where C_ab is the 6 x 6 matrix of the 2pt.
    - C_j(T) = c_sign jk_j(mean over t_src of C_ab), Hermitian part.
- **GEVP, for every sample j, half and T:** C_j(T) u = lambda C_j(t0) u, with u^dag C_j(t0) u = 1.
    - The ground state is the largest lambda for T > t0 and the smallest for T < t0.
- **M_eff:** R_j(T) = Re[u^dag K_j(T) u] / lambda - Re[u^dag K_j(t0) u], averaged over the halves; M_eff,j(T) = R_j(T+1) - R_j(T).

### 2pt fit for E0 (used in S)

- **Data:** C2 of the ratio basis, averaged over the halves per cfg, then jackknifed.
- **Model:** sum over n < 3 of c_n [e^(-E_n t) + e^(-E_n (96 - t))].
- **Windows:**
    - pion t 3-16, 3-15, 3-14, 3-13, 3-12, 2-12, 2-12;
    - proton 4-14 (pz 0-3), 4-12, 4-10, 4-10.
- **Priors:**
    - E0 centered on the lattice dispersion relation from m = 0.1412 (pion) or 0.4989 (proton), width 0.2;
    - gaps log-normal 0.25(40) (pion) or 0.5(5) (proton);
    - c_n 0(1e4).
- **svdcut** 1e-7.
- **Change from the notebook:** every jackknife sample is fitted with its own covariance (section 2), and S uses each sample's E0. The notebook uses the central E0 for every sample, which leaves the E0 error out of S.

## 2. Fit settings common to all three methods

- **Covariance:**
    - Full jackknife covariance of every fitted point of all ten w (cross-w factor 1, nothing removed).
    - Jackknife sample j is fitted with the covariance of the other 1364 cfg (rebuilt without cfg j). This is the same rule for every sample, as in the ratio notebook.
    - The fit of the mean uses the covariance of all cfg; it serves only as the start of the sample fits.
- **Minimizer:** lsqfit, scipy_least_squares, maxit 20000, tol 1e-8.
    - The fit of the mean keeps the lowest chi2 of five starts, dE = 0.15, 0.3, 0.5, 0.8, 1.2. The sGEVP fit has a sixth start (section 4, 2026-11-17).
    - Every jackknife fit starts from the fit of the mean.
    - The number of fits that stop at maxit is reported.
- **svdcut scan:** 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-12.

## 3. Methods 1 and 2: ratio fits

**Input:** the ratio of `matrix_element_analysis_fit_per_pz.ipynb`, sections 0, 1, 3 and 4, unchanged.

- GEVP t0 3, t_ref 5, bases pion_default / proton_default, one GEVP vector from the mean 2pt.
- R = C3 / C2, with C3 and C2 each averaged over the two halves first.
- Folded data: y(tsep, tau) = [R(tsep, tau) + R(tsep, tsep - tau)] / 2, for tau = tau_skip ... floor(tsep / 2).

**Models:**

| Method | Model | Parameters |
|---|---|---|
| 1 separate | R_w = M00_w + Ai_w [e^(-dE tau) + e^(-dE (tsep - tau))] + Afi_w e^(-dE tsep) | 10 + 10 + 10 + 1 = 31 |
| 2 shared | R_w = M00_w [1 + Ai (e^(-dE tau) + e^(-dE (tsep - tau))) + Afi e^(-dE tsep)] | 10 + 1 + 1 + 1 = 13 |

**Priors** (the notebook's `pion_fit` / `proton_fit`, cell 14):

- **M00:** log(-M00_w) = 0(10), so M00 stays negative. Pion pz 5-6 use 0(100).
- **Ai, Afi:** 0(1e4).
- **dE:** log-normal, dE = 0.5(5). Pion pz 5-6 use 0.3(5); proton pz 0 uses 0.5(1.0).
- **Starts:** log(-M00) = 0, Ai = Afi = 0.

**Fixed windows** (the notebook's; points per w, ten w per fit):

| pz | pion tsep, tau_skip | points per w | proton tsep, tau_skip | points per w |
|---|---|---|---|---|
| 0 | 6-15, 3 | 30 | 7-14, 3 | 24 |
| 1 | 5-13, 2 | 29 | 6-13, 2 | 28 |
| 2 | 4-13, 2 | 30 | 6-13, 2 | 28 |
| 3 | 4-13, 2 | 30 | 6-13, 2 | 28 |
| 4 | 4-12, 2 | 25 | 4-10, 2 | 16 |
| 5 | 4-9, 2 | 12 | 4-9, 2 | 12 |
| 6 | 4-9, 2 | 12 | 4-10, 2 | 16 |

## 4. Method 3: sGEVP M_eff fit

**Input:** `matrix_element_analysis_sgevp.ipynb`, sections 0-2, unchanged.

- All six operators at every pz, sGEVP t0 2, M_eff also below t0.
- K(t) = sum over tau = 1 ... t-1 of the method-1 3pt.
- C(t) u = lambda C(t0) u is solved at every t, for every jackknife sample and for each half.
- R(t) = u^dag K(t) u / lambda - u^dag K(t0) u, averaged over the two halves.
- M_eff(t) = R(t+1) - R(t).

**Model:** M_eff,w(t) = A_w + B_w t e^(-dE t). Parameters: 10 + 10 + 1 = 21. A_w plays the role of M00_w.

**Priors:** A_w = -exp(a_w) with a_w = log(-A_w) = 0(10), the same form as the ratio fits' M00 (section 3); B_w 0(10); dE log-normal 0.5(5). Starts: -A = -M_eff at the last t (at least 1e-3), B from the first point.

**Sign of M (changed 2026-11-17, after the v1 reports):**

- Why: with A_w sign free, the sGEVP fit has a second minimum with dE about 0.1-0.2 in which A is an extrapolation far past the window and comes out positive. At proton pz 5, svdcut 1e-2, window 2-6 it has chi2 8.11 against 8.17 for the normal minimum (dE 0.81, A(w 0) -0.69), so the lowest-chi2 rule took it. v1 had 58 pion and 46 proton fits with M(w 0) > 0.
- Rule (2026-11-17, user): M is negative in all three methods, at every hadron, pz and w: M00_w = -exp(a_w) in the ratio fits (as in section 3, unchanged) and A_w = -exp(a_w) in the sGEVP fit.
- Check that M is negative in the data: for each hadron, pz and w, the folded ratio (1365 cfg, default basis, section 1b) averaged over the points of the fixed ratio window (section 3). Script: `scripts/ratio_sign.py`.
- Result: negative at every hadron, pz and w. w 0-7 are at least 3.8 sigma below zero; w 8-9 at least 1.3 sigma (smallest: pion pz 5 w 9).
- The v1 sGEVP fits (sign free), away from the second minimum: no w significantly positive; the largest is +0.040(25) at pion pz 6 w 9 (svdcut 1e-2), and pion pz 5 w 9, pz 6 w 8 are +0.005(9), +0.026(23).
- The sign is forced in every fit: every svdcut, every AIC window, every subset of part C, every jackknife sample.
- Where the sGEVP data want M at or above zero (pion pz 5 w 9, pz 6 w 8-9), the fit pins M just below zero with a far too small error (pion pz 6 w 9: -0.0005(2)). The report lists these w.
    - 2026-11-18, tested and not adopted (your decision: keep the forced sign; more statistics should fix it): freeing the sign of A(w 9) alone at pion pz 6 moves the pinning to w 8 (-0.0009(6)) and gives D(w 9) = -0.45(21); freeing w 8-9 leaves no pinned point but gives D(w 8-9) = -0.40(24), -0.94(49). Nothing was refit.
- Sixth start (2026-11-17, after the v2 reports): with A forced negative, the five starts can all end in a minimum with dE about 0.02-0.04 and A(w 0) pinned at zero, where B t e^(-dE t) is almost a straight line and fits the data by itself. Example: pion pz 6, first 785 cfg, svdcut 1e-2: all five starts give chi2 39.9 and A(w 0) = -0.0006, but a minimum with chi2 12.1, dE 1.02 and A(w 0) = -0.380 exists. About 100 v2 fits were in such a minimum, all in cfg subsets (part C); the 1365-cfg fits were not.
- The sixth start: fit the mean first with A sign free (the v1 fit: A 0(10), the five dE starts, lowest chi2); set every A above -1e-3 to -1e-3; keep B and dE; start the forced-sign fit from these numbers. The fit of the mean keeps the lowest chi2 of the six starts; every jackknife sample starts from it, as before.
- The ratio fits already fit log(-M00) (section 3), so they are not changed or rerun.
- dE has a fixed sign in every fit (the three methods and the 2pt fit): the fits use log(dE), so dE > 0.

**Fixed windows**, both hadrons:

| pz | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| t | 2-12 | 2-12 | 2-9 | 2-8 | 2-7 | 2-6 | 2-6 |
| points per w | 11 | 11 | 8 | 7 | 6 | 5 | 5 |

**Two changes from the notebook:** the covariance is rebuilt per sample (section 2), and the window scan with AIC is done separately (part B), not inside the fixed-window fit.

## 5. Quantities from the fits (every sample, then jackknife)

- **Double ratio:** D(pz, w) = [M(pz, w) / M(pz, 0)] / [M(0, w) / M(0, 0)]. The pz 0 reference comes from the same method, the same svdcut, and the same AIC choice.
- **Single ratio:** S(pz, w) = M(pz, w) / M(0, w) x E(0) / E(pz).
    - E is the E0 of the same jackknife sample, from the 2pt fit of section 1b, on the same cfgs.
    - If the 2pt fit of the mean stops at maxit, it is restarted from the settings (E0 and gaps at the prior centers, equal c), as for proton pz 5.
- **Pull:** (data mean - mean over the samples of the model curve) / jackknife error of the point.
    - Mean pull and rms pull are given per w.
    - A w with |mean pull| > 0.5 is flagged: the band does not pass through the points.

## 6. Part A: svdcut scan, no AIC

- The three methods, fixed windows (sections 3 and 4), all ten svdcuts, both hadrons: 2 x 3 x 7 x 10 = 420 fits.
- Results: M(w), dE, D, S, chi2/dof, number of modes raised, pulls, and fit sheets.

## 7. Part B: window scan with AIC

Done at **svdcut 1e-2 and 1e-8**, the same two cuts for all three methods. Each result is compared with the fixed window at the same cut.

**Ratio window grid**, around the notebook window (f0-l0, s0):

- first tsep f0-1, f0, f0+1
- last tsep l0-2 ... l0+2, at most 16
- tau_skip s0-1, s0, s0+1, at least 1
- first >= 2 tau_skip (every tsep has a point)
- at least 5 tsep values

Number of windows:

| pz | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| pion | 20 | 35 | 25 | 25 | 25 | 19 | 19 |
| proton | 35 | 40 | 40 | 40 | 23 | 19 | 23 |

k = 31 (separate) or 13 (shared).

**sGEVP window grid:**

- first t 2, 3, 4
- last t from first + 4 up to the signal end: the last t up to which |M_eff(w 0)| / error >= 2 at every t, at most 16.
- The signal end is found from the data before any fit.
- Proton signal end pz 0-6: 16, 15, 12, 11, 11, 9, 8, giving 30, 27, 18, 15, 15, 9, 6 windows. Pion: from its own data in the same way.
- **Change of 2026-11-17 (after approval; the plan had not covered it):** the pion signal ends came out as 16, 16, 16, 9, 8, 6, 5, giving 30, 30, 30, 9, 6, 1, 0 windows. pz 6 had no window, because its signal end (5) lies below the last t of the fixed window (6). Rule added: when the signal end is below the last t of the fixed window, the grid runs up to that last t instead. So pion pz 6 gets 1 window (2-6), and pion pz 5-6 have only the fixed window: their AIC result is the fixed-window result. Nothing else changes.
- k = 21.

**Two AIC variants**, both made from the same fits (the second grid is part of the first, so it costs no extra fits):

| Variant | What changes | Ratio windows per pz 0-6 | sGEVP windows per pz 0-6 |
|---|---|---|---|
| B1, full grid | first, last and tau_skip (grids above) | pion 20, 35, 25, 25, 25, 19, 19; proton 35, 40, 40, 40, 23, 19, 23 | proton 30, 27, 18, 15, 15, 9, 6 |
| B2, last tsep only | the last tsep (last t) only; first tsep and tau_skip stay at the notebook values (sGEVP first t 2) | pion 4, 5, 5, 5, 5, 4, 4; proton 5, 5, 5, 5, 5, 4, 5 | proton 11, 10, 7, 6, 6, 4, 3 |

- B2 ratio: last tsep l0-2 ... l0+2, at most 16, at least 5 tsep values. B2 sGEVP: last t 6 ... signal end.
- N_cut in B2 counts the points of the B2 union only.
- The pion sGEVP counts come from its own signal end.

**In every jackknife sample:** fit every window, make the AIC weights, and average M (all w), dE, A or B. D and S are then made from the averaged M of that sample.

**Shown, for each variant (B1, B2) and each svdcut (1e-2, 1e-8), next to the fixed window:**

- AIC result and fixed-window result side by side, for M, D and S.
- The error ratio (AIC / fixed window), per pz and w.
- The weight of every window (mean over samples, and its spread).
- M(w 0) and D(w 4) of every window, with its weight.
- The spread over windows, printed only.
- The single and double ratio of the AIC result alone, every pz against nu, B1 and B2 side by side, for each method and svdcut (added 2026-11-17, at your request).
- The fit sheet of the window with the largest weight.

**Rule (yours):**

1. If B1 makes the jackknife error much larger than the fixed window, use B2.
2. If B2 still makes it much larger, drop AIC and keep the fixed window.

The decision is yours, made from the error-ratio pages. Whatever is decided, B1, B2 and the fixed window all stay plotted, saved and in the report.

**Note on svdcut 1e-2:**

- At 1e-2 the ratio-fit chi2 is far below the number of points: chi2/point 0.02-0.22 at 1365 cfg (separate fraction).
- One more point then adds about 0.1 to chi2 but takes 2 off the 2 N_cut term. So AIC at 1e-2 puts the weight on the windows with the most points, whatever the data do.
- At 1e-8, chi2/point is 0.49-1.21, so there the weights depend on the data. This is why both cuts are run.

## 8. Part C: how the errors scale with the number of cfg (no AIC)

**Subsets:** N = 200, 400, 585, 785, 985, 1185, 1365. Everything is rebuilt from the N cfg: GEVP vectors, 3pt, ratio, K and C, sGEVP, and the 2pt fit for S.

- **(a) In data order:** the first N cfg (Delta by cfg number, then Frontier by cfg number). This is how the results moved as the data came in. Fitted at every svdcut.
- **(b) Random:** 4 draws per N < 1365 (numpy default_rng, seeds 1-4, drawn from the 1365-cfg list), fitted at svdcut 1e-2.
    - Neighbouring cfg are correlated, so (a) alone does not measure the scaling. At 785 cfg the in-order errors were smaller than random-subset errors of the same N.

**Before the fit:** the jackknife error of every fitted point (folded ratio points, M_eff points), all w.

- Shown as err(N) x sqrt(N / 1365) / err(1365), where a flat line at 1 means 1/sqrt(N).
- Median and 16-84 % band per pz.
- Exponent alpha from err ~ N^(-alpha) for N >= 400.

**After the fit:** the same for M(w 0, 4, 8), D(w 2, 4, 6, 8), S(w 0) and dE, for each method.

## 9. Part D: AIC over the operator basis and over the GEVP times (after A-C are seen)

The two are done separately; a joint grid (63 bases x 9 time pairs) would take too long.

### D1: operator basis

- **Bases:** all 63 subsets of the six operators.
    - Ratio: GEVP t0 3, t_ref 5, the same basis at every pz.
    - sGEVP: t0 2.
- Fixed windows of sections 3 and 4, svdcut 1e-2 unless you pick another cut after parts A-C.
- **Weight per sample:** w ~ exp(-chi2 / 2). k and N_cut are the same for every basis.
- **Caveat:** each basis gives different data, so the weights compare chi2 values of different data sets. This is common practice (HadStruc averages over t0 choices), but it is not the same-data model comparison AIC was made for.
- **Rule (yours; wording made clear 2026-11-17):** the decision is about the AIC average over bases as a whole, not about single bases. If averaging over the bases makes the error much larger, or makes the double ratio look unpleasant, we do not average over bases and keep the default basis of each pz. No basis is removed from the average because of its own error. Its plots stay saved and in the report either way.

### D2: GEVP times of the 2pt

- **Ratio:** GEVP t0 = 2, 3, 4 and t_ref = t0 + 1, t0 + 2, t0 + 3, giving 9 pairs (the notebook uses t0 3, t_ref 5).
    - Default bases (pion_default, proton_default).
    - For each pair, the GEVP vectors, projected 2pt, 3pt and ratio are rebuilt (section 1b).
- **sGEVP:** t0 = 2, 3, 4, 5, 6, all six operators, with M_eff at every t (also below t0), so the fixed windows stay the same.
- **Fit settings:** the fixed windows of sections 3 and 4, and the same svdcut as D1.
- **Weight per sample:** w ~ exp(-chi2 / 2).
- **Caveat:** the same as for D1, since each pair gives different data.
- **Rule:** the same as for D1, about the average over the GEVP times as a whole: if averaging over them makes the error much larger or D look unpleasant, we do not average and keep the notebook's GEVP times (ratio t0 3, t_ref 5; sGEVP t0 2). No single pair is removed because of its own error. Its plots stay saved and in the report.

### Decision (2026-11-18, yours, after the part D reports)

- **D1 is not used; the default bases stay.**
    - Averaging over bases makes the error of D(w 4) 1.2-4.6x larger in the pion ratio fits and 1.0-5.4x in the proton ratio fits.
    - In the sGEVP the pz 0 weight goes to UG3G5 alone (chi2 10 against 22-24 for every other basis; weight 0.95 pion, 0.49 proton). For a pion at rest gamma3 gamma5 has no overlap with the ground state, so its M(w 0) is -0.43 against -0.14, and S(w 0) falls to about 0.3.
- **D2 is the final result for all three methods.** Error of D(w 4) against the default: ratio fits 0.99-1.01, sGEVP 1.02-1.11, except pion pz 6.
- **Pion sGEVP at pz 6: the average runs over t0 2-5; t0 6 gets weight 0.**
    - At t0 6 the fit gives M(w 9)/M(w 0) = 0.735. Every other t0 at pion pz 6 gives 0.001-0.002, and every other hadron, pz and t0 (70 fits) gives 0.001-0.09.
    - M(w) falls from -0.82 at w 0 to -0.38 at w 5 and rises again to -0.60 at w 9.
    - Its chi2 is the lowest (3.3 against 6.9-11.6), so it took weight 0.80, giving D(w 4) 1.04(1.05) and D(w 8) 5.5(6.2).
    - It is left out because it fails a physics check (M must fall towards zero at large w), not because of its error. It stays on the weights page with weight 0.
    - With t0 2-5: D(w 4) 0.817(63), D(w 8) 0.201(100), S(w 0) 0.334(102); t0 2 alone gives 0.821(60), 0.221(94), 0.417(112).
- Setting: `D2_SGEVP_T0_ONLY = {("pion", 6): [2, 3, 4, 5]}` in `scripts/settings_final.py`.

## 10. Report and files

**Moved 2026-11-18 (your request):** every report, fit sheet, `plots*/`, `preview_v2A/` and `results/` named below as in `local_analysis/final_analysis/` is now in `~/Desktop/1365cfg_fit_test/`, with the same names. Only `FIT_PLAN.md`, `FIT_PLAN.pdf` and `scripts/md_to_pdf.py` stay in the repo.

One main report and one fit-sheet file per hadron, in `local_analysis/final_analysis/`:

- **`three_fits_report_{pion,proton}_1365cfg.pdf`** (about 110 pages):
    - settings (this plan)
    - before the fit: ratio sheets and M_eff sheets per pz, 2 x 5 panels (one per w)
    - error scaling of the data
    - part A: summaries against svdcut, ITD at every cut, three-method comparison, numbers
    - part B: AIC (B1 and B2, svdcut 1e-2 and 1e-8) against the fixed window
    - part C: error scaling of the fit results
    - part D: operator AIC (D1) and GEVP-time AIC (D2), added later
    - final result (added 2026-11-18): D2 for each method, single and double ratio, and the three methods together
- **`three_fits_fit_sheets_{pion,proton}_1365cfg.pdf`** (about 260 pages): every fit of part A (3 methods x 7 pz x 10 svdcuts) and the largest-weight window of B1 and of B2. One panel per w, with points, bands and pulls.

**v1 kept (2026-11-17):** the reports with the sign-free sGEVP fit stay as `three_fits_report_{pion,proton}_1365cfg_v1.pdf` and `three_fits_fit_sheets_{pion,proton}_1365cfg_v1.pdf`, their PNGs in `plots_v1/`, and their sGEVP fit files in `results/fits_v1_signfree/`. **v2 kept (2026-11-17):** sign forced, five starts: `*_v2.pdf`, `plots_v2/`, `results/fits_v2_five_starts/`.

**Everything is shown:** every variant (fixed window, B1, B2, operator AIC, GEVP-time AIC) and every svdcut goes into the report, including variants that you later drop. Every figure is also saved as its own PNG in `local_analysis/final_analysis/plots/{pion,proton}/`, in subfolders by part (`before_fit`, `A_svdcut`, `B_aic`, `C_scaling`, `D_operators`, `D_gevp_times`).

Other files:

- Scripts, job lists and logs go in `~/Desktop/1365cfg_fit_test/scripts/` (moved out of the code directory on 2026-11-18, at your request); only `scripts/md_to_pdf.py` stays in the repo. The scripts write results, plots and reports to `~/Desktop/1365cfg_fit_test/` as well.
- Fit results (h5 / npz, git-ignored) go in `local_analysis/final_analysis/results/`.
- The final result (2026-11-18) per hadron: `results/final/final_{pion,proton}_1365cfg.npz`, with M, D, S and dE of every jackknife sample for each method (D2 average), and E0 of the 2pt fit.
- The large rebuilt inputs (ratio and M_eff per subset and per basis, about 30 GB) go in `~/Desktop/1365cfg_report/three_fits_inputs/`; they can be rebuilt.
- No notebook is changed. Each input step is first checked to reproduce the notebook's own output.

## 11. Run time (estimated, heavy builds one at a time)

| Part | Time |
|---|---|
| A | 0.7 h |
| B, two svdcuts | 3.5 h |
| C (builds 1.7 h, fits 1.8 h) | 3.5 h |
| A-C in total | about 8 h; report v1 then |
| D1 (63-basis inputs + 2.4 M ratio fits) | about 4.5 h more |
| D2 (9 GEVP pairs, 5 sGEVP t0) | about 1 h more |
