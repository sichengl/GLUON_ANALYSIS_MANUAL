# Fit plan: ratio fits and sGEVP M_eff fit, 1365 cfg

**Status: draft 3, 2026-11-17. Not approved yet.** When you approve it, this line changes to "approved" with the date, and the work follows the plan exactly. Any later change is written here first, with its date.

Changes: draft 2 added the definitions, the window AIC and the operator AIC. Draft 3 adds a second AIC variant (last tsep only) and says that every variant is plotted, saved and reported, even one that is dropped.

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

## 2. Fit settings common to all three methods

- **Covariance:**
    - Full jackknife covariance of every fitted point of all ten w (cross-w factor 1, nothing removed).
    - Jackknife sample j is fitted with the covariance of the other 1364 cfg (rebuilt without cfg j). This is the same rule for every sample, as in the ratio notebook.
    - The fit of the mean uses the covariance of all cfg; it serves only as the start of the sample fits.
- **Minimizer:** lsqfit, scipy_least_squares, maxit 20000, tol 1e-8.
    - The fit of the mean keeps the lowest chi2 of five starts, dE = 0.15, 0.3, 0.5, 0.8, 1.2.
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

**Priors:** A_w 0(10), sign free; B_w 0(10); dE log-normal 0.5(5). Starts: A = M_eff at the last t, B from the first point.

**Fixed windows**, both hadrons:

| pz | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| t | 2-12 | 2-12 | 2-9 | 2-8 | 2-7 | 2-6 | 2-6 |
| points per w | 11 | 11 | 8 | 7 | 6 | 5 | 5 |

**Two changes from the notebook:** the covariance is rebuilt per sample (section 2), and the window scan with AIC is done separately (part B), not inside the fixed-window fit.

## 5. Quantities from the fits (every sample, then jackknife)

- **Double ratio:** D(pz, w) = [M(pz, w) / M(pz, 0)] / [M(0, w) / M(0, 0)]. The pz 0 reference comes from the same method, the same svdcut, and the same AIC choice.
- **Single ratio:** S(pz, w) = M(pz, w) / M(0, w) x E(0) / E(pz).
    - E is the central E0 of the notebook's 2pt fit (section 2a settings) on the same cfgs.
    - If that fit of the mean stops at maxit, it is restarted from the settings (E0 and gaps at the prior centers, equal c), as for proton pz 5.
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

## 9. Part D: AIC over the operator basis (after A-C are seen)

- **Bases:** all 63 subsets of the six operators.
    - Ratio: GEVP t0 3, t_ref 5, the same basis at every pz.
    - sGEVP: t0 2.
- Fixed windows of sections 3 and 4, svdcut 1e-2 unless you pick another cut after parts A-C.
- **Weight per sample:** w ~ exp(-chi2 / 2). k and N_cut are the same for every basis.
- **Caveat:** each basis gives different data, so the weights compare chi2 values of different data sets. This is common practice (HadStruc averages over t0 choices), but it is not the same-data model comparison AIC was made for.
- **Rule (yours):** dropped if it increases the error a lot or makes the double ratio look unpleasant. Its plots stay saved and in the report either way.

## 10. Report and files

One main report and one fit-sheet file per hadron, in `local_analysis/final_analysis/`:

- **`three_fits_report_{pion,proton}_1365cfg.pdf`** (about 110 pages):
    - settings (this plan)
    - before the fit: ratio sheets and M_eff sheets per pz, 2 x 5 panels (one per w)
    - error scaling of the data
    - part A: summaries against svdcut, ITD at every cut, three-method comparison, numbers
    - part B: AIC (B1 and B2, svdcut 1e-2 and 1e-8) against the fixed window
    - part C: error scaling of the fit results
    - part D (added later)
- **`three_fits_fit_sheets_{pion,proton}_1365cfg.pdf`** (about 260 pages): every fit of part A (3 methods x 7 pz x 10 svdcuts) and the largest-weight window of B1 and of B2. One panel per w, with points, bands and pulls.

**Everything is shown:** every variant (fixed window, B1, B2, operator AIC) and every svdcut goes into the report, including variants that you later drop. Every figure is also saved as its own PNG in `local_analysis/final_analysis/plots/{pion,proton}/`, in subfolders by part (`before_fit`, `A_svdcut`, `B_aic`, `C_scaling`, `D_operators`).

Other files:

- Scripts go in `local_analysis/final_analysis/scripts/`.
- Fit results (h5 / npz, git-ignored) go in `local_analysis/final_analysis/results/`.
- The large rebuilt inputs (ratio and M_eff per subset and per basis, about 30 GB) go in `~/Desktop/1365cfg_report/three_fits_inputs/`; they can be rebuilt.
- No notebook is changed. Each input step is first checked to reproduce the notebook's own output.

## 11. Run time (estimated, heavy builds one at a time)

| Part | Time |
|---|---|
| A | 0.7 h |
| B, two svdcuts | 3.5 h |
| C (builds 1.7 h, fits 1.8 h) | 3.5 h |
| A-C in total | about 8 h; report v1 then |
| D (63-basis inputs + 2.4 M ratio fits) | about 4.5 h more |
