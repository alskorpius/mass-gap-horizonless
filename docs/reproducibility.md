# Reproducibility map

Every number in the paper maps to one command and one result file. Commands are run from the repository root with the interpreter of the virtual environment created from `requirements.txt`. Units are geometric (G = c = 1) unless a result is quoted in solar masses, kilometres, kilohertz or seconds.

## Environment

Python 3.13.6; NumPy 2.3.5; SciPy 1.17.0; SymPy 1.14.0; mpmath 1.3.0; Matplotlib 3.10.8. Every script is single-threaded and deterministic: there is no random number generation anywhere in this repository, so repeated runs differ only through integrator tolerances and the pinned library versions.

Scripts locate the repository root by looking for `CITATION.cff` next to a `src/` directory, and every other path is derived from that, so they can be started from any working directory.

## Number → script → result file

### The tower and its curvature cap

| Number in the paper | Command | Result file |
|---|---|---|
| Truncated-tower interior exponent p(N) = 2 − 3/N: N = 2 / 6 / 100 → 0.5 / 1.5 / 1.97, matching the analytic value to 5–7 digits | `python src/stage6_microphysics/core_scale_solve.py` | `data/stage6_microphysics/core_scale_results.json` (`truncated_tower_exponents`) |
| Geometric resummation is identically Hayward with ℓ² = α | `python src/stage6_microphysics/core_scale_solve.py` | `data/stage6_microphysics/core_scale_results.json` (`geometric_check`) |
| Exponential resummation: soft logarithmic singularity, K = 4286 at r = 2.8·10⁻³ rising to 17250 at r = 1.9·10⁻⁵ | `python src/stage6_microphysics/core_scale_solve.py` | `data/stage6_microphysics/core_scale_results.json` (`exponential_resummation`) |
| ε_c = 3/(8πα); ε_c = 1 ⇔ α = 3/(8π) = 0.1194 | `python src/stage6_microphysics/core_scale_solve.py` | `data/stage6_microphysics/core_scale_results.json` (`eps_c_identification`) |
| FRW cap in closed form ψ = S/(1 + αS) < 1/α, giving H² = ε(1 − ε/ε_c) with ε_c = 1/α at first order | `python src/friedmann_tower/friedmann_tower.py` | `data/friedmann_tower/friedmann_tower_results.json` (`resummed`, `limits`) |
| Curvature invariants of the tower background | `python src/friedmann_tower/step2_invariants.py` | `data/friedmann_tower/step2_invariants.json` |

### Mini-theorem E4

| Number in the paper | Command | Result file |
|---|---|---|
| E2 evaluated on every E1 root is strictly positive, so a triple root (κ₋ = 0) is impossible in any tower with α₁ = 1, αₙ ≥ 0: geometric ψ\* = 1/3 → E2 = +0.2222; αₙ = nαⁿ⁻¹ ψ\* = 0.200 → +0.1111; logarithmic ψ\* = 0.5336 → +0.4582 | `python src/stage_E4_tower_sigma/tower_sigma.py` | `data/stage_E4_tower_sigma/tower_sigma_results.json` |
| Two-term and three-term mixtures: E2 stays in +0.089…+0.427 | `python src/stage_E4_tower_sigma/tower_sigma.py` | same file (`twoterm_*`, mixture entries) |
| Cross-check of the counterexample class from the companion paper | `python src/verify_triple_root_monotone/triple_root_check.py` | `data/verify_triple_root_monotone/triple_root_check.json` |
| Independent audit of the companion paper's profile family | `python src/verify_review1/review1_checks.py` | `data/verify_review1/` |

### Horizonlessness criterion and the threshold range

| Number in the paper | Command | Result file |
|---|---|---|
| Shape factor c_ext ∈ [0.18, 0.39], with n = 6 giving 0.2314 | `python src/parallel_massgap_inversion/massgap_inversion.py` | `data/parallel_massgap_inversion/massgap_inversion.json` (`c_ext`) |
| Threshold M_crit as a range (low / n = 6 / high): ε_c = 1 × nuclear → 8.45 / 10.87 / 18.32 M☉; 2 × → 5.98 / 7.68 / 12.95; 4.7 × → 3.90 / 5.01 / 8.45; 10 × → 2.67 / 3.44 / 5.79; 22 × → 1.80 / 2.32 / 3.90 | `python src/parallel_massgap_inversion/massgap_inversion.py` | `data/parallel_massgap_inversion/massgap_inversion.json` (`rows`), `.csv` |
| Thresholds at ε_c = 5 and 29 × nuclear: 8.43 / 4.95 / 3.87 M☉ (n = 2 / 6 / 40) and 3.50 / 2.06 / 1.61 M☉ | `python src/polar_sectors/p19_tower_remnant.py` | `data/polar_sectors/p19_tower_remnant.json` (`thresholds`) |
| Horizonlessness of a 2.7 M☉ object requires ε_c ≲ 48.8 (n = 2), 16.8 (n = 6), 10.3 (n = 40) × nuclear | `python src/polar_sectors/p19_tower_remnant.py` | `data/polar_sectors/p19_tower_remnant.json` (`condition 2.7`) |
| Extremal core scan | `python src/stage4_extremal_core/extremal_core.py` | `data/stage4_extremal_core/stage4_summary.json`, `extremal_snapshots.csv` |

### Echoes and time delays

| Number in the paper | Command | Result file |
|---|---|---|
| M_ext(n = 6, ε_c = 1) = 0.23136; two photon spheres throughout, r_ph,outer → 3M to 4–5 digits; compactness 2M/R₉₉ ≤ 0.352 | `python src/stage5_horizonless_observables/horizonless_observables.py` | `data/stage5_horizonless_observables/stage5_summary.json`, `horizonless_scan.csv` |
| Echo time τ/(2M) = 12.20 / 18.72 / 55.64 / 379.31 at M/M_ext = 0.88 / 0.95 / 0.995 / 0.9999, with f_min falling from 8.2·10⁻² to 6.64·10⁻⁵ | `python src/stage5_horizonless_observables/horizonless_observables.py` | `data/stage5_horizonless_observables/stage5_summary.json` (`rows`) |
| The same band computed independently: τ/2M = 12.24 / 18.72 / 55.64 / 379.31 | `python src/parallel_gw_echo_bands/echo_bands.py` | `data/parallel_gw_echo_bands/echo_bands_summary.json`, `.csv` |
| Echo trains and the time-domain versus geometric-optics agreement | `python src/stage_E1_qnm_echo/echo_towers.py` | `data/stage_E1_qnm_echo/echo_towers_results.json` |
| Amplitude constraints against the O3 limit A ≤ 0.42: M/M_ext = 0.95 gives τ = 5.53 ms at 30 M☉ with A(|R| = 1) = 0.283, which passes | `python src/stage_E2_echo_constraints/echo_constraints.py` | `data/stage_E2_echo_constraints/echo_constraints_summary.json`, `echo_constraints_table.csv` |
| Corrected echo amplitudes (the earlier values 0.081 / 0.283 / 0.321 were a windowing artefact) | `python src/self_audit/self_audit.py part2` | `data/self_audit/` |

### Tidal deformability

| Number in the paper | Command | Result file |
|---|---|---|
| Validation on a polytrope: k₂ = 0.06608, Λ = 285.70 at C = 0.17283 | `python src/parallel_tidal_deformability/tidal_deformability.py` | `data/parallel_tidal_deformability/tidal_summary.json` (`validation_polytrope`) |
| Geometric horizonless branch: Λ̃ ≈ 0.03–0.16, k₂ = 0.6–1.6·10⁻³ | `python src/parallel_tidal_deformability/tidal_deformability.py` | `data/parallel_tidal_deformability/tidal_summary.json` (`horizonless_models`), `tidal_results.csv` |
| Corner branch n_t = 4: Λ = 93.49 at 1.679 M☉ falling to ≈ 0 by 3.47 M☉, with R = 11.40 km at the onset | `python src/polar_sectors/lambda_curve.py` | `data/polar_sectors/lambda_curve_results.json` (`branches`) |
| SLy reference: Λ(1.4) = 349.52, M_max = 2.067 M☉ | `python src/polar_sectors/lambda_curve.py` | `data/polar_sectors/lambda_curve_results.json` (`sly`) |
| GW190814 secondary: Λ(2.59) = 1.68, Λ(2.67) = 1.28, R(2.6) = 10.81 km (n_t = 4); SLy does not reach 2.6 M☉ | `python src/polar_sectors/lambda_curve.py` | `data/polar_sectors/lambda_curve_results.json` (`gw190814`) |

### f-modes with validation

| Number in the paper | Command | Result file |
|---|---|---|
| Validation anchor A (the RelModPy author's own anchor): Re(ωM) = 0.1708438 against 0.171, a deviation of 0.09 %; Im(ωM) = 6.1916·10⁻⁵ | `python src/polar_sectors/fmode.py` | `data/polar_sectors/fmode_results.json` (`validation_A`) |
| Validation anchor B (literature): SLy at 1.4 M☉ gives f = 1.8992 kHz, inside the 1.8–2.1 kHz band, with τ = 1.25 s and R = 11.549 km | `python src/polar_sectors/fmode.py` | `data/polar_sectors/fmode_results.json` (`validation_B`, `sly_fmodes`) |
| SLy series: 1.8 M☉ → 2.170 kHz (R = 11.253 km, τ = 0.41 s); 2.0 M☉ → 2.385 kHz | `python src/polar_sectors/fmode.py` | `data/polar_sectors/fmode_results.json` (`sly_fmodes`) |
| Corner branch: 1.8 → 2.161 kHz, 2.2 → 2.298 kHz, 3.0 → 1.926 kHz, all stable, the sequence non-monotonic and flagged | `python src/polar_sectors/fmode.py` | `data/polar_sectors/fmode_results.json` (`hybrid_branch_fmodes`) |
| The 2.6 M☉ root is not localised in the original scan window: minimum \|A_in\| = 8.54·10⁻² | `python src/polar_sectors/fmode.py` | `data/polar_sectors/fmode_results.json` (`hybrid_26`) |
| Convergence rerun with the widened window: f(2.59 / 2.63 / 2.67 M☉) = 2.222 / 2.204 / 2.184 kHz, all stable, \|A_in\| = 1.6·10⁻¹⁰…4.7·10⁻¹⁰, insensitive to Γ₁ over 10³–10⁵ | `python src/polar_sectors/fmode26_retry.py` | `data/polar_sectors/fmode26_retry.json` |
| Equation-of-state interface and its SLy validation | `python src/lab_interface/lab_interface.py` | `data/lab_interface/lab_interface_results.json` |

### Baryon budget and the GW170817 remnant

| Number in the paper | Command | Result file |
|---|---|---|
| Baryonic mass on the corner branch: M_b(2.7 gravitational) = 2.920 M☉, M_b,max = 3.330 M☉; the GW170817 remnant (M_b ≈ 2.8–3.0) fits on the branch | `python src/polar_sectors/baryonic_budget.py` | `data/polar_sectors/baryonic_budget.json` (`comparison`) |
| P13, additional constraints from the late-time Chandra excess: L ≈ 5.74·10³⁹ erg s⁻¹ | `python src/polar_sectors/p13_remnant.py` | `data/polar_sectors/p13_remnant.json` |
| P17, spin-down and rotational-energy budget: L_sd(B = 10¹² G) = 2.45·10⁴⁰ erg s⁻¹ at M = 2.9 M☉, P = 5 ms; B_quiet = 3.5·10¹¹ G; E_rot = 1.86·10⁵¹ erg | `python src/polar_sectors/p17_gwbudget.py`, `python src/polar_sectors/outcomeA_constraints.py` | `data/polar_sectors/p17_gwbudget.json`, `outcomeA_constraints.json` |
| P18, bar-mode and r-mode emission: r-mode at α = 10⁻² gives h_rss = 3.41·10⁻²⁸, far below the search limit | `python src/polar_sectors/p18_barmode.py` | `data/polar_sectors/p18_barmode.json` |
| P19, geometric reading of the remnant: throat R\* = 10.3 km = 1.29·2M, χ = 0.194, E_rot = 3.90·10⁵² erg; Blandford-Znajek power 3.48·10⁴⁸ erg s⁻¹ needs B = 2.25·10¹⁵ G | `python src/polar_sectors/p19_tower_remnant.py` | `data/polar_sectors/p19_tower_remnant.json` |
| Ergoregion of the rotating solution | `python src/ergoregion_rotation/ergoregion_rotation.py` | `data/ergoregion_rotation/ergoregion_results.json` |

### Quasinormal modes

| Number in the paper | Command | Result file |
|---|---|---|
| Axial QNM of the horizonless branch | `python src/stage_E1_qnm_echo/axial_qnm.py` | `data/stage_E1_qnm_echo/` |
| Time-domain QNM, used where the WKB approximation is unreliable | `python src/stage_E1_qnm_echo/timedomain_qnm.py` | `data/stage_E1_qnm_echo/` |
| Frequency-domain (Wronskian) cross-check | `python src/stage_E1_qnm_echo/wronskian_qnm.py` | `data/stage_E1_qnm_echo/` |
| The conclusion that the QNM shift is set by the halo mass rather than by the core | `python src/self_audit/self_audit.py part1` | `data/self_audit/` |

### Figures

| Figure | Command | Output |
|---|---|---|
| QNM shifts against the small-core law | `python figures/make_figures.py` | `figures/fig_qnm_shifts.pdf`, `.png` |
| Echo train | same | `figures/fig_echo_train.pdf`, `.png` |
| Shell stability | same | `figures/fig_stability.pdf`, `.png` |
| Critical stiffness Gamma_c | same | `figures/fig_gamma_c.pdf`, `.png` |
| Ergoregion against spin | same | `figures/fig_ergo_spin.pdf`, `.png` |

`make_figures.py` recomputes nothing except the waveform for the echo figure; every other panel is read from `data/`. Its inputs are `data/stage_E1_qnm_echo/timedomain_qnm_results.json`, `data/corrections_e1_e5/corrections_results.json`, `data/stage_E3_shell_stability/shell_stability_results.json`, `data/nonlinear_shell_eos/`, and `data/ergoregion_rotation/ergoregion_results.json`, so those five must be produced first.

### Supporting modules added for completeness

| Module | Why it is here | Output |
|---|---|---|
| `src/stage4_extremal_core/extremal_core.py` | `p19_tower_remnant.py` reads its summary | `data/stage4_extremal_core/` |
| `src/ergoregion_rotation/ergoregion_rotation.py` | read by P19 and by the ergoregion figure | `data/ergoregion_rotation/` |
| `src/stage_E5_rotating/rotating_audit.py` | read by `self_audit.py part5` | `data/stage_E5_rotating/` |
| `src/two_phase_return/two_phase.py` | read by `self_audit.py part3` | `data/two_phase_return/` |
| `src/corrections_e1_e5/corrections.py` | supplies the accepted QNM shift for the figure | `data/corrections_e1_e5/` |
| `src/stage_E3_shell_stability/shell_stability.py`, `src/nonlinear_shell_eos/nonlinear_eos.py` | inputs to two figures | `data/stage_E3_shell_stability/`, `data/nonlinear_shell_eos/` |
| `src/bhp2_family/triple_root_monotone.py`, `src/approach_map/`, `src/baseline/` | the companion paper's profile family, used by the cross-audit in `verify_review1` | library only |

### One input that is not generated here

`data/stage_E1_qnm_echo/_waveform_schw.npy` is a cached Schwarzschild time-domain waveform used by `corrections.py` for its FFT-peak diagnostic. No script in this repository writes it; it is carried over from the original time-domain run and is included as data. Deleting it makes `corrections.py` fail rather than silently change a number.

### Self-audit

`self_audit.py` takes a part number: `part1` (the QNM shift and the halo criterion, verdict "CONFIRMED (triple check)", Leaver control to 1.8e-3), `part2` (the corrected echo amplitudes), `part3` (needs `two_phase_return`), `part4`. `part5_kretschmann.py` is a separate entry point and needs `stage_E5_rotating`.

## Order of execution

Most scripts are independent. The dependencies that do constrain the order are:

```
python src/lab_interface/lab_interface.py                   # first: EOS interface
python src/polar_sectors/lambda_curve.py                    # needs lab_interface
python src/polar_sectors/fmode.py                           # needs lab_interface + src/third_party/relmodpy
python src/polar_sectors/fmode26_retry.py                   # needs fmode.py (imports its adapters)
python src/polar_sectors/baryonic_budget.py                 # needs lab_interface
python src/stage4_extremal_core/extremal_core.py            # before p19
python src/stage5_horizonless_observables/horizonless_observables.py   # before p19
python src/ergoregion_rotation/ergoregion_rotation.py       # before p19
python src/polar_sectors/p19_tower_remnant.py               # needs the three above
python src/stage_E1_qnm_echo/echo_towers.py                 # before self_audit part2
python src/self_audit/self_audit.py part1 … part4           # needs stage_E1 results
```

Everything else can be run in any order.

## A discrepancy found during acceptance

The working notes and an earlier draft quote **Λ(1.4) = 343** for the SLy reference. The code in this repository computes **349.52**, a difference of 1.9 %. The two agree bit-for-bit between this repository and the original working directory, so the difference is in the text, not in the migration. Both values satisfy the GW170817 constraint Λ(1.4) < 580 and no conclusion changes, but the figure quoted in the paper should be 349.5.

## Results that live only in logs

`p13_remnant.py`, `p17_gwbudget.py`, `p18_barmode.py`, `outcomeA_constraints.py` and the `stage_E1_qnm_echo` scripts print part of their diagnostics to stdout; those transcripts are kept in `logs/`. The numbers quoted in the paper are taken from the JSON files listed above, not from the logs.

## Determinism and numerical limits

No random number generator is used. Results depend on the NumPy and SciPy versions only through integrator tolerances, and those versions are pinned in `requirements.txt`. The f-mode root finding is the most delicate step: the third-party Muller iteration diverges from imprecise starting points and was replaced by an |A_in| scan followed by two-dimensional Nelder-Mead minimisation, which is why the scan window matters and why the 2.6 M☉ case failed until the window was widened past ωM = 0.178. See [`limitations.md`](limitations.md).
