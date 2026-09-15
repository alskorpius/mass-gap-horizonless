# mass-gap-horizonless

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22780350.svg)](https://doi.org/10.5281/zenodo.22780350)

Code and data for the paper **"A horizonless branch in the mass gap and its gravitational-wave signatures"**.

A tower of curvature terms caps curvature without invoking exotic matter and produces a branch of horizonless ultracompact objects that can populate the lower black-hole mass gap. The branch is constrained, not detected: its upper mass is a range rather than a number, its echo amplitudes are already partly excluded by O3, and its f-mode spectrum does not violate the universal f–Λ–M relation, so asteroseismology cannot discriminate it. The discriminators are the mass range and the tidal deformability.

## Install and run

```
git clone https://github.com/alskorpius/mass-gap-horizonless
cd mass-gap-horizonless
python -m venv .venv && .venv/Scripts/activate      # POSIX: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.13.6, five dependencies, no compiled extensions. Every script is run from the repository root and locates its neighbours relative to its own file:

```
python src/stage6_microphysics/core_scale_solve.py          # the curvature cap
python src/stage_E4_tower_sigma/tower_sigma.py              # mini-theorem E4
python src/parallel_massgap_inversion/massgap_inversion.py  # the threshold mass range
python src/stage5_horizonless_observables/horizonless_observables.py   # echoes
python src/polar_sectors/fmode.py                           # f-modes with validation
```

The full ordered run is in [`docs/reproducibility.md`](docs/reproducibility.md).

## The threshold is a range

The central observational statement of the paper is a *range*, not a single mass. The branch criterion is M·√ε_c ⋛ c_ext(n), and the shape factor c_ext runs over [0.18, 0.39] (n = 6 gives 0.2314), which spreads the upper mass of the horizonless branch by a factor 2.2. At ε_c = 4.7 × nuclear density the threshold is 3.90 / 5.01 / 8.45 M☉ for the low, n = 6, and high shape factors; at 1 × nuclear it is 8.45 / 10.87 / 18.32 M☉ and at 22 × nuclear 1.80 / 2.32 / 3.90 M☉. Quoting a single number here would misrepresent the calculation.

## Figure and table map

| Item in the paper | Script | Output |
|---|---|---|
| Curvature cap, truncated-tower exponent p(N) = 2 − 3/N | `src/stage6_microphysics/core_scale_solve.py` | `data/stage6_microphysics/core_scale_results.json` |
| Tower on FRW: ψ = S/(1 + αS), cap 1/α, H² = S(1 − S/ε_c) | `src/friedmann_tower/friedmann_tower.py`, `step2_invariants.py` | `data/friedmann_tower/friedmann_tower_results.json`, `step2_invariants.json` |
| Mini-theorem E4: E2 > 0 on every E1 root, so κ₋ = 0 is impossible in the tower | `src/stage_E4_tower_sigma/tower_sigma.py` | `data/stage_E4_tower_sigma/tower_sigma_results.json` |
| Counterexample class cross-check | `src/verify_triple_root_monotone/triple_root_check.py` | `data/verify_triple_root_monotone/triple_root_check.json` |
| Threshold M(ℓ₀) as a range, c_ext ∈ [0.18, 0.39] | `src/parallel_massgap_inversion/massgap_inversion.py` | `data/parallel_massgap_inversion/massgap_inversion.json`, `.csv` |
| Horizonless observables: two photon spheres, 2M/R₉₉ ≤ 0.352, echo times | `src/stage5_horizonless_observables/horizonless_observables.py` | `data/stage5_horizonless_observables/stage5_summary.json`, `horizonless_scan.csv` |
| Echo bands τ/2M = 12.24 / 18.72 / 55.64 / 379.31 | `src/parallel_gw_echo_bands/echo_bands.py` | `data/parallel_gw_echo_bands/echo_bands_summary.json`, `.csv` |
| Echo trains and time-domain agreement | `src/stage_E1_qnm_echo/echo_towers.py`, `timedomain_qnm.py`, `wronskian_qnm.py`, `axial_qnm.py` | `data/stage_E1_qnm_echo/echo_towers_results.json` |
| Echo constraints against O3 limits | `src/stage_E2_echo_constraints/echo_constraints.py` | `data/stage_E2_echo_constraints/` |
| Λ(M) of the corner branch; GW190814 secondary | `src/polar_sectors/lambda_curve.py` | `data/polar_sectors/lambda_curve_results.json` |
| Λ validation on a polytrope (k₂ = 0.066, Λ = 286 at C = 0.173) | `src/parallel_tidal_deformability/tidal_deformability.py` | `data/parallel_tidal_deformability/tidal_summary.json`, `tidal_results.csv` |
| f-modes with both validation anchors | `src/polar_sectors/fmode.py` | `data/polar_sectors/fmode_results.json` |
| f-mode convergence rerun at the GW190814 mass | `src/polar_sectors/fmode26_retry.py` | `data/polar_sectors/fmode26_retry.json` |
| Baryon budget: M_b(2.7) = 2.92, M_b,max = 3.33 M☉ | `src/polar_sectors/baryonic_budget.py` | `data/polar_sectors/baryonic_budget.json` |
| Equation-of-state interface and SLy validation | `src/lab_interface/lab_interface.py` | `data/lab_interface/lab_interface_results.json` |
| P13 — GW170817 remnant constraints | `src/polar_sectors/p13_remnant.py` | `data/polar_sectors/` |
| P17 — spin budget | `src/polar_sectors/p17_gwbudget.py` | `data/polar_sectors/` |
| P18 — bar-mode | `src/polar_sectors/p18_barmode.py` | `data/polar_sectors/` |
| P19 — geometric reading of the remnant | `src/polar_sectors/p19_tower_remnant.py` | `data/polar_sectors/` |
| Independent audit of the companion paper's claims | `src/verify_review1/review1_checks.py` | `data/verify_review1/` |
| Self-audit of the echo amplitudes | `src/self_audit/self_audit.py` | `data/self_audit/` |

The complete map, number by number, is [`docs/reproducibility.md`](docs/reproducibility.md).

## Third-party code

The f-mode solver is **RelModPy** by Fabian Gittins (MIT), vendored unmodified in `src/third_party/relmodpy/` together with its licence. It implements Lindblom & Detweiler (1985) and Andersson, Kokkotas & Schutz (1995). This work's own contribution around it is the equation-of-state adapters and the replacement of the library's Muller root finder — which diverges from imprecise starting points — by an |A_in| scan followed by two-dimensional Nelder-Mead minimisation. See [`src/third_party/README.md`](src/third_party/README.md).

## Layout

```
src/stage6_microphysics/, src/friedmann_tower/   the tower and its curvature cap
src/stage_E4_tower_sigma/                        mini-theorem E4
src/parallel_massgap_inversion/                  the threshold mass range
src/stage5_horizonless_observables/              photon spheres, echo times
src/stage_E1_qnm_echo/, src/stage_E2_echo_constraints/   QNM, echo trains, O3 limits
src/polar_sectors/                               Lambda(M), f-modes, baryon budget, P13-P19
src/parallel_tidal_deformability/                Lambda validation
src/lab_interface/                               equation-of-state interface
src/self_audit/, src/verify_review1/, src/verify_triple_root_monotone/   internal audits
src/third_party/relmodpy/                        RelModPy (MIT, F. Gittins) — unmodified
src/bhp2_family/, src/approach_map/, src/baseline/   the companion paper's family, for the cross-audit
data/                                            results as JSON and CSV
figures/                                         figure output
logs/                                            raw stdout logs of the runs
docs/                                            reproducibility map and limitations
```

## Limitations

The threshold is a range spanning a factor 2.2, not a single mass. The echo amplitudes originally reported were a windowing artefact and have been corrected; the clean values are 0.995 → [0.59, 0.69] and 0.95 → [0.32, 0.56], with the 0.88 case unreliable. The f-mode computation depends on third-party machinery whose root finder had to be replaced, and the corner-branch sequence is non-monotonic in a way that may indicate interface-mode contamination. The universal f–Λ–M relation is not violated, so f-modes do not discriminate the branch. The branch does not explain the emptiness of the 3.5–5 M☉ region and makes no prediction about how populated it is. Full discussion in [`docs/limitations.md`](docs/limitations.md).

## Related

The inner-horizon structure that this work's mini-theorem rules out for the tower is analysed in the companion paper: https://github.com/alskorpius/inner-extremal-rbh

## Citation

Oleh Popenkov, *A horizonless branch in the mass gap and its gravitational-wave signatures* (2026).
Code and data: https://github.com/alskorpius/mass-gap-horizonless — see [`CITATION.cff`](CITATION.cff).

To cite this repository, use the archived release rather than the URL — see [`CITATION.cff`](CITATION.cff):

| | DOI |
|---|---|
| all versions (resolves to the latest) | [10.5281/zenodo.22780350](https://doi.org/10.5281/zenodo.22780350) |
| this version, `v1.0.1` | [10.5281/zenodo.22780351](https://doi.org/10.5281/zenodo.22780351) |

ORCID: [0009-0008-9894-2982](https://orcid.org/0009-0008-9894-2982)

## License

Code in `src/` (excluding `src/third_party/`) and `figures/`: MIT. Data in `data/`, raw logs in `logs/`, and the generated figure images: CC-BY-4.0. `src/third_party/relmodpy/` is MIT by Fabian Gittins and carries its own licence file. See [`LICENSE`](LICENSE).

## How this work was produced

The calculations in this repository were carried out with automated agents under the author's direction, with an independent cross-audit performed between two separate research projects. The author is responsible for the results.
