"""P12: outcome A (the GW170817 remnant survived) -- cross-check with
late-time observations.

We compute condition X for a "quiet" surviving remnant on the corner
branch: spin-down dipole luminosity L_sd(B, P) against observational
limits.

Observational anchors (from abstracts, added to SOURCES):
- The afterglow falls off ~t^-2 after the peak (~day 160) out to
  1000+ days; late-time X-ray ~10^39-10^40 erg/s around day ~1000
  (Hajela 2019; Troja 2019; Makhathini 2020; Balasubramanian 2021/2022)
  -- no sign of ongoing injection.
- Ai et al. 2018 (ApJ 860, 57): allowed parameter space for a surviving
  NS: P ~ ms, B_p <= 10^10-10^12 G (depending on epsilon/xi); "no clear
  exclusion... but the parameter space is limited".
- Piro et al. 2019 (MNRAS 483, 1912): a weak-field B~10^12 G surviving
  remnant is consistent.

Our calculation: L_sd = B^2 R^6 Omega^4 / (6 c^3); E_rot = 0.5 I Omega^2,
I = k M R^2, k~0.35 (our corner: M=2.7-2.9, R=10.8 km). Condition X:
L_sd(B, P=1ms) < L_X-afterglow(t~1000d) ~ 10^39.5 erg/s OR the total
deposited energy eta*E_rot < E_kn~10^51 erg (the kilonova). Table of
B thresholds.
"""
import json
from pathlib import Path

import numpy as np
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "CITATION.cff").is_file() and (candidate / "src").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("project root not found")


PROJECT_ROOT = find_project_root(HERE)
DATA = PROJECT_ROOT / "data" / HERE.name
LOGS = PROJECT_ROOT / "logs" / HERE.name
FIGS = PROJECT_ROOT / "figures"
for _d in (DATA, LOGS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)


def safe_path(target) -> Path:
    resolved = Path(target).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path outside project root: {resolved}")
    return resolved


C = 2.998e10           # cm/s
R_CM = 10.8e5          # our remnant: R = 10.8 km
K_INERTIA = 0.35


def L_sd(B_g, P_ms):
    Omega = 2 * np.pi / (P_ms * 1e-3)
    return B_g**2 * R_CM**6 * Omega**4 / (6 * C**3)      # erg/s


def E_rot(M_solar, P_ms):
    I = K_INERTIA * (M_solar * 1.989e33) * R_CM**2        # g cm^2
    Omega = 2 * np.pi / (P_ms * 1e-3)
    return 0.5 * I * Omega**2                             # erg


def main():
    out = {"meta": "P12: condition X for outcome A (surviving remnant on "
                   "the corner branch) against late-time GW170817 "
                   "observations"}
    L_late = 3e39       # erg/s: late-time X-ray ~10^39-10^40 (day ~1000)
    E_km_cap = 1e51     # erg: cap on the contribution to the kilonova (energetics)
    rows = []
    for M in (2.7, 2.9):
        for P in (1.0, 2.0, 5.0):
            L = L_sd(1e12, P)
            E = E_rot(M, P)
            B_lim_lum = 1e12 * np.sqrt(L_late / L)       # B^2 scaling
            # if all of E_rot were deposited into the kilonova with efficiency eta:
            eta_max = E_km_cap / E
            rows.append(dict(M=M, P_ms=P,
                             L_sd_B12=L, L_late_bound=L_late,
                             B_limit_for_quiet=float(B_lim_lum),
                             E_rot=E, eta_max_km=float(eta_max)))
            print(f"M={M}, P={P} ms: L_sd(B=1e12)={L:.2e} erg/s; "
                  f"B_quiet={B_lim_lum:.1e} G; E_rot={E:.2e} erg "
                  f"(eta_max into kilonova {eta_max:.3f})")
    out["rows"] = rows
    out["condition_X"] = (
        "A surviving remnant on the corner branch (M~2.7-2.9, R=10.8 km) "
        "is consistent with the late-time GW170817 observations if it is "
        "\"quiet\": spin-down dipole luminosity L_sd < L_X-afterglow ~ "
        "3e39 erg/s (scale of 1000+ days) => at P=1 ms B_p <= 1.4e10 G; "
        "at P=2 ms B_p <= 5.6e10; at P=5 ms B_p <= 3.5e11 G. Alternative: "
        "early spin-down (B>=1e15 G, tau<days) with <= ~2%% of "
        "E_rot~4-5e52 erg deposited as radiation (otherwise it "
        "overpowers the kilonova, cap ~1e51 erg). Consistent with "
        "Ai+2018 (P~ms, B<=1e10-1e12 G depending on the coupling, 'no "
        "clear exclusion... limited') and Piro+2019 (B~1e12 G allowed in "
        "their setup); a millisecond magnetar with B~1e14-1e15 G is "
        "EXCLUDED (L_sd=1.5e45-1.5e47 >> 1e40 observed).")
    out["verdict"] = ("outcome A is NOT EXCLUDED under condition X: a "
                      "quiet remnant (B_p <= 1.4e10 G at P=1 ms, weaker "
                      "for shorter periods / up to ~3.5e11 G at 5 ms) or "
                      "early spin-down loss with <= 2% of E_rot deposited; "
                      "a millisecond magnetar with B >= 1e13.5 G is "
                      "excluded by late-time X-ray/radio")
    safe_path(DATA / "outcomeA_constraints.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(out["verdict"])


if __name__ == "__main__":
    main()
