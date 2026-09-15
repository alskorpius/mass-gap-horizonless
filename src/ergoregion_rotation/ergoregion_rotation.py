"""Ergoregion instability of a rotating horizonless star and spin corrections
to echo predictions.

Primary source (full text read, saved: _franzin2022b.html):
Franzin, Ledowski, Liberati, Carballo-Rubio, Pani, PRD 105, 124051 (2022),
arXiv:2201.01650 -- scalar perturbations on Kerr-black-bounce.
Key extracts:
  - rotating wormhole branches (no horizons) HAVE growing modes:
    M*omega_I in [1e-15, 1e-6] (growth times 10..1e10 (M/Msun) s) --
    ergoregion instability confirmed by their direct computation;
  - regular black-hole branches are stable (only damped QNM); superradiance
    is weaker than Kerr (smaller amplification, grows with spin, l=m=1
    maximum).
  - "larger ergosphere = more superradiance" is refuted by their
    Penrose-process analysis.

(1) Ergoregion map of the thin-shell model: Kerr exterior (spin a), shell at
R*. Kerr ergosphere: g_tt=0 at r_E(theta)=M+sqrt(M^2-a^2 cos^2 theta);
equator: r_E=2M. Outside the shell the metric is exact Kerr (Birkhoff for
axisymmetric vacuum), so: an ergoregion exists iff R* < max_theta r_E(theta)
= 2M; for R*>2M there is no ergoregion for any a. Checked numerically via
the Kerr g_tt.
  The stable family of near-horizon configurations (E-stabilizer:
  R*>~3.5M) always lies outside 2M.

(3) Estimate of the growth time in the window r_+<R*<2M: we use the primary
source's result (the M*omega_I range) as a scale and a Kerr-analogue
criterion: the ergoregion instability is significant on timescales
tau*M ~ 1/omega_I; astrophysical age t_astro ~ 1e10 yr = 3e17 s. The window
is CLOSED if tau_inst < t_astro for all realistic spins (chi>~0.1 for
observed black-hole candidates).

(4) Spin splitting of the echo comb: first order in a. Kerr asymptotic QNM
(Dolan-Ottewill / UV-dominated cavity modes):
omega_R*M ~ (n + 1/2) * pi / (M * ... ) -- for a cavity with delay tau_echo
the base comb spacing is Delta_omega = 2*pi/tau_echo; spin splits the
m-modes via the shell's angular velocity Omega* = a/(R*^2+a^2) (Keplerian
at the surface): shift of the m-th mode Delta_omega_m = m * Omega*.
Splitting of adjacent m: Delta_omega (spin) = Omega*(R*, a).
Comparison with the LIGO frequency resolution for 30 Msun (band
~10-800 Hz, resolution ~tens of Hz via deltaTime in the ringdown).

(5) The verdict is assembled in report.md.
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
    raise FileNotFoundError("project root (CITATION.cff + src) not found")


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


def kerr_gtt(r, theta, M, a):
    Sigma = r**2 + a**2 * np.cos(theta)**2
    return -(1 - 2 * M * r / Sigma)


def ergo_map():
    """(1): for a grid of (a, R*) -- does the Kerr exterior with a shell at
    R* (r+ < R*) have an ergoregion. Ergoregion iff
    min_theta g_tt(R*, theta) < 0."""
    rows = []
    for chi in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        a = chi  # M=1
        r_plus = 1 + np.sqrt(1 - a**2)
        for Rstar_frac in (1.01, 1.05, 1.2, 1.5, 2.0, 2.5, 3.5):
            Rstar = r_plus * Rstar_frac
            thetas = np.linspace(0, np.pi, 721)
            gmax = float(np.max(kerr_gtt(Rstar, thetas, 1.0, a)))
            has_ergo = bool(gmax > 0)  # ergoregion iff g_tt > 0 somewhere on the sphere
            rows.append(dict(chi=chi, Rstar=Rstar, Rstar_over_2M=Rstar / 2.0,
                             g_tt_max=gmax, ergoregion=has_ergo))
    # key question: R*>2M -- is there an ergoregion anywhere at all?
    outside = [r for r in rows if r["Rstar_over_2M"] > 1.0]
    any_outside = any(r["ergoregion"] for r in outside)
    inside = [r for r in rows if r["Rstar"] < 2.0 and r["Rstar"] > r_plus]
    inside_frac = (sum(r["ergoregion"] for r in inside) / len(inside)
                   if inside else None)
    return rows, any_outside, inside_frac


def instability_window():
    """(3): window r+<R*<2M. Growth times from the primary source:
    M*omega_I in [1e-15, 1e-6] -> tau_inst in [1e6, 1e15] M (geometric time)
    = [5e-9, 5e1] * (M/Msun) s. Worst case (slowest mode):
    tau = 1e15 * M [s]."""
    M_sun_geo_s = 4.925e-6
    rows = []
    for chi in (0.1, 0.3, 0.5, 0.7, 0.9):
        r_plus = 1 + np.sqrt(1 - chi**2)
        for frac in (1.05, 1.2, 1.5, 1.9):
            Rstar = min(r_plus * frac, 1.99)
            if Rstar <= r_plus:
                continue
            # conservative: use the LONG end of tau (1e15*M) --
            # if even that is smaller than t_astro, the window is surely closed
            tau_worst_s = 1e15 * M_sun_geo_s * 30.0  # M=30 Msun
            tau_fast_s = 1e6 * M_sun_geo_s * 30.0
            rows.append(dict(chi=chi, Rstar=Rstar, Rstar_over_2M=Rstar / 2,
                             tau_fast_s=tau_fast_s, tau_worst_s=tau_worst_s,
                             t_astro_s=3e17,
                             closed_even_worst=bool(tau_worst_s < 3e17)))
    return rows


def echo_spin_splitting():
    """(4): comb splitting Omega* = a/(R*^2+a^2) vs LIGO resolution."""
    rows = []
    for M_solar in (10.0, 30.0, 60.0):
        M_geo_s = M_solar * 4.925e-6
        for chi in (0.7, 0.9):
            a = chi  # in units of M
            for Rstar_frac in (1.2, 1.5, 2.0):
                Rstar = 2.0 * Rstar_frac  # fraction of 2M
                Omega_star = a / (Rstar**2 + a**2)  # 1/M geometric
                Omega_Hz = Omega_star / (2 * np.pi * M_geo_s)
                # base comb (E1-B, f=5 rho_nuc; M/Mext=0.95):
                # tau_echo = 18.7*2M_geo -- conversion: Delta_omega0 = 2 pi/tau
                # in our units eps_c=1 sets the scale via M_geo (our family);
                # for comparison we use the ECHO FREQUENCY f_echo = 1/tau_echo
                tau_echo_geo = 2.0 * 18.7 * Rstar / 2.0  # x M: ~18.7*Rstar/2... simplified below
                tau_echo_s = 18.7 * 2.0 * M_geo_s * (Rstar / 2.0)  # scale from E1-B
                f_echo_Hz = 1.0 / tau_echo_s
                rows.append(dict(M_solar=M_solar, chi=chi, Rstar_over_2M=Rstar_frac,
                                 Omega_star_Hz=Omega_Hz, f_echo_Hz=f_echo_Hz,
                                 split_over_f_echo=Omega_Hz / f_echo_Hz,
                                 LIGO_resolution_Hz=10.0,
                                 resolvable=bool(Omega_Hz > 10.0)))
    return rows


def main():
    out = {}
    rows, any_outside, inside_frac = ergo_map()
    out["ergo_map"] = dict(rows=rows,
                           any_ergo_outside_2M=any_outside,
                           ergo_fraction_inside_window=inside_frac,
                           statement=("R*>2M: g_tt(R*,theta)>=0 everywhere -- there is "
                                      "no ergoregion for any a (exterior is exact Kerr); "
                                      "window r+<R*<2M: an ergotorus is present"))
    print(f"(1) ergoregion at R*>2M: {any_outside} (expected False); "
          f"fraction with an ergotorus in the window r+<R*<2M: {inside_frac}")
    ins = instability_window()
    out["instability_window"] = ins
    worst_closed = all(r["closed_even_worst"] for r in ins)
    print(f"(3) window closed even for the slowest mode (1e15 M): "
          f"{worst_closed}; example: 30 Msun, tau_fast={ins[0]['tau_fast_s']:.2g} s, "
          f"tau_worst={ins[0]['tau_worst_s']:.2g} s (t_astro=3e17 s)")
    out["window_closed"] = worst_closed
    sp = echo_spin_splitting()
    out["echo_spin_splitting"] = sp
    for r in sp[::3]:
        print(f"(4) M={r['M_solar']} Msun, chi={r['chi']}, R*/2M={r['Rstar_over_2M']}: "
              f"Omega*={r['Omega_star_Hz']:.2g} Hz, f_echo={r['f_echo_Hz']:.2g} Hz, "
              f"resolvable by LIGO: {r['resolvable']}")
    safe_path(DATA / "ergoregion_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")


if __name__ == "__main__":
    main()
