"""E3 (corrected version): Visser-Wiltshire analysis of the radial stability
of stalling shells (the stage-10 family: core A_I inside, Schwarzschild M
outside).

Key refinement found by a direct dynamical test (noted):
the point V_eff(R0)=0 is by itself NOT an equilibrium of the EOS shell: the
stage-10 stall was sustained by flux (m'>0); at m'=0 we need STATIONARITY of
the effective potential along the adiabatic branch mu(R) = mu0 (R/R0)^{-2 kappa}:

  equilibrium:  V_eff(R*) = 0  and  v'(R*) = 0,  v(R) = V_eff(R, mu(R; R*)).
  stability: v''(R*) > 0 (a minimum).

Method: (1) numerically find the roots g(R0) = v'(R0) over R0 for given
(M, kappa); (2) evaluate v'' at the roots; (3) verdict; (4) direct check via
stage-10 dynamics (small perturbation, root switching at turning points,
return/escape criterion). Validation anchors: a dust shell in
Minkowski/Schwarzschild (known: the equilibrium is unstable or absent), and
agreement of the analytic v'' with the dynamics.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
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


ALPHA = 1.0
M_CRIT = 3.0 * np.sqrt(3.0) / 4.0
M_I = 0.65 * M_CRIT


def A_int(r, a=ALPHA, m_i=M_I):
    return 1.0 - 2.0 * m_i * r**2 / (r**3 + 2.0 * m_i * a)


def eta_of(R, M, mu, A_in):
    F = 1.0 - 2.0 * M / R
    return (R**2 * (A_in(R) - F) - mu**2) / (2.0 * mu * R)


def V_eff(R, M, mu, A_in):
    F = 1.0 - 2.0 * M / R
    return F - eta_of(R, M, mu, A_in) ** 2


def mu_static(R, M, A_in):
    return R * (np.sqrt(A_in(R)) - np.sqrt(max(1.0 - 2.0 * M / R, 0.0)))


def v_curve(R0, M, kappa, A_in, span=0.3, n=4001):
    """v(R) along the adiabatic branch, anchored at mu0=mu_static(R0)."""
    mu0 = mu_static(R0, M, A_in)
    Rs = R0 + np.linspace(-span, span, n)
    mus = mu0 * (Rs / R0) ** (-2.0 * kappa)
    return Rs, V_eff(Rs, M, mus, A_in)


def equilibrium_points(M, kappa, A_in, R_lo, R_hi):
    """Roots of g(R0) = dv/dR|R0 (stationarity of the pseudopotential)."""
    grid = np.linspace(R_lo, R_hi, 400)
    gvals = []
    for R0 in grid:
        if mu_static(R0, M, A_in) <= 0:
            gvals.append(np.nan)
            continue
        h = 1e-5 * R0
        mu0 = mu_static(R0, M, A_in)
        mu_p = mu0 * ((R0 + h) / R0) ** (-2 * kappa)
        mu_m = mu0 * ((R0 - h) / R0) ** (-2 * kappa)
        vp = V_eff(R0 + h, M, mu_p, A_in)
        vm = V_eff(R0 - h, M, mu_m, A_in)
        gvals.append((vp - vm) / (2 * h))
    gvals = np.array(gvals)
    roots = []
    for j in range(len(grid) - 1):
        if np.isfinite(gvals[j]) and np.isfinite(gvals[j + 1]) \
                and gvals[j] * gvals[j + 1] < 0:
            def g(R0):
                h = 1e-5 * R0
                mu0 = mu_static(R0, M, A_in)
                mu_p = mu0 * ((R0 + h) / R0) ** (-2 * kappa)
                mu_m = mu0 * ((R0 - h) / R0) ** (-2 * kappa)
                return (V_eff(R0 + h, M, mu_p, A_in)
                        - V_eff(R0 - h, M, mu_m, A_in)) / (2 * h)
            roots.append(float(brentq(g, grid[j], grid[j + 1])))
    return roots


def vpp_at(R0, M, kappa, A_in):
    h = 1e-4 * R0
    mu0 = mu_static(R0, M, A_in)
    mus = [mu0 * ((R0 + s * h) / R0) ** (-2 * kappa) for s in (-1, 0, 1)]
    vs = [V_eff(R0 + s * h, M, m, A_in) for s, m in zip((-1, 0, 1), mus)]
    return float((vs[2] - 2 * vs[1] + vs[0]) / h**2)


def direct_dynamics(M, R0, kappa, A_in, push=0.01, v_max=400.0, dv=0.005):
    """Exact stage-10 dynamics from equilibrium with a small velocity; at
    turning points the sign of Rdot flips (oscillations). Returns: outcome."""
    Rdot = -push
    # consistent mu at (R0, Rdot)
    F = 1 - 2 * M / R0
    eta_out = np.sqrt(F + Rdot**2)
    mu = R0 * (np.sqrt(A_int(R0) + Rdot**2) - eta_out)
    R, v = R0, 0.0
    turns, Rmin, Rmax = 0, R0, R0
    while v < v_max:
        F = 1 - 2 * M / R
        eta_out = (R**2 * (A_int(R) - F) - mu**2) / (2 * mu * R)
        disc = eta_out**2 - F
        if disc <= 0:
            turns += 1
            Rdot = -Rdot  # bounce
            disc = max(disc, 0.0)
        Rdot = np.sign(Rdot) * np.sqrt(disc) if disc > 0 else Rdot
        udot = 1 / (eta_out - Rdot)
        P = kappa * mu / (4 * np.pi * R**2)
        dmudtau = -8 * np.pi * R * Rdot * P
        R += Rdot / udot * dv
        mu += dmudtau / udot * dv
        Rmin, Rmax = min(Rmin, R), max(Rmax, R)
        if R <= 2 * M + 1e-9:
            return dict(outcome="BH_submersion", v=float(v), turns=turns)
        if R > 20 * R0:
            return dict(outcome="escaped", v=float(v), turns=turns)
        if mu <= 0:
            return dict(outcome="mu_depleted", v=float(v), turns=turns)
        v += dv
    return dict(outcome="bounded_oscillation", v=float(v), turns=turns,
                Rmin=float(Rmin), Rmax=float(Rmax))


def analyze(M, kappa, A_in, R_lo, R_hi):
    roots = equilibrium_points(M, kappa, A_in, R_lo, R_hi)
    out = []
    for R0 in roots:
        vpp = vpp_at(R0, M, kappa, A_in)
        dyn = direct_dynamics(M, R0, kappa, A_in)
        out.append(dict(R0=R0, mu0=float(mu_static(R0, M, A_in)),
                        v_double_prime=vpp,
                        analytic_verdict="stable" if vpp > 0 else "unstable",
                        dynamics_verdict=dyn["outcome"], dynamics=dyn))
    return out


def main():
    out = {}
    # Anchor: Minkowski inside, dust.
    A_mink = lambda r: 1.0
    dust = analyze(1.0, 0.0, A_mink, 2.5, 30.0)
    out["anchor_minkowski_dust"] = dust
    print("anchor Mink/dust:", json.dumps(dust, ensure_ascii=False))
    # Family of stalls.
    results = {}
    for M in (M_I + 0.5, M_I + 1.0):
        for kappa in (0.0, 0.25, 0.5, 1.0, -0.25):
            res = analyze(M, kappa, A_int, 2 * M + 0.05, 4 * M)
            results[f"M={M:.3f},kappa={kappa}"] = res
            print(f"M={M:.3f} kappa={kappa}: equilibria={[r['R0'] for r in res]}, "
                  f"vpp={[round(r['v_double_prime'],4) for r in res]}, "
                  f"dyn={[r['dynamics_verdict'] for r in res]}")
    out["stall_family"] = results
    # Stationarity of stall-without-flux points (checking that V'=0 does
    # not hold at an arbitrary point): example.
    R0 = 2 * (M_I + 0.5) + 1.0
    mu0 = mu_static(R0, M_I + 0.5, A_int)
    h = 1e-5 * R0
    gp = (V_eff(R0 + h, M_I + 0.5, mu0 * ((R0 + h) / R0) ** 0.0, A_int)
          - V_eff(R0 - h, M_I + 0.5, mu0, A_int)) / (2 * h)
    out["stall_point_not_stationary_example"] = dict(
        R0=R0, M=M_I + 0.5, kappa=0, dv_dr=float(gp),
        note="V_eff'(R0) != 0: a stall point is not an equilibrium without flux")
    safe_path(DATA / "shell_stability_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
