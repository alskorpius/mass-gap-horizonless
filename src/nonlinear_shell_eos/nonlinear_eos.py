"""Stabiliser of the horizonless branch: nonlinear EOS for a thin shell.

Continuation of E3 (stage_E3_shell_stability). Same setup: core A_I inside,
Schwarzschild M outside, master constraint mu = R(eta_in - eta_out),
motion Rdot^2 = -V_eff(R, mu).

Reduced criterion (new here):
  Along the adiabat dmu/dR = -8 pi R P(sigma), sigma = mu/(4 pi R^2).
  At equilibrium R*: mu* = mu_static(R*), V_eff = 0 and v'(R*) = 0, where
    Pi = P/sigma (at equilibrium), Gamma = dP/dsigma (at equilibrium).
  Then
    v'' = V_RR + 2 V_Rmu mu' + V_mumu mu'^2 + V_mu mu'',
    mu'  = -2 Pi mu/R,
    mu'' = (2 mu/R^2) [ -Pi + 2 Gamma (Pi + 1) ]   (consequence of the adiabat).
  The condition v' = 0 FIXES Pi from the geometry: Pi_eq(R*) = R V_R / (2 mu V_mu).
  v'' is linear in Gamma => for each R* there is a critical
    Gamma_c(R*) = -c0/c1; stable <=> Gamma lies on the correct side of Gamma_c.

Physically meaningful EOS classes (at the equilibrium point):
  - linear P = kappa sigma:              Pi = Gamma = kappa      (E3 check);
  - power law P = kappa sigma^{1+delta}: Pi = kappa sigma^delta,
    Gamma = (1+delta) Pi; "dP/dsigma grows under compression" <=> delta > 0;
  - general causal bound: 0 <= Gamma <= 1 (sound speed <= 1);
  - stiff-like Gamma = 1 for arbitrary Pi;
  - Chaplygin P = A sigma^2 - B: Gamma = 2P/(sigma) + B/sigma = 2Pi + B/sigma
    (B/sigma>0 free) => Gamma = 2Pi + x, x >= 0.

Numerically: scan R* over the family of stalls (two E3 masses), curves
Pi_eq(R*), Gamma_c(R*); check whether stability is reachable for the
classes above; direct dynamical check for any stable cases found.
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


ALPHA = 1.0
M_CRIT = 3.0 * np.sqrt(3.0) / 4.0
M_I = 0.65 * M_CRIT


def A_int(r, a=ALPHA, m_i=M_I):
    return 1.0 - 2.0 * m_i * r**2 / (r**3 + 2.0 * m_i * a)


def eta_of(R, M, mu):
    F = 1.0 - 2.0 * M / R
    return (R**2 * (A_int(R) - F) - mu**2) / (2.0 * mu * R)


def V_eff(R, M, mu):
    F = 1.0 - 2.0 * M / R
    return F - eta_of(R, M, mu) ** 2


def mu_static(R, M):
    return R * (np.sqrt(A_int(R)) - np.sqrt(max(1.0 - 2.0 * M / R, 0.0)))


def reduced_coefficients(R0, M):
    """Numerical partial derivatives of V at (R0, mu*) and the coefficients of v''(Pi,G)."""
    mu0 = mu_static(R0, M)
    hR = 1e-5 * R0
    hm = 1e-5 * max(mu0, 1e-12)
    V = lambda R, mu: V_eff(R, M, mu)
    V_R = (V(R0 + hR, mu0) - V(R0 - hR, mu0)) / (2 * hR)
    V_mu = (V(R0, mu0 + hm) - V(R0, mu0 - hm)) / (2 * hm)
    V_RR = (V(R0 + hR, mu0) - 2 * V(R0, mu0) + V(R0 - hR, mu0)) / hR**2
    V_mm = (V(R0, mu0 + hm) - 2 * V(R0, mu0) + V(R0, mu0 - hm)) / hm**2
    V_Rm = (V(R0 + hR, mu0 + hm) - V(R0 + hR, mu0 - hm)
            - V(R0 - hR, mu0 + hm) + V(R0 - hR, mu0 - hm)) / (4 * hR * hm)
    Pi_eq = R0 * V_R / (2.0 * mu0 * V_mu)
    c1 = 4.0 * mu0 / R0**2 * V_mu * (Pi_eq + 1.0)          # coefficient of Gamma
    c0 = (V_RR - 4.0 * Pi_eq * mu0 / R0 * V_Rm
          + 4.0 * Pi_eq**2 * mu0**2 / R0**2 * V_mm
          - 2.0 * mu0 / R0**2 * V_mu * Pi_eq)               # value at Gamma=0
    return dict(mu0=mu0, Pi_eq=Pi_eq, c0=c0, c1=c1,
                Gamma_c=(-c0 / c1) if c1 != 0 else None,
                sigma=mu0 / (4 * np.pi * R0**2))


def vpp_general(R0, M, Pi, Gamma):
    r = reduced_coefficients(R0, M)
    # v'' for arbitrary (Pi, Gamma) — for direct checks of EOS classes:
    mu0 = r["mu0"]
    # recompute coefficients for general Pi:
    mu_p = -2 * Pi * mu0 / R0
    mu_pp = 2 * mu0 / R0**2 * (-Pi + 2 * Gamma * (Pi + 1))
    hm = 1e-5 * max(mu0, 1e-12)
    hR = 1e-5 * R0
    V = lambda R, mu: V_eff(R, M, mu)
    V_RR = (V(R0 + hR, mu0) - 2 * V(R0, mu0) + V(R0 - hR, mu0)) / hR**2
    V_mm = (V(R0, mu0 + hm) - 2 * V(R0, mu0) + V(R0, mu0 - hm)) / hm**2
    V_Rm = (V(R0 + hR, mu0 + hm) - V(R0 + hR, mu0 - hm)
            - V(R0 - hR, mu0 + hm) + V(R0 - hR, mu0 - hm)) / (4 * hR * hm)
    return V_RR + 2 * V_Rm * mu_p + V_mm * mu_p**2 \
        + ((V(R0, mu0 + hm) - V(R0, mu0 - hm)) / (2 * hm)) * mu_pp


def validate_linear():
    """Reduction to E3: Pi=Gamma=kappa should reproduce the equilibria and v''."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "s3", PROJECT_ROOT / "src/stage_E3_shell_stability/shell_stability.py")
    s3 = importlib.util.module_from_spec(spec)
    sys.modules["s3"] = s3
    spec.loader.exec_module(s3)
    out = []
    for M, kappa in ((M_I + 0.5, 0.5), (M_I + 1.0, 0.5), (M_I + 0.5, 1.0)):
        roots = s3.equilibrium_points(M, kappa, s3.A_int, 2 * M + 0.05, 4 * M)
        for R0 in roots:
            r = reduced_coefficients(R0, M)
            vpp_E3 = s3.vpp_at(R0, M, kappa, s3.A_int)
            # at Pi=Gamma=kappa: v'' = c0 + c1*Gamma (Pi_eq=kappa at equilibrium)
            vpp_here = vpp_general(R0, M, kappa, kappa)
            out.append(dict(M=M, kappa=kappa, R0=R0, Pi_eq=r["Pi_eq"],
                            Gamma_c=r["Gamma_c"],
                            vpp_E3=vpp_E3, vpp_reduced=vpp_here,
                            rel_diff=abs(vpp_here - vpp_E3) / abs(vpp_E3)))
    return out


def scan_family(M, n=300):
    rows = []
    for R0 in np.linspace(2 * M + 0.05, 4 * M, n):
        r = reduced_coefficients(R0, M)
        rows.append(dict(R=R0, Pi_eq=r["Pi_eq"], Gamma_c=r["Gamma_c"],
                         sigma=r["sigma"]))
    return rows


def power_law_scan(M):
    """Power-law EOS P=kappa sigma^{1+delta}: Gamma=(1+delta) Pi_eq.
    Stability: v''>0 at Pi=Pi_eq(R*), Gamma=(1+delta)Pi_eq(R*)."""
    rows = []
    for R0 in np.linspace(2 * M + 0.05, 4 * M, 400):
        r = reduced_coefficients(R0, M)
        if r["Gamma_c"] is None:
            continue
        for delta in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
            G = (1 + delta) * r["Pi_eq"]
            vpp = vpp_general(R0, M, r["Pi_eq"], G)
            rows.append(dict(R=R0, delta=delta, Pi=r["Pi_eq"], Gamma=G,
                             Gamma_c=r["Gamma_c"], vpp=vpp,
                             stable=bool(vpp > 0), causal=bool(0 <= G <= 1)))
    return rows


def causal_scan(M):
    """Causal class 0<=Gamma<=1 (any EOS form): is there an R* that is
    stable for some Gamma in [0,1]?"""
    rows = []
    for R0 in np.linspace(2 * M + 0.05, 4 * M, 400):
        r = reduced_coefficients(R0, M)
        if r["Gamma_c"] is None:
            continue
        G_lo, G_hi = 0.0, 1.0
        v_lo = vpp_general(R0, M, r["Pi_eq"], G_lo)
        v_hi = vpp_general(R0, M, r["Pi_eq"], G_hi)
        # linearity in G: stability holds on [0,1] <=> max(v_lo,v_hi)>0
        rows.append(dict(R=R0, Pi=r["Pi_eq"], Gamma_c=r["Gamma_c"],
                         vpp_G0=v_lo, vpp_G1=v_hi,
                         stable_causal=bool(max(v_lo, v_hi) > 0)))
    return rows


def direct_check(R0, M, Pi, Gamma, push=0.01, v_max=300.0, dv=0.005):
    """Direct dynamics with a general EOS: dmu/dR = -8 pi R P(sigma),
    P(sigma) linearized about equilibrium: P = Pi sigma0 +
    Gamma (sigma - sigma0)... implemented as P(sigma) = Pi*sigma0 + Gamma*(sigma-sigma0)."""
    mu0 = mu_static(R0, M)
    sigma0 = mu0 / (4 * np.pi * R0**2)
    P = lambda sig: Pi * sigma0 + Gamma * (sig - sigma0)
    Rdot = -push
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
            Rdot = -Rdot
            disc = max(disc, 0.0)
        Rdot = np.sign(Rdot) * np.sqrt(disc) if disc > 0 else Rdot
        udot = 1 / (eta_out - Rdot)
        dmudtau = -8 * np.pi * R * Rdot * P(mu / (4 * np.pi * R**2))
        R += Rdot / udot * dv
        mu += dmudtau / udot * dv
        Rmin, Rmax = min(Rmin, R), max(Rmax, R)
        if R <= 2 * M + 1e-9:
            return "BH_submersion", turns
        if R > 20 * R0:
            return "escaped", turns
        if mu <= 0:
            return "mu_depleted", turns
        v += dv
    return "bounded_oscillation", turns


def main():
    out = {}
    out["validation_linear"] = validate_linear()
    for row in out["validation_linear"]:
        print("VALIDATE", row["M"], row["kappa"], "R0=%.4f Pi_eq=%.4f "
              "vpp_E3=%.4f vpp_reduced=%.4f rel=%.2e"
              % (row["R0"], row["Pi_eq"], row["vpp_E3"], row["vpp_reduced"],
                 row["rel_diff"]))
    masses = {"M_lo": M_I + 0.5, "M_hi": M_I + 1.0}
    for tag, M in masses.items():
        fam = scan_family(M, n=200)
        pl = power_law_scan(M)
        cs = causal_scan(M)
        out[tag] = dict(family=fam, power_law=pl, causal=cs)
        stable_pl = [r for r in pl if r["stable"]]
        stable_causal = [r for r in cs if r["stable_causal"]]
        print(f"{tag} (M={M:.3f}): power-law stable: {len(stable_pl)}/{len(pl)}; "
              f"causal-class stable R*: {len(stable_causal)}/{len(cs)}")
        if stable_causal:
            ex = stable_causal[0]
            print(f"   example: R*={ex['R']:.3f} Pi={ex['Pi']:.3f} "
                  f"Gamma_c={ex['Gamma_c']:.3f}")
    # direct checks: best candidate + counter-example
    probes = []
    M = M_I + 0.5
    for R0 in (2 * M + 0.05, 2 * M + 0.5, 2 * M + 1.0, 3 * M):
        r = reduced_coefficients(R0, M)
        if r["Gamma_c"] is None:
            continue
        for G in (0.0, 0.5, 1.0, 2.0):
            vpp = vpp_general(R0, M, r["Pi_eq"], G)
            verdict, turns = direct_check(R0, M, r["Pi_eq"], G)
            probes.append(dict(R=R0, Gamma=G, Pi=r["Pi_eq"], vpp=vpp,
                               dynamic=verdict, turns=turns))
            print(f"PROBE R*={R0:.3f} Pi={r['Pi_eq']:.3f} G={G}: "
                  f"vpp={vpp:+.4f} dyn={verdict} turns={turns}")
    out["direct_probes"] = probes
    safe_path(DATA / "nonlinear_eos_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")


if __name__ == "__main__":
    main()
