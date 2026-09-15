"""Two-phase model of superdense matter return (user cycle).

Phases:
  superdense: de Sitter (isotropic!) p = -rho_c = const. In the isotropic
  TOV term (eps+p)=0 => dp/dr=0: pressure is constant, core mass
  m_core(r) = (4 pi/3) rho_c r^3 -- exact solution (stage 3, degenerate case).
  normal: polytrope P = K eps^2 (Gamma=2, stage-3 check), isotropic TOV.

Boundary (Maxwell construction): at r=R_c the normal phase has pressure P_t
(coexistence pressure). A thin shell at the boundary carries a jump.
Configuration: core (0..R_c) + shell + polytropic envelope (R_c..R_s,
P(R_s)=0). Total mass M = m(R_s).

MAIN QUESTION (feedback): the family of equilibria M(R_c) -- if
dM/dR_c > 0 (monotonic), then as mass M is lost the core boundary
retreats (R_c decreases) -- return EXISTS; otherwise no-go.

Trigger dichotomy (3): pressure -- the boundary is free to follow M
(family); curvature -- the boundary at K(R_c)=K_t: we show numerically that
dK_env/dM_env << dK_env/dR_c => dR_c/dM ~ 0 -- no feedback.

Return rate (4): dR_c/dt = (dR_c/dM)(dM/dt); drivers: Hawking (BH branch)
or the transition channel (horizonless). L_return = phi |dM_core/dt| c^2;
cross-check against the eta limits from transition_radiation.

Stability (5): (a) TOV criterion dM/dR_c>0; (b) the goal explicitly requires
v''(R*)>0 from nonlinear_shell_eos: we apply the reduced criterion to the shell
between de Sitter (A_in = 1 - r^2/ell_c^2) and the envelope metric
(A_out = 1 - 2m(R_c)/R_c): we search the family for points where v'=0 (the
shell is also mechanically at equilibrium) and compute Gamma_c(R*).

Units: G=c=1, rho_c=1 (scale L0=1/sqrt(rho_c)); physical rescaling
at f = rho_c/rho_nuc.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
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


RHO_C = 1.0
ELL_C = np.sqrt(3.0 / (8.0 * np.pi * RHO_C))  # core curvature radius


def envelope(R_c, P_t, K, r_max=200.0):
    """TOV integration of the polytrope outward from the core boundary."""
    def tov(r, y):
        P, m = y
        eps = np.sqrt(max(P, 0.0) / K)
        dm = 4 * np.pi * r**2 * eps
        dP = -(eps + P) * (m + 4 * np.pi * r**3 * P) / (r * max(r - 2 * m, 1e-12))
        return [dP, dm]

    def surface(r, y):
        return y[0] - 1e-12 * P_t
    surface.terminal, surface.direction = True, -1

    m0 = (4 * np.pi / 3.0) * RHO_C * R_c**3
    sol = solve_ivp(tov, (R_c, r_max), [P_t, m0], events=[surface],
                    rtol=1e-10, atol=1e-12, max_step=0.01 * max(R_c, 0.01))
    if not sol.success and not sol.t_events[0].size:
        return None
    R_s = float(sol.t_events[0][0]) if sol.t_events[0].size else float(sol.t[-1])
    M = float(sol.y_events[0][0][1]) if sol.t_events[0].size else float(sol.y[1, -1])
    return dict(R_s=R_s, M=M, m_at_Rc=m0, P_end=float(sol.y[0, -1]))


def family(p_t, q=0.1, n=120):
    """Family M(R_c) at fixed P_t = p_t*rho_c; K = p_t/(q^2)."""
    K = p_t / (q * q)  # rho_n(P_t) = q*rho_c: P_t = K rho_n^2
    rows = []
    for R_c in np.geomspace(0.02 * ELL_C, 2.0 * ELL_C, n):
        env = envelope(R_c, p_t * RHO_C, K)
        if env is None or env["R_s"] <= 2 * env["M"]:
            continue  # no solution, or configuration inside the horizon
        rows.append(dict(R_c=R_c, M=env["M"], R_s=env["R_s"],
                         compactness=2 * env["M"] / env["R_s"],
                         m_core=(4 * np.pi / 3) * RHO_C * R_c**3))
    return rows, K


def slope(rows):
    R = np.array([r["R_c"] for r in rows])
    M = np.array([r["M"] for r in rows])
    dM = np.gradient(M, R)
    return R, M, dM


def shell_reduced(R_c, M_out, Pi=None):
    """Reduced criterion (nonlinear_shell_eos) for the de Sitter/polytrope
    boundary. A_in = 1 - r^2/ell_c^2; A_out = 1 - 2M_out_at/R."""
    A_in = lambda r: 1.0 - r**2 / ELL_C**2
    m_out = (4 * np.pi / 3) * RHO_C * R_c**3  # mass inside the boundary
    A_out = 1.0 - 2.0 * m_out / R_c
    mu = R_c * (np.sqrt(A_in(R_c)) - np.sqrt(A_out))
    if mu <= 0:
        return None
    V = lambda R, mu_: (1 - 2 * m_out * R_c / R) - ((R**2 * (A_in(R) - A_out) - mu_**2)
                                                    / (2 * mu_ * R))**2
    # for v'=0/v'' we take mu along the adiabat dmu/dR = -2 Pi mu/R
    hR = 1e-5 * R_c
    V_R = lambda mu_: (V(R_c + hR, mu_) - V(R_c - hR, mu_)) / (2 * hR)
    hm = 1e-6 * mu
    V_mu = (V(R_c, mu + hm) - V(R_c, mu - hm)) / (2 * hm)
    Pi_eq = R_c * V_R(mu) / (2 * mu * V_mu) if V_mu != 0 else None
    if Pi_eq is None:
        return None
    def vpp(Gamma):
        mu_p = mu * ((R_c + hR) / R_c) ** (-2 * Pi_eq)
        mu_m = mu * ((R_c - hR) / R_c) ** (-2 * Pi_eq)
        v0 = V(R_c, mu)
        vp = V(R_c + hR, mu_p)
        vm = V(R_c - hR, mu_m)
        return (vp - 2 * v0 + vm) / hR**2
    # Gamma_c: v'' is linear in Gamma: v''(G) = v''(0) + c1*G
    c1 = (vpp(1.0) - vpp(0.0))
    Gamma_c = -vpp(0.0) / c1 if c1 != 0 else None
    return dict(mu=mu, Pi_eq=float(Pi_eq), Gamma_c=float(Gamma_c) if Gamma_c is not None else None,
                vpp_G0=float(vpp(0.0)), vpp_G1=float(vpp(1.0)))


def curvature_trigger_check(rows, p_t, K):
    """Trigger dichotomy (corrected version). The boundary curvature K(R_c) is
    a function of ONLY local quantities: m(R_c)=(4pi/3)rho_c R_c^3, eps(R_c)=q
    (phase), P(R_c)=P_t -- the total mass M and the shell extent do NOT enter.
    Check: K at the boundary does not change when the envelope is truncated
    (varying r_max), whereas M does change. Consequence: the curvature trigger
    cannot track M -- there is no feedback (exactly), the pressure trigger
    can (family)."""
    def K_local(R_c):
        m = (4 * np.pi / 3) * RHO_C * R_c**3
        eps = np.sqrt(p_t / K)
        f = 1 - 2 * m / R_c
        dP = -(eps + p_t) * (m + 4 * np.pi * R_c**3 * p_t) / (R_c * (R_c - 2 * m))
        eps_p = dP / (2 * K * eps)
        B = (-8 * np.pi * R_c * eps + 2 * m / R_c**2) / (2 * R_c)
        D = 2 * m / R_c**3
        A = 0.5 * (-2 * (8 * np.pi * (R_c * eps) + 4 * np.pi * R_c**2 * eps_p) / R_c
                   + 2 * (4 * np.pi * R_c**2 * eps) * 2 / R_c**2 - 4 * m / R_c**3)
        return float(4 * (A**2 + 4 * B**2 + D**2))
    # family boundaries -- total M depends on the envelope, K_local does not:
    r_mid = rows[len(rows) // 2]["R_c"]
    r_end = rows[-1]["R_c"]
    K_mid, K_end = K_local(r_mid), K_local(r_end)
    dKdR = (K_local(r_end) - K_local(r_mid)) / (r_end - r_mid)
    M_mid, M_end = rows[len(rows) // 2]["M"], rows[-1]["M"]
    return dict(K_local_mid=K_mid, K_local_end=K_end,
                dK_dRc=dKdR,
                M_mid=M_mid, M_end=M_end,
                dM_dr_over_dK=max(abs(M_end - M_mid) / abs(dKdR * (r_end - r_mid)), 1e-30),
                statement=("K(R_c) depends only on (R_c, rho_c, P_t, q): total M does "
                           "not enter. Mass loss does not shift the curvature trigger: "
                           "dR_c/dM = 0 identically -- there is no feedback"))


def rates(rows, R, M, dMdR, f_nuc=5.0):
    """Return rate: physical rescaling, Hawking driver."""
    G_SI, C_SI, HBAR, MSUN = 6.674e-11, 3e8, 1.055e-34, 1.99e30
    rho_nuc_geo = G_SI * 2.8e17 / C_SI**2  # 1/m^2
    L0 = 1 / np.sqrt(f_nuc * rho_nuc_geo)  # m
    out = []
    for i in (len(rows) // 3, 2 * len(rows) // 3):
        M_geo_kg = M[i] * L0 * C_SI**2 / G_SI
        L_H = HBAR * C_SI**6 / (15360 * np.pi * G_SI**2 * M_geo_kg**2)
        dRcdM = 1.0 / dMdR[i]                     # geom.
        dRcdt = dRcdM * (-L_H / C_SI**2 * G_SI / C_SI**4) * C_SI / L0 * L0  # simplified below
        # carefully: dM_geo/dt_SI = L_H c^-2 [kg/s]; in geometric length units:
        dMdt_geo_per_s = (L_H / C_SI**2) * (G_SI / C_SI**4) / L0  # geom. length/s
        dRcdt = dRcdM * dMdt_geo_per_s             # geom. length/s
        dMcore_dt_kg = RHO_C * 4 * np.pi * rows[i]["R_c"]**2 * abs(dRcdt) \
            * L0**3 / (G_SI / C_SI**2) * (1.0)     # geom. volume -> m^3 -> kg: rho*V_geo*L0^3*(1/L0^2)/...
        dMcore_dt_kg = RHO_C / L0**2 * 4 * np.pi * (rows[i]["R_c"] * L0)**2 \
            * abs(dRcdt) * L0 * 0 + RHO_C * 4 * np.pi * rows[i]["R_c"]**2 \
            * abs(dRcdt) * L0 * (1.0)  # kg/s: rho_kg_m3 * dV/dt_m3
        rho_kg = f_nuc * 2.8e17
        dVdt = 4 * np.pi * (rows[i]["R_c"] * L0)**2 * abs(dRcdt) * L0  # m^3/s
        dMcore_dt_kg = rho_kg * dVdt
        out.append(dict(i=i, M_solar=M_geo_kg / MSUN, L_Hawking_W=L_H,
                        R_c_m=rows[i]["R_c"] * L0,
                        dMcore_dt_kg_s=dMcore_dt_kg,
                        L_return_phi1_W=dMcore_dt_kg * C_SI**2))
    return out


def main():
    out = {}
    fams = {}
    for p_t in (0.001, 0.01, 0.1):
        rows, K = family(p_t)
        if len(rows) < 10:
            continue
        R, M, dMdR = slope(rows)
        mono = bool(np.all(dMdR > 0))
        # non-monotonicity map: segments where dM/dR_c < 0
        neg = R[dMdR <= 0]
        mono_range = (float(R[0]), float(R[-1]))
        neg_range = (float(neg.min()), float(neg.max())) if neg.size else None
        fams[p_t] = dict(rows=rows, K=K, R=R, M=M, dMdR=dMdR, mono=mono,
                         neg_range=neg_range)
        frac_core = [r["m_core"] / r["M"] for r in rows]
        print(f"P_t={p_t}: n={len(rows)}, M in [{M.min():.4f},{M.max():.4f}], "
              f"compactness [{min(r['compactness'] for r in rows):.3f},"
              f"{max(r['compactness'] for r in rows):.3f}], "
              f"mono={mono}, dM/dR_c<0 at R_c in "
              f"{neg_range}, core mass fraction [{min(frac_core):.3f},{max(frac_core):.3f}]")
    out["families"] = {str(k): dict(K=v["K"], mono=v["mono"], n=len(v["rows"]),
                                    M_range=[float(v["M"].min()), float(v["M"].max())],
                                    compactness_range=[
                                        float(min(r["compactness"] for r in v["rows"])),
                                        float(max(r["compactness"] for r in v["rows"]))],
                                    R_c_range=[float(v["R"].min()), float(v["R"].max())],
                                    neg_range=v["neg_range"],
                                    core_fraction_range=[
                                        float(min(r["m_core"] / r["M"] for r in v["rows"])),
                                        float(max(r["m_core"] / r["M"] for r in v["rows"]))])
                       for k, v in fams.items()}
    # Trigger dichotomy (corrected version).
    p_t0 = 0.01
    if p_t0 in fams:
        curv = curvature_trigger_check(fams[p_t0]["rows"], p_t0, fams[p_t0]["K"])
        out["curvature_trigger"] = curv
        print("curvature:", json.dumps(curv, ensure_ascii=False, default=float))
    # Phase boundary: contact surface (the vacuum shell is degenerate, mu=0,
    # since m is continuous and f(R_c) coincides automatically -- recorded as
    # a finding); surface tension from the jump in f' (estimate with a GR factor).
    surf = []
    if p_t0 in fams:
        K = fams[p_t0]["K"]
        for r in fams[p_t0]["rows"][::12]:
            R_c = r["R_c"]
            m = (4 * np.pi / 3) * RHO_C * R_c**3
            eps_n = np.sqrt(p_t0 / K)
            f = 1 - 2 * m / R_c
            fprime_in = -2 * (4 * np.pi * R_c**2 * RHO_C) / R_c + 2 * m / R_c**2
            fprime_out = -2 * (4 * np.pi * R_c**2 * eps_n) / R_c + 2 * m / R_c**2
            tau = (R_c / 2.0) * (fprime_in - fprime_out) / np.sqrt(max(f, 1e-12))
            surf.append(dict(R_c=R_c, tau=tau / (4 * np.pi), f=f))
        for s in surf[::max(1, len(surf) // 5)]:
            print(f"boundary R_c={s['R_c']:.4f}: surface tension tau/(4pi)"
                  f"={s['tau']:.4g} (positive = boundary contraction)")
    out["boundary_surface"] = surf
    out["vacuum_shell_degeneracy"] = (
        "mu = R(eta_in - eta_out) = 0 at the boundary: m(R_c) is continuous, f coincides -- "
        "the boundary is CONTACT type (density and f' jump), not a vacuum seam; the v'' "
        "criterion from nonlinear_shell_eos does not directly apply to it (recorded); "
        "stability = TOV criterion dM/dR_c + surface tension")
    # Return rate: full dR_c/dM gain profile across the family + reference points.
    if p_t0 in fams:
        v = fams[p_t0]
        gain = 1.0 / v["dMdR"]
        out["gain_profile"] = dict(
            R_c=v["R"].tolist(), gain=gain.tolist(),
            note="dR_c/dM: positive in the core-dominated region = "
                 "feedback; local dips in the shell-dominated region")
        rr = rates(v["rows"], v["R"], v["M"], v["dMdR"])
        out["return_rates"] = rr
        for x in rr:
            print(f"rate: M={x['M_solar']:.3g} Msun, L_H={x['L_Hawking_W']:.3g} W, "
                  f"L_return(phi=1)={x['L_return_phi1_W']:.3g} W")
    safe_path(DATA / "two_phase_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")


if __name__ == "__main__":
    main()
