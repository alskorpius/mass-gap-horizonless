"""Laboratory and astrophysical matching of the ultradense phase (target 12.09.2026).

Part 1 -- hybrid stars: SLy envelope (piecewise-polytropic parametrization
Read et al. 2009, Table II: K of the low-density segments needs a c^2
correction -- pyTOVpp note; low/high part join at rho_0 ~ 1.5e14 g/cm^3;
Gamma1=3.005 from rho_0 to rho_1, Gamma2 up to rho_2, Gamma3 above) + de Sitter
core (p = -eps_c, m(r) = 4 pi eps_c r^3/3) at threshold P_t = P_SLy(rho_t).
Families M(R), twin branches, R_1.4, M_max, Lambda_1.4 (Love numbers with the
Postnikov+2010 jump corrected by Takatsy-Kovacs 2020:
y(r_d+) = y(r_d-) - Delta_eps/(eps_bar/3 + p_d), eps_bar = m/(4 pi r^3/3)).
Constraints: NICER R=12-14 km; M_max >= 2.0 M_sun; GW170817 Lambda_1.4=70-580.

Part 2 -- threshold on the QCD phase diagram (P_t, eps_t, n_t, mu_B, Delta eps).

Units: G=c=1; lengths in km; masses in M_sun (1.476625 km); pressures/energy
densities in km^-2 (converted from CGS via G/c^4, G/c^2 with cm^-2 -> km^-2 = 1e-10).
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
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


KM_PER_MSUN = 1.476625
C_CGS = 2.99792458e10
G_CGS = 6.6743e-8
C2 = C_CGS**2
C_P = G_CGS / C_CGS**4 * 1e10         # km^-2 per (dyn/cm^2); cm^-2 -> km^-2 = x1e10
C_RHO = G_CGS / C2 * 1e10             # km^-2 per (g/cm^3)*c^2
RHO_SAT = 2.66e14                     # g/cm^3, n_sat = 0.16 fm^-3
M_N_G = 1.66e-24                      # g, nucleon mass

# ---------- SLy (Read et al. 2009, Table II; K of low segments x c^2) ----------
_G1, _G2, _G3 = 3.005, 2.988, 2.851
_RHO1, _RHO2 = 10.0**14.7, 10.0**15.0
_P1 = 10.0**34.384                    # dyn/cm^2
K1 = _P1 / _RHO1**_G1
K2 = K1 * _RHO1**(_G1 - _G2)
K3 = K2 * _RHO2**(_G2 - _G3)
# low-density segments (rho_bot, K, gamma) -- table K x c^2
LOW = [
    (2.62789e12, 3.99874e-8 * C2, 1.35692),
    (3.78358e11, 5.32697e1 * C2, 0.62223),
    (2.44034e7, 1.06186e-6 * C2, 1.28733),
    (0.0, 6.80110e-9 * C2, 1.58425),
]
# segment L1's upper bound joins the K1 branch at rho_0:
RHO_JOIN = (LOW[0][1] / K1) ** (1.0 / (_G1 - LOW[0][2]))   # ~1.5e14 g/cm^3


def _segment(rho):
    """Returns the segment's (K, gamma) for the given mass density rho."""
    if rho >= _RHO2:
        return K3, _G3
    if rho >= _RHO1:
        return K2, _G2
    if rho >= RHO_JOIN:
        return K1, _G1
    for i, (bot, K, g) in enumerate(LOW):
        top = LOW[i - 1][0] if i > 0 else RHO_JOIN
        if bot <= rho < top:
            return K, g
    return LOW[-1][1], LOW[-1][2]


def sly_P_of_rho(rho):
    K, g = _segment(rho)
    return K * rho**g


def sly_rho_of_P(P):
    """Segment-wise inversion (P in dyn/cm^2)."""
    if not np.isfinite(P) or P <= 0.0:
        return 0.0
    if P >= K3 * _RHO2**_G3:
        return (P / K3) ** (1 / _G3)
    if P >= K2 * _RHO1**_G2:
        return (P / K2) ** (1 / _G2)
    if P >= K1 * RHO_JOIN**_G1:
        return (P / K1) ** (1 / _G1)
    for i, (bot, K, g) in enumerate(LOW):
        top = LOW[i - 1][0] if i > 0 else RHO_JOIN
        P_top = K * top**g
        if P < P_top:
            return (P / K) ** (1 / g)
    return (P / LOW[-1][1]) ** (1 / LOW[-1][2])


def sly_eps_of_rho(rho):
    """eps [km^-2] = (rho c^2 + P/(gamma-1)) [dyn/cm^2] -> xG/c^4."""
    K, g = _segment(rho)
    P = K * rho**g
    return (rho * C2 + P / (g - 1.0)) * C_P


def sly_cs2(P, eps):
    rho = sly_rho_of_P(P / C_P)
    K, g = _segment(rho)
    return g * P / (eps + P)


# ---------------- TOV and Love numbers ----------------

def _rhs(P, m, r):
    """Right-hand sides of TOV (without y)."""
    rho = sly_rho_of_P(P / C_P)
    e = sly_eps_of_rho(rho)
    dP = -(e + P) * (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
    dm = 4 * np.pi * r * r * e
    return dP, dm


def _rhs_y(r, s):
    P, m, y = s
    rho = sly_rho_of_P(P / C_P)
    e = sly_eps_of_rho(rho)
    cs2 = sly_cs2(P, e)
    lam = 1.0 / (1.0 - 2 * m / r)
    nup = (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
    Q = 4 * np.pi * lam * (5 * e + 9 * P + (e + P) / cs2) \
        - 6 * lam / r**2 - nup**2
    dy = (-y * y - r * r * Q - y * lam * (1 + 4 * np.pi * r * r * (P - e))) / r
    dP, dm = _rhs(P, m, r)
    return [dP, dm, dy]


def love_k2(beta, yR):
    num = 8.0 / 5.0 * beta**5 * (1 - 2 * beta) ** 2 * (2 - yR + 2 * beta * (yR - 1))
    den = 2 * beta * (6 - 3 * yR + 3 * beta * (5 * yR - 8)
                      + 2 * beta**2 * (13 - 11 * yR + beta * (3 * yR - 2)
                                       + 2 * beta**2 * (1 + yR))) \
        + 3 * (1 - 2 * beta) ** 2 * (2 - yR + 2 * beta * (yR - 1)) * np.log(1 - 2 * beta)
    return num / den


def surface_y_jump(P_s, m, R):
    """Surface jump (p_s -> 0, Postnikov limit of the corrected formula)."""
    rho_s = sly_rho_of_P(max(P_s / C_P, 1e-30))
    eps_s = sly_eps_of_rho(rho_s)
    eps_bar = m / (4 * np.pi * R**3 / 3)
    return -(0.0 - eps_s) / (eps_bar / 3 + P_s)


def tov_sly(Pc, y_need=False):
    """Pure SLy star; Pc in km^-2. (M[M_sun], R[km]) or (+y_R, Lambda)."""
    rho_c = sly_rho_of_P(Pc / C_P)
    m0 = 4 * np.pi / 3 * sly_eps_of_rho(rho_c) * 1e-18

    def surf(r, s):
        return s[0] - 1e-13 * Pc
    surf.terminal, surf.direction = True, -1

    def rhs2(r, s):
        dP, dm = _rhs(s[0], s[1], r)
        return [dP, dm]

    s0 = [Pc, m0, 2.0] if y_need else [Pc, m0]
    atol = [Pc * 1e-12, m0 * 1e-10 + 1e-18, 1e-9] if y_need \
        else [Pc * 1e-12, m0 * 1e-10 + 1e-18]
    sol = solve_ivp(_rhs_y if y_need else rhs2,
                    (1e-5, 3e3), s0, events=[surf],
                    rtol=1e-10, atol=atol, max_step=0.05)
    if not sol.t_events[0].size:
        return None
    R = float(sol.t_events[0][0])
    P, m = float(sol.y_events[0][0][0]), float(sol.y_events[0][0][1])
    M = m / KM_PER_MSUN
    if not y_need:
        return M, R
    yR = float(sol.y_events[0][0][2]) + surface_y_jump(P, m, R)
    k2 = love_k2(m / R, yR)
    Lam = 2.0 / 3.0 * k2 * (R / m) ** 5
    return M, R, yR, Lam


# ---------------- hybrid: de Sitter core + SLy envelope ----------------

def hybrid_star(eps_c, P_t, Rc, y_need=False, p_d_mode="envelope"):
    """Core (p=-eps_c) up to Rc, SLy from P_t. eps_c, P_t in km^-2, Rc in km."""
    m_c = 4 * np.pi / 3 * eps_c * Rc**3
    if Rc > 0 and 2 * m_c / Rc >= 0.999:
        return None

    def surf(r, s):
        return s[0] - 1e-13 * P_t
    surf.terminal, surf.direction = True, -1

    if not y_need:
        def rhs2(r, s):
            dP, dm = _rhs(s[0], s[1], r)
            return [dP, dm]
        sol = solve_ivp(rhs2, (Rc, 2e3),
                        [P_t, m_c], events=[surf], rtol=1e-10,
                        atol=[P_t * 1e-12, m_c * 1e-10 + 1e-18],
                        max_step=0.05)
        if not sol.t_events[0].size:
            return None
        R = float(sol.t_events[0][0])
        M = float(sol.y_events[0][0][1]) / KM_PER_MSUN
        return M, R

    # y through the core: (eps+p)=0 => Q_core = -16 pi eps_c lam - 6 lam/r^2 - nu'^2
    def core_dy(r, y):
        m = 4 * np.pi / 3 * eps_c * r**3
        P = -eps_c
        lam = 1.0 / (1 - 2 * m / r)
        nup = (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
        Q = -16 * np.pi * eps_c * lam - 6 * lam / r**2 - nup**2
        return (-y * y - r * r * Q - y * lam * (1 + 4 * np.pi * r * r * (P - eps_c))) / r

    r0 = 1e-5 * Rc
    solc = solve_ivp(core_dy, (r0, Rc), [2.0], rtol=1e-11, atol=1e-14,
                     max_step=Rc / 2000)
    y_in = float(solc.y[0, -1])
    rho_t = sly_rho_of_P(P_t / C_P)
    eps_out = sly_eps_of_rho(rho_t)
    p_d = P_t if p_d_mode == "envelope" else 0.0
    y_out = y_in - (eps_out - eps_c) / (eps_c / 3 + p_d)   # eps_bar(core) = eps_c

    sol = solve_ivp(_rhs_y, (Rc, 2e3), [P_t, m_c, y_out], events=[surf],
                    rtol=1e-10, atol=[P_t * 1e-12, m_c * 1e-10 + 1e-18, 1e-9],
                    max_step=0.05)
    if not sol.t_events[0].size:
        return None
    R = float(sol.t_events[0][0])
    P, m, yR_neg = sol.y_events[0][0]
    yR = float(yR_neg) + surface_y_jump(float(P), float(m), R)
    k2 = love_k2(m / R, yR)
    Lam = 2.0 / 3.0 * k2 * (R / m) ** 5
    return float(m) / KM_PER_MSUN, R, float(yR), float(Lam)


# ---------------- families and validation ----------------

def sly_family():
    fam = []
    for Pc in np.geomspace(8e-7, 1.2e-3, 90):
        r = tov_sly(Pc)
        if r:
            fam.append(dict(Pc=Pc, M=r[0], R=r[1]))
    return fam


def validate_sly():
    fam = sly_family()
    Ms = np.array([x["M"] for x in fam])
    Rs = np.array([x["R"] for x in fam])
    Mmax = float(Ms.max())
    R14 = float(np.interp(1.4, Ms, Rs))
    Pc14 = brentq(lambda p: tov_sly(p)[0] - 1.4, 8e-7, 1.2e-3, xtol=1e-14)
    res14 = tov_sly(Pc14, y_need=True)
    return dict(n=len(fam), Mmax=Mmax, Mmax_ref=2.05, R14=R14, R14_ref=11.7,
                Lam14=float(res14[3]), Lam14_ref=(300, 320),
                k2_newton_incompressible=float(love_k2(1e-3, -1.0)),
                join_rho=RHO_JOIN)


def hybrid_family(f_nuc, n_t_nsat):
    eps_sat = RHO_SAT * C2 * C_P
    eps_c = f_nuc * eps_sat
    rho_t = n_t_nsat * RHO_SAT
    P_t = sly_P_of_rho(rho_t) * C_P
    rows = []
    for Rc in np.geomspace(0.02, 14.0, 80):
        r = hybrid_star(eps_c, P_t, Rc)
        if r:
            M, R = r
            if 2.0 * M * KM_PER_MSUN >= R:   # surface inside the horizon
                continue
            m_core_km = 4 * np.pi / 3 * eps_c * Rc**3
            rows.append(dict(Rc=Rc, M=M, R=R,
                             M_core=m_core_km / KM_PER_MSUN))
    return dict(f=f_nuc, n_t=n_t_nsat, eps_c=eps_c, P_t=P_t,
                rho_t=rho_t, rows=rows)


def love_for_mass(f_nuc, n_t_nsat, M_target):
    """Lambda(M_target) on the rising tail of the hybrid branch: bisection over Rc."""
    eps_sat = RHO_SAT * C2 * C_P
    eps_c = f_nuc * eps_sat
    rho_t = n_t_nsat * RHO_SAT
    P_t = sly_P_of_rho(rho_t) * C_P

    rcs = np.geomspace(0.05, 14.0, 60)
    pts = []
    for Rc in rcs:
        r = hybrid_star(eps_c, P_t, Rc)
        if r and 2.0 * r[0] * KM_PER_MSUN < r[1]:
            pts.append((Rc, r[0], r[1]))
    if len(pts) < 5:
        return None
    P = np.array(pts)
    j0 = int(np.argmin(P[:, 1]))
    j1 = j0 + int(np.argmax(P[j0:, 1]))
    tail = P[j0:j1 + 1]
    xs = np.maximum.accumulate(tail[:, 1])
    ok = tail[:, 1] >= xs - 1e-9
    if M_target < tail[ok][:, 1].min() or M_target > tail[ok][:, 1].max():
        return None
    Rc_guess = float(np.interp(M_target, tail[ok][:, 1], tail[ok][:, 0]))
    for _ in range(60):
        r = hybrid_star(eps_c, P_t, Rc_guess, y_need=True)
        if r is None:
            return None
        if abs(r[0] - M_target) < 2e-4:
            return dict(Rc=Rc_guess, M=r[0], R=r[1], yR=r[2], Lambda=r[3])
        Rc_guess *= (1 + (M_target - r[0]) * 0.05)
    return None


def main():
    out = {}
    val = validate_sly()
    out["validation_sly"] = val
    print("SLy validation:", json.dumps(val, ensure_ascii=False, default=float))
    if not (1.9 <= val["Mmax"] <= 2.2 and 11.0 <= val["R14"] <= 12.4):
        print("!! SLy validation FAILED -- stopping for debugging")
        return

    # SLy family: curves M(Pc), R(M) and central density -> trigger mass
    fam = sly_family()
    fam.sort(key=lambda x: x["Pc"])
    Pc = np.array([x["Pc"] for x in fam])
    Ms = np.array([x["M"] for x in fam])
    Rs = np.array([x["R"] for x in fam])
    rho_c = np.array([sly_rho_of_P(p / C_P) for p in Pc])
    n_c = rho_c / RHO_SAT
    # monotonic branch up to the maximum
    i_max = int(np.argmax(Ms))
    out["sly"] = dict(Mmax=float(Ms[i_max]), R14=float(np.interp(1.4, Ms, Rs)),
                      nc_14=float(np.interp(1.4, Ms, n_c)),
                      nc_max=float(n_c[i_max]))

    NICER_14 = (12.0, 14.0)        # target statement
    NICER_14_2S = (11.9, 14.0)     # 2-sigma lower edge (J0030 13.02-1.06)
    NICER_208 = (11.1, 13.7)       # J0740+6620: 12.4 +/- 1.3
    LAM_14 = (70.0, 580.0)
    M_PSR = 2.08

    grid = []
    for f_nuc in (5.0, 10.0, 15.0, 20.0, 40.0):
        for n_t in (1.5, 2.0, 3.0, 3.5, 4.0, 5.0):
            hf = hybrid_family(f_nuc, n_t)
            if len(hf["rows"]) < 5:
                continue
            hM = np.array([r["M"] for r in hf["rows"]])
            hR = np.array([r["R"] for r in hf["rows"]])
            hRc = np.array([r["Rc"] for r in hf["rows"]])
            dMdRc = np.gradient(hM, hRc)
            Mmax_h = float(hM.max())
            if n_t <= n_c[i_max]:
                M_trig = float(np.interp(n_t, n_c[:i_max + 1], Ms[:i_max + 1]))
            else:
                M_trig = None

            def R_at(Mt):
                """R(Mt) on the rising tail of the hybrid branch (from the valley
                to the maximum), or None."""
                j0 = int(np.argmin(hM))              # valley
                j1 = j0 + int(np.argmax(hM[j0:]))     # maximum after the valley
                xs, ys_ = [], []
                for k in range(j0, j1 + 1):
                    if not xs or hM[k] > xs[-1]:
                        xs.append(hM[k])
                        ys_.append(hR[k])
                if not xs or Mt < xs[0] or Mt > xs[-1]:
                    return None
                return float(np.interp(Mt, xs, ys_))

            hyb14 = M_trig is not None and M_trig < 1.4
            hyb208 = M_trig is not None and M_trig < M_PSR
            R14_h = R_at(1.4) if hyb14 else None
            R208_h = R_at(M_PSR) if hyb208 else None
            R14_obs = R14_h if hyb14 else float(np.interp(1.4, Ms, Rs))
            if hyb208:
                R208_obs = R208_h
            else:
                R208_obs = float(np.interp(M_PSR, Ms, Rs)) \
                    if M_PSR <= Ms[i_max] else None
            Mmax_obs = max(Mmax_h, float(Ms[i_max]))
            ok14s = R14_obs is not None and NICER_14[0] <= R14_obs <= NICER_14[1]
            ok14w = R14_obs is not None and NICER_14_2S[0] <= R14_obs <= NICER_14_2S[1]
            # 2.08 cannot exist on the pure SLy branch if the trigger is lower
            ok_psr = Mmax_obs >= M_PSR and not (hyb208 and (R208_h is None))
            ok_nicer208 = (R208_obs is None and not hyb208) or \
                (R208_obs is not None and NICER_208[0] <= R208_obs <= NICER_208[1])
            stable_tail = hM[dMdRc > 0]
            twin = bool(M_trig is not None and stable_tail.size
                        and stable_tail.max() > M_trig + 0.05)
            # twin signature: dR at fixed mass at 2.0 (if both branches exist)
            dR_twin = None
            if M_trig is not None and 1.4 < M_trig < 2.0:
                Rh = R_at(2.0)
                if Rh is not None and 2.0 <= Ms[i_max]:
                    dR_twin = float(Rh - np.interp(2.0, Ms, Rs))
            grid.append(dict(
                f=f_nuc, n_t=n_t, n=len(hf["rows"]),
                M_trig=M_trig, Mmax_hybrid=Mmax_h, Mmax_obs=Mmax_obs,
                R14_obs=R14_obs, R14_on_hybrid=bool(hyb14),
                R208_obs=R208_obs, twin_branch=twin, dR_twin=dR_twin,
                n_unstable=int(np.sum(dMdRc < 0)),
                ok_nicer14_strict=bool(ok14s), ok_nicer14_2sig=bool(ok14w),
                ok_2msun=bool(ok_psr), ok_nicer208=bool(ok_nicer208),
                viable_strict=bool(ok14s and ok_psr and ok_nicer208),
                viable_2sig=bool(ok14w and ok_psr and ok_nicer208)))
    out["hybrid_grid"] = grid
    out["constraints"] = dict(
        NICER_14=NICER_14, NICER_14_2sigma=NICER_14_2S, NICER_208=NICER_208,
        Lambda14=LAM_14, M_pulsar=M_PSR,
        pure_SLy_R14=float(np.interp(1.4, Ms, Rs)),
        note="pure SLy (our fit) R14=11.55 vs. lit. 11.7: the EOS itself sits "
             "near the lower edge of NICER -- the relative comparison of branches is decisive")
    for g in grid:
        print(f"f={g['f']:>4}, n_t={g['n_t']}: M_trig="
              f"{g['M_trig'] if g['M_trig'] is None else round(g['M_trig'],2)}, "
              f"Mmax={g['Mmax_obs']:.2f}, R14="
              f"{g['R14_obs'] if g['R14_obs'] is None else round(g['R14_obs'],2)}"
              f"{'(hybrid)' if g['R14_on_hybrid'] else '(SLy)'}, R208="
              f"{g['R208_obs'] if g['R208_obs'] is None else round(g['R208_obs'],2)}, "
              f"twin={g['twin_branch']}, dR_twin={g['dR_twin']}, "
              f"viable={g['viable_strict']}/{g['viable_2sig']}")

    # step 2: Love numbers of representative hybrid configurations
    loves = []
    for (f_nuc, n_t, Mt, tag) in [
            (5.0, 3.0, 1.4, "hybrid at 1.4 (R exclusion)"),
            (10.0, 3.0, 1.4, "hybrid at 1.4 (R exclusion)"),
            (5.0, 4.0, 2.08, "marginal corner f=5,n_t=4"),
            (5.0, 5.0, 2.08, "corner f=5,n_t=5")]:
        res = love_for_mass(f_nuc, n_t, Mt)
        if res:
            loves.append(dict(f=f_nuc, n_t=n_t, tag=tag, **res))
            print(f"Lambda: f={f_nuc}, n_t={n_t}, M={Mt}: Rc={res['Rc']:.3f}, "
                  f"R={res['R']:.2f}, y_R={res['yR']:.3f}, "
                  f"Lambda={res['Lambda']:.1f} [{tag}]")
    out["love_selected"] = loves

    safe_path(DATA / "lab_interface_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print("step 1 written:", len(grid), "configurations; viable:",
          sum(g["viable_strict"] for g in grid), "(strict) /",
          sum(g["viable_2sig"] for g in grid), "(2-sigma)")


if __name__ == "__main__":
    main()
