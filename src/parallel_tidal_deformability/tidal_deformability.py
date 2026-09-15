"""Parallel mini-stage P4: tidal deformability of the horizonless branch.

Question (user): what is Lambda for the horizonless objects on our branch --
is it consistent with the absence of a tidal signature for mass-gap objects
(GW190814 2.6 M_sun, GW230529)?

Method (primary source: Rincón et al., *Tidal Love numbers of anisotropic
stars within the complexity factor formalism*, [arXiv:2411.05171]
(https://arxiv.org/abs/2411.05171); their equations 39-43):
  y(r) = r H'/H:  r y' + y^2 + y e^λ [1 + 4 pi r^2 (p - rho)]
                  + r^2 { 4 pi e^λ [5 rho + 9 p + (rho+p)/c_s^2]
                          - 6 e^λ/r^2 - (nu')^2 } = 0,   y(0)=2.
  K0 = (1-2C)^2 [2C(y_R-1) - y_R + 2],  C = M/R,
  k2 = (8 C^5/5) K0 / [3 K0 ln(1-2C) + P5(C)],
  P5 = 2C[4C^4(y+1)+2C^3(3y-2)+2C^2(13-11y)+3C(5y-8)-3y+6],
  Lambda_tilde = 2 k2/(3 C^5).

Key simplification for our source (p_r = -rho, anisotropic,
stages 3-5): (rho+p) = 0 identically => the (rho+p)/c_s^2 term vanishes and
the equation closes WITHOUT any assumption about the sound speed (the
stage-5 trap does not apply). Anisotropy enters only through the background
(the metric) -- the approach is borrowed from the primary source with an
explicit caveat: their closure of source perturbations is a published
ansatz, not our microphysics.

Matching: our profile has an exponential tail; R_match = the radius where
m/M = 1-δ for δ in {1e-3, 1e-4, 1e-5}; convergence in δ is checked.

Validations: (1) the stage-3 polytropic star (isotropic, p=K rho^Gamma) --
k2 must fall in the typical NS range 0.04-0.12; (2) convergence in
R_match; (3) the (rho+p) terms are zeroed out and verified.
"""

import csv
import io
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import gamma as gamma_fn, gammainc, gammaincinv
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


EPS_C = 1.0
N = 6.0
M_EXT = 0.23136165195307468


# ---------------- background of the horizonless branch (n=6) ----------------
def rc_for_mass(M):
    return (M * N / (4.0 * np.pi * EPS_C * gamma_fn(3.0 / N))) ** (1.0 / 3.0)


def rho_core(r, rc):
    return EPS_C * np.exp(-((r / rc) ** N))


def m_core(r, rc, M):
    return M * gammainc(3.0 / N, (r / rc) ** N)


def f_core(r, rc, M):
    return 1.0 - 2.0 * m_core(r, rc, M) / r


def fp_core(r, rc, M):
    return -8.0 * np.pi * r * rho_core(r, rc) + 2.0 * m_core(r, rc, M) / r**2


def R_match_core(rc, delta):
    return float(rc * gammaincinv(3.0 / N, 1.0 - delta) ** (1.0 / N))


# ---------------- y-ODE (Rincón 2411.05171, Eq. 41-42) ----------------
def integrate_y(rho, p, f, fp, m, r_start, r_end):
    def cs2(r):
        return -1.0  # for our source (rho+p)=0 => the (rho+p)/cs2 term does not matter

    def rhs(r, yv):
        fr = float(f(np.array([r]))[0])
        fpr = float(fp(np.array([r]))[0])
        rho_r = float(rho(np.array([r]))[0])
        p_r = float(p(np.array([r]))[0])
        nu_p = fpr / (2.0 * fr)
        Q = (4.0 * np.pi * (1.0 / fr) * (5.0 * rho_r + 9.0 * p_r + (rho_r + p_r) / cs2(r))
             - 6.0 / (fr * r**2) - nu_p**2)
        y0 = float(yv[0])
        return [-(y0**2 + y0 * (1.0 / fr) * (1.0 + 4.0 * np.pi * r**2 * (p_r - rho_r))
                 + r**2 * Q) / r]
    r0 = max(r_start, 1e-9)
    for method in ("LSODA", "Radau", "RK45"):
        sol = solve_ivp(rhs, (r0, r_end), [2.0], rtol=1e-10, atol=1e-12,
                        method=method, dense_output=False)
        if sol.success:
            return float(sol.y[0, -1]), method
    return None, "all_failed"


def love_numbers(y_R, M, R):
    C = M / R  # convention of the primary source: C = M/R
    one2c = 1.0 - 2.0 * C
    K0 = one2c**2 * (2.0 * C * (y_R - 1.0) - y_R + 2.0)
    P5 = 2.0 * C * (4.0 * C**4 * (y_R + 1.0) + 2.0 * C**3 * (3.0 * y_R - 2.0)
                    + 2.0 * C**2 * (13.0 - 11.0 * y_R) + 3.0 * C * (5.0 * y_R - 8.0)
                    - 3.0 * y_R + 6.0)
    denom = 3.0 * K0 * np.log(one2c) + P5
    k2 = (8.0 * C**5 / 5.0) * K0 / denom
    lam_tilde = 2.0 * k2 / (3.0 * C**5)
    return k2, lam_tilde


def core_model(M_over_ext, deltas=(1e-3, 1e-4, 1e-5)):
    M = M_EXT * M_over_ext
    rc = rc_for_mass(M)
    rho = lambda r: rho_core(r, rc)
    p = lambda r: -rho_core(r, rc)  # p_r = -rho
    f = lambda r: f_core(r, rc, M)
    fp = lambda r: fp_core(r, rc, M)
    m = lambda r: m_core(r, rc, M)
    out = []
    for d in deltas:
        R = R_match_core(rc, d)
        y_R, method = integrate_y(rho, p, f, fp, m, 1e-6, R)
        if y_R is None:
            out.append(dict(delta=d, failed=True, note=method))
            continue
        k2, lam = love_numbers(y_R, M, R)
        out.append(dict(delta=d, failed=False, R=R, y_R=y_R, k2=k2,
                        Lambda=lam, C=M / R, compactness_2M_R=2 * M / R,
                        solver=method))
    return dict(M=M, M_over_ext=M_over_ext, results=out)


# ---------------- validation: stage-3 polytropic star ----------------
def polytrope_validation(eps0=0.15, K=1.0, Gam=2.0):
    """Isotropic polytropic star (TOV), y-ODE with c_s^2 = dp/drho."""
    r0 = 1e-6

    def eos_p(eps):
        return K * eps**Gam

    def rhs_tov(r, y):
        p, m = y
        eps = (p / K)**(1.0 / Gam)
        return [-((eps + p) * (m + 4.0 * np.pi * r**3 * p)) / (r * (r - 2.0 * m)),
                4.0 * np.pi * r**2 * eps]

    def surface(r, y):
        return y[0]
    surface.terminal, surface.direction = True, -1
    sol = solve_ivp(rhs_tov, (r0, 1e3), [eos_p(eps0), 4.0 / 3.0 * np.pi * r0**3 * eps0],
                    events=[surface], rtol=1e-10, atol=1e-14, max_step=1e-3)
    R, M = float(sol.t[-1]), float(sol.y[1, -1])

    def rho(r):
        return (np.maximum(eos_p_from_tov(sol, r), 0.0) / K)**(1.0 / Gam)

    def p_of(r):
        return np.maximum(eos_p_from_tov(sol, r), 0.0)

    def eos_p_from_tov(sol_v, r):
        # dense interpolation of p(r) over the TOV solution
        return np.interp(r, sol_v.t, sol_v.y[0])

    def m_of(r):
        return np.interp(r, sol.t, sol.y[1])

    def f_of(r):
        return 1.0 - 2.0 * m_of(r) / r

    def fp_of(r):
        eps = (np.maximum(np.interp(r, sol.t, sol.y[0]), 0.0) / K)**(1.0 / Gam)
        return -8.0 * np.pi * r * eps + 2.0 * m_of(r) / r**2

    def rhs_y(r, yv):
        p_r = float(p_of(np.array([r]))[0])
        rho_r = float(rho(np.array([r]))[0])
        fr = float(f_of(np.array([r]))[0])
        fpr = float(fp_of(np.array([r]))[0])
        nu_p = fpr / (2.0 * fr)
        cs2 = K * Gam * rho_r**(Gam - 1.0) if rho_r > 1e-14 else 1.0
        Q = (4.0 * np.pi / fr * (5.0 * rho_r + 9.0 * p_r
                                 + ((rho_r + p_r) / cs2 if cs2 != 0 else 0.0))
             - 6.0 / (fr * r**2) - nu_p**2)
        y0 = float(yv[0])
        return [-(y0**2 + y0 / fr * (1.0 + 4.0 * np.pi * r**2 * (p_r - rho_r))
                 + r**2 * Q) / r]
    soly = solve_ivp(rhs_y, (1e-7, R * (1 - 1e-10)), [2.0],
                     rtol=1e-10, atol=1e-12)
    y_R = float(soly.y[0, -1])
    k2, lam = love_numbers(y_R, M, R)
    return dict(eps0=eps0, R=R, M=M, y_R=y_R, k2=k2, Lambda=lam, C=M / R)


def main():
    out = dict()
    out["validation_polytrope"] = polytrope_validation()
    models = {}
    for ratio in (0.88, 0.95, 0.995, 0.9999):
        models[f"M_over_ext={ratio}"] = core_model(ratio)
    out["horizonless_models"] = models
    rows = []
    for tag, mod in models.items():
        for res in mod["results"]:
            rows.append(dict(model=tag, M=mod["M"], delta=res.get("delta"),
                             failed=res.get("failed", False), R=res.get("R"),
                             y_R=res.get("y_R"), k2=res.get("k2"),
                             Lambda=res.get("Lambda"),
                             compactness_2M_R=res.get("compactness_2M_R")))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()),
                            lineterminator="\r\n", extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    safe_path(DATA / "tidal_results.csv").write_text(
        buf.getvalue(), encoding="utf-8", newline="")
    safe_path(DATA / "tidal_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print("VALIDATION (polytrope):", out["validation_polytrope"])
    for tag, mod in models.items():
        for res in mod["results"]:
            if not res.get("failed"):
                print(f"{tag}: delta={res['delta']:g} R={res['R']:.4f} "
                      f"y_R={res['y_R']:.4f} k2={res['k2']:.5g} "
                      f"Lambda={res['Lambda']:.5g} 2M/R={res['compactness_2M_R']:.4f}")


if __name__ == "__main__":
    main()
