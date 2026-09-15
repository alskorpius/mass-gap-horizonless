"""Independent check of the counterexample from the companion paper (the companion paper)
to the exchange hypothesis "monotone density <=> kappa_- = 0 is impossible".

Analytics (our class f=1-2m(r)/r, m'=4 pi r^2 rho):
  a triple root of f at R=R_- requires
    f=0  : m(R)=R/2
    f'=0 : m'(R)=1/2  <=>  rho(R)=1/(8 pi R^2)
    f''=0: m''(R)=0    <=>  sigma(R)=-dln rho/dln R|_R = 2
  (m''=8 pi R rho (1 - sigma/2); zero at sigma=2.)
  Transverse NEC: rho+p_perp = -r rho'/2 = rho*sigma/2 >= 0 for sigma>=0,
  i.e. a monotonically decreasing rho (sigma>=0) does NOT prevent
  sigma(R_-)=2. Hence the exchange hypothesis fails in its strong form; the
  correct, weakened form: the exchange holds for monotone sigma (single-scale
  profiles).

Construction for the independent check (our own profile, not the companion paper's code):
  sigma_shape(x) = [x^4/(1+x^4)] * [1 + (sigma_max-1)/(1+(x/x_b)^6)]
    (0 at the center -> rises to sigma_max -> settles to 1),
  rho(r) = rho_c * exp(-Int sigma dln r) * exp(-(x/x_c)^8)  -- truncation.
  sigma>=0 everywhere => rho decreases monotonically => the NEC is clean. A
  2D search over (R_-, x_b) solves {m(R)=R/2, m'(R)=1/2}; we then check
  f''=0 (via sigma=2), the sign change of f, the outer horizon R_+,
  K(0)=24/l^4, and the sensitivity to tuning (shifting x_b by 0.1% -> |kappa_-|).
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


ELL = 1.0        # core scale (unit of length)
X_C = 6.0        # truncation (power 16; contamination at x<4 is negligible)
X_D = 4.5        # damping of sigma after passing 2


def sigma_shape(x, sigma_max, x_b):
    bump = (x**4 / (1.0 + x**4)) * (1.0 + (sigma_max - 1.0) / (1.0 + (x / x_b)**6))
    return bump / (1.0 + (x / X_D)**4)


def build_profile(rho_c, sigma_max, x_b, n=400001):
    x = np.geomspace(1e-8, 200.0, n)
    r = x * ELL
    sig = sigma_shape(x, sigma_max, x_b) + 16.0 * (x / X_C)**16
    # Int sigma dlnx via cumulative trapezoid on the log grid.
    lnx = np.log(x)
    integrand = sig  # sigma dlnx
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                           * np.diff(lnx))])
    rho = rho_c * np.exp(-cum)
    m = 4.0 * np.pi * np.concatenate(
        [[0.0], np.cumsum(0.5 * (rho[1:] * r[1:]**2 + rho[:-1] * r[:-1]**2)
                          * np.diff(r))])
    return r, rho, sig, m


def interp_at(grid, values, point):
    return float(np.interp(point, grid, values))


def solve_triple(sigma_max=2.84, start=(0.07, 1.6, 2.5)):
    """3 equations for (rho_c, R, x_b):
       m(R) = R/2;  m'(R) = 1/2;  sigma(R) = 2.
    Then f=f'=f''=0 at R (triple root) and rho is monotone (sigma>=0)."""
    from scipy.optimize import fsolve

    def resid(u):
        rho_c, R, x_b = float(np.exp(u[0])), float(u[1]), float(np.exp(u[2]))
        r, rho, sig, m = build_profile(rho_c, sigma_max, x_b)
        mR = interp_at(r, m, R)
        rhoR = interp_at(r, rho, R)
        sigR = interp_at(r, sig, R)
        return [mR - 0.5 * R, 4.0 * np.pi * R**2 * rhoR - 0.5, sigR - 2.0]

    u0 = [np.log(start[0]), start[1], np.log(start[2])]
    u, info, ier, msg = fsolve(resid, u0, full_output=True, xtol=1e-13)
    if ier != 1:
        return None, msg
    return (float(np.exp(u[0])), float(u[1]), float(np.exp(u[2]))), None


def audit(sigma_max=2.84):
    sol, err = solve_triple(sigma_max)
    if sol is None:
        return dict(failed="fsolve did not converge: " + str(err))
    rho_c, R0, x_b = sol
    r, rho, sig, m = build_profile(rho_c, sigma_max, x_b)
    M_tot = float(m[-1])
    f = 1.0 - 2.0 * m / r

    def fp_of(R, rgrid=r, rhogrid=rho, mgrid=m):
        return -2.0 * (4 * np.pi * R**2 * interp_at(rgrid, rhogrid, R)) / R \
               + 2.0 * interp_at(rgrid, mgrid, R) / R**2

    rho_at = interp_at(r, rho, R0)
    sig_at = interp_at(r, sig, R0)
    m2 = 8.0 * np.pi * R0 * rho_at * (1.0 - sig_at / 2.0)
    fpp = -2.0 * m2 / R0 + 4.0 * 0.5 / R0**2 - 4.0 * (0.5 * R0) / R0**3
    zeros = []
    for j in range(len(r) - 1):
        if f[j] * f[j + 1] < 0:
            zeros.append(float(brentq(lambda q: float(
                1.0 - 2.0 * interp_at(r, m, q) / q), r[j], r[j + 1])))
    # Sensitivity: shift x_b by +0.1% -- kappa at the nearest zero of f.
    kap_pert = None
    r2, rho2, sig2, m2p = build_profile(rho_c, sigma_max, x_b * 1.001)
    f2 = 1.0 - 2.0 * m2p / r2
    for j in range(len(r2) - 1):
        if f2[j] * f2[j + 1] < 0:
            Rh = brentq(lambda q: float(1.0 - 2.0 * interp_at(r2, m2p, q) / q),
                        r2[j], r2[j + 1])
            kap_pert = float(fp_of(Rh, r2, rho2, m2p) / 2.0)
            break
    ell2 = 3.0 / (8.0 * np.pi * rho_c)
    return dict(
        sigma_max=sigma_max, rho_c=rho_c, x_b=x_b, R_minus=R0,
        M_total=M_tot, R_minus_over_M=R0 / M_tot,
        cond_8piR2rho=float(8.0 * np.pi * R0**2 * rho_at),
        sigma_at_R_minus=float(sig_at),
        kappa_minus=float(fp_of(R0) / 2.0),
        f_double_prime=float(fpp),
        f_positive_inside=bool(np.min(f[r < 0.98 * R0]) > 0),
        outer_horizons=zeros,
        M_inside_Rplus=float(interp_at(r, m, zeros[-1]) / M_tot) if zeros else None,
        rho_monotone_decreasing=bool(np.all(np.diff(rho) <= 1e-13 * rho[1:])),
        NEC_transverse_min=float(np.min(rho * sig / 2.0)),
        sigma_range=[float(np.min(sig)), float(np.max(sig))],
        K_center=float(512 * np.pi**2 * rho_c**2 / 3.0),
        K_center_via_ell=float(24.0 / ell2**2),
        kappa_perturbed_0_1pct=kap_pert,
        note="our own profile; independent check of the counterexample from the companion paper")


def main():
    out = audit(2.84)
    safe_path(DATA / "triple_root_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
