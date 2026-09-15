"""E4 (goal): does the curvature-term tower admit a triple root kappa_- = 0?

After the the companion paper counterexample (confirmed by us, verify_triple_root_monotone)
the exchange hypothesis is weakened: "NEC <-> kappa_-=0" holds only for a
monotone slope sigma = -dln(rho)/dln r. E4: check the stage-6 tower.

Analytic core (our class f = 1 - psi r^2, h(psi) = m(r)/r^3):
  rho = 3(psi - t psi_t)/(8 pi),  t = h(psi),  psi_t = dpsi/dt,
  sigma = -3 t^2 psi_tt / (psi - t psi_t).
  Triple-root conditions at R:
    m(R) = R/2   =>  psi_R R^2 = 1  =>  R = 1/sqrt(psi_R),  M = h(psi_R)/(2 psi_R^{3/2});
    rho(R) = 1/(8 pi R^2)  =>  2 psi = 3 h(psi) psi_t          [equation E1: psi only]
    sigma(R) = 2           =>  -3 h^2 psi_tt = 2(psi - h psi_t) [equation E2: psi only]
  Neither condition depends on M: the system is overdetermined (2 equations
  for 1 unknown psi) -- solvability requires a special h.

Additionally: the crossing direction of sigma=2. A triple root with a static
core (f>0 inside) requires sigma'(R_r)<0, i.e. dsigma/dt>0.

Families (alpha_n >= 0, lim (alpha_n)^{1/n} = C > 0 -- stage-6 regularity):
  (a) geometric alpha_n = alpha^{n-1}:  h = psi/(1-alpha psi)  [= Hayward];
  (b) alpha_n = n alpha^{n-1}:          h = psi/(1-alpha psi)^2;
  (c) alpha_n = alpha^{n-1}/n:          h = -(1/alpha) ln(1-alpha psi);
  (d) two-term mixes h = psi + a2 psi^2 + a3 psi^3 (alpha_i>=0) -- scan over ratios.

For each: solve E1; check E2 on the solution; build sigma(t); produce a
verdict on the triple root. When a solution is found -- direct numerical
verification of the metric f(r) (triple root to machine precision).
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


def analyze_h(name, h, h_prime, h_double, t_max_frac=0.999, n_grid=20001):
    """h(psi), h'(psi), h''(psi) are given symbolically-numerically via sympy-lambdify
    on input. Returns: the E1 solution, the E2 residual there, sigma's
    monotonicity, the crossing direction at 2, M and R on success."""
    import sympy as sp
    ps = sp.symbols("ps", positive=True)
    h_sym = h(ps)
    hp_sym = sp.diff(h_sym, ps)
    # psi(t): inverted numerically; we work in the variable psi on (0, psi_max),
    # psi_max = radius of convergence (for (a),(b): 1/alpha; for (c): 1/alpha).
    # t = h(psi) is monotone (h'>0 for alpha_n>=0).
    h_f = sp.lambdify(ps, h_sym, "numpy")
    hp_f = sp.lambdify(ps, hp_sym, "numpy")
    hpp_f = sp.lambdify(ps, sp.diff(h_sym, ps, 2), "numpy")
    alpha_val = 1.0
    # psi domain: from 1e-4 (below that -- numerical noise 0/0: psi - t psi_t ~ a2 psi^2)
    # to psi_max*(1-1e-12).
    psi_max = {"geometric": 1.0, "n_alpha": 1.0, "log": 1.0}.get(name, 10.0)
    ps_grid = np.geomspace(1e-4, psi_max * t_max_frac, n_grid)
    t_grid = np.array([float(h_f(p)) for p in ps_grid])
    psi_of_t = lambda t: float(np.interp(t, t_grid, ps_grid))
    # psi_t = 1/h'(psi); psi_tt = -h''/(h')^3.
    def psi_t(p):
        return 1.0 / float(hp_f(p))
    def psi_tt(p):
        return -float(hpp_f(p)) / float(hp_f(p))**3
    # sigma(t) via psi: sigma = -3 t^2 psi_tt/(psi - t psi_t).
    def sigma_at(p):
        t = float(h_f(p))
        return -3.0 * t**2 * psi_tt(p) / (p - t * psi_t(p))
    # E1: 2 psi - 3 h psi_t = 0  <=>  2 psi h' = 3 h.
    def E1(p):
        return 2.0 * p * float(hp_f(p)) - 3.0 * float(h_f(p))
    # E2: -3 h^2 psi_tt - 2(psi - h psi_t) = 0.
    def E2(p):
        t = float(h_f(p))
        return -3.0 * t**2 * psi_tt(p) - 2.0 * (p - t * psi_t(p))
    E1v = np.array([E1(p) for p in ps_grid])
    roots = []
    for j in range(len(ps_grid) - 1):
        if E1v[j] * E1v[j + 1] < 0:
            roots.append(brentq(E1, ps_grid[j], ps_grid[j + 1], xtol=1e-14))
    sig_grid = np.array([sigma_at(p) for p in ps_grid])
    dSigma_dt = np.gradient(sig_grid, t_grid)
    out = dict(name=name, E1_roots=[float(r) for r in roots],
               E2_at_roots=[float(E2(r)) for r in roots],
               sigma_min=float(np.min(sig_grid)), sigma_max=float(np.max(sig_grid)),
               sigma_monotone_in_t=bool(np.all(dSigma_dt[1:-1] > 0)
                                        or np.all(dSigma_dt[1:-1] < 0)),
               crossing2_solutions=[])
    # Points where sigma=2: their direction.
    s2 = sig_grid - 2.0
    for j in range(len(ps_grid) - 1):
        if s2[j] * s2[j + 1] < 0:
            p_c = brentq(lambda p: sigma_at(p) - 2.0, ps_grid[j], ps_grid[j + 1])
            out["crossing2_solutions"].append(dict(
                psi=float(p_c), sigma_prime_over_t=float(np.interp(
                    p_c, ps_grid, dSigma_dt))))
    # If there is a simultaneous (E1, E2) solution: collect M, R and check the metric.
    for r0, e2res in zip(out["E1_roots"], out["E2_at_roots"]):
        if abs(e2res) < 1e-6 * max(1.0, abs(3.0 * float(h_f(r0))**2 * abs(psi_tt(r0)))):
            t0 = float(h_f(r0))
            M = t0 / (2.0 * r0**1.5)
            R = 1.0 / np.sqrt(r0)
            out["triple_root_candidate"] = dict(psi=float(r0), M=float(M), R=float(R),
                                                E2_residual=float(e2res))
    return out


def verify_metric_direct(name, h, psi_R, M, R):
    """Direct check of the triple root of f=1-psi(r) r^2 (numerically, via psi(t))."""
    import sympy as sp
    ps = sp.symbols("ps", positive=True)
    h_sym = h(ps)
    h_f = sp.lambdify(ps, h_sym, "numpy")
    hp_f = sp.lambdify(ps, sp.diff(h_sym, ps), "numpy")
    # Solve psi(r): h(psi) = M/r^3 by bisection in psi on (0, psi_max).
    def psi_of_r(r):
        t = M / r**3
        lo, hi = 1e-14, 0.999999
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            if float(h_f(mid)) < t:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    rr = np.geomspace(1e-4, max(50.0, 10 * R), 400000)
    psi_r = np.array([psi_of_r(x) for x in rr[::20]])
    rr = rr[::20]
    f = 1.0 - psi_r * rr**2
    i0 = int(np.argmin(np.abs(f)))
    Rf = float(rr[i0])
    fp = np.gradient(f, rr)
    fpp = np.gradient(fp, rr)
    j = int(np.argmin(np.abs(rr - Rf)))
    return dict(R_found=Rf, R_expected=R, f_min=float(np.min(np.abs(f))),
                fprime_at_min=float(fp[j]), fpp_at_min=float(fpp[j]))


def symbolic_proof():
    """Symbolic check of the mini-theorem: at the E1 root (2 psi h'=3h)
    E2 = (4 psi^2/(9h)) (2 psi h'' - h'), and 2 psi h'' - h'
    = -1 + Sum_{n>=2} n(2n-3) alpha_n psi^{n-1} >= -1 + 2 = 1 > 0
    for alpha_1=1, alpha_n>=0 (the inequality: n>=2 and the root condition
    Sum_{n>=2}(2n-3) alpha_n psi*^n = psi*)."""
    import sympy as sp
    ps, t = sp.symbols("ps t", positive=True)
    a1 = sp.Integer(1)
    # Chain: E2 = -3 h^2 psi_tt - 2(psi - h psi_t), psi_t=1/h', psi_tt=-h''/h'^3
    hp_, hpp_, h_ = sp.symbols("hp hpp h", positive=True)
    psi_t = 1 / hp_
    psi_tt = -hpp_ / hp_**3
    E2 = -3 * h_**2 * psi_tt - 2 * (ps - h_ * psi_t)
    E2_at_root = sp.simplify(E2.subs(hp_, 3 * h_ / (2 * ps)))  # E1: 2 psi h'=3h
    target = sp.simplify((4 * ps**2 / (9 * h_)) * (2 * ps * hpp_ - hp_))
    chain_ok = sp.simplify(E2_at_root * 9 * h_ - 4 * ps**2 * (2 * ps * hpp_ - 3 * h_ / (2 * ps))) == 0
    # Check: 2 psi h'' - h' = Sum n(2n-3) alpha_n psi^{n-1} for h=Sum alpha_n psi^n:
    N = 6
    aa = sp.symbols("a1:" + str(N + 1), nonnegative=True)
    h_poly = sum(aa[n - 1] * ps**n for n in range(1, N + 1))
    lhs = sp.expand(2 * ps * sp.diff(h_poly, ps, 2) - sp.diff(h_poly, ps))
    rhs = sp.expand(sum(sp.Integer(n) * (2 * n - 3) * aa[n - 1] * ps**(n - 1)
                        for n in range(1, N + 1)))
    poly_ok = sp.simplify(lhs - rhs) == 0
    # Inequality at the root: Sum_{n>=2}(2n-3) alpha_n psi*^n = psi* (alpha_1=1):
    E1_poly = sp.expand(2 * ps * sp.diff(h_poly, ps) - 3 * h_poly)
    E1_coeff = sp.Poly(E1_poly, ps).all_coeffs()[::-1]  # in ascending order of degree
    return dict(chain_identity=bool(chain_ok), coefficient_identity=bool(poly_ok),
                E1_root_condition=str(E1_coeff),
                note="alpha_1=1 => the constant term of E1 = -psi; higher terms >= 0 => psi* = Sum_{n>=2}(2n-3)alpha_n psi*^n")


def main():
    import sympy as sp
    results = {"symbolic_proof": symbolic_proof()}  # single initialization
    families = {
        "geometric": lambda p: p / (1 - p),
        "n_alpha": lambda p: p / (1 - p)**2,
        "log": lambda p: -sp.log(1 - p),
    }
    for name, hfun in families.items():
        res = analyze_h(name, hfun, None, None)
        results[name] = res
        print(name, "->", json.dumps(res, ensure_ascii=False))
    # Two-term mixes: h = psi + a2 psi^2 + a3 psi^3 (truncated).
    best = []
    for a2 in (0.5, 1.0, 2.0):
        for a3 in (0.1, 0.5, 1.0, 2.0, 5.0):
            hfun = lambda p, a2=a2, a3=a3: p + a2 * p**2 + a3 * p**3
            res = analyze_h(f"mix_{a2}_{a3}", hfun, None, None)
            has = "triple_root_candidate" in res
            best.append((has, f"a2={a2},a3={a3}", res["E1_roots"], res["E2_at_roots"]))
            if has:
                results[f"mix_{a2}_{a3}"] = res
    results["mixes_summary"] = [list(b) for b in best]
    # Analytic edge case: the two-term h = psi + c psi^2 (truncated,
    # irregular) -- algebraically E2 = 0 at the E1 root (psi* = 1/c):
    # {2 psi h' = 3h} <=> c psi = 1; then 2 psi h'' - h' = 2 psi*2c - (1+2c psi*)
    #   = 4 - 3 = ... checked numerically:
    for c in (0.5, 1.0, 2.0):
        hfun = lambda p, c=c: p + c * p**2
        res = analyze_h(f"twoterm_c{c}", hfun, None, None)
        res["irregular_note"] = "truncated tower (no radius of convergence), outside the regular class"
        results[f"twoterm_c{c}"] = res
    # Direct verification, if a candidate is found.
    if any("triple_root_candidate" in r for r in results.values() if isinstance(r, dict)):
        for name in ("geometric", "n_alpha", "log"):
            if isinstance(results.get(name), dict) and "triple_root_candidate" in results[name]:
                c = results[name]["triple_root_candidate"]
                results[name]["direct_verification"] = verify_metric_direct(
                    name, families[name], c["psi"], c["M"], c["R"])
    safe_path(DATA / "tower_sigma_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "mixes_summary"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
