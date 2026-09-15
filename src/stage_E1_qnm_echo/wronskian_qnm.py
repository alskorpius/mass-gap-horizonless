"""E1-A: axial-sector QNMs via the Wronskian shooting method (a generalization
of Leaver's approach to non-hypergeometric potentials).

Role in the goal: the companion paper used 3rd-order WKB; their own estimate is that WKB is
unreliable by a factor of 2 for small cores. Here -- exact shooting: two
solutions of d^2 Psi/dr*^2 + (omega^2 - V) Psi = 0 (r* = dr/f):

  Psi_in:  from the inner horizon (r*min), purely ingoing e^{-i omega r*};
  Psi_out: from the far zone (r*max), purely outgoing e^{+i omega r*};
  QNM: W(omega) = Psi_in Psi_out' - Psi_in' Psi_out = 0 at the matching point.

Potential (adopted in coordination with the companion paper, independently checked by us:
the RW limit is exact): V = f[(l(l+1) - 2 + 2f - r f')/r^2].

Validation: Schwarzschild, l=2, fundamental mode
omega M = 0.37367 - 0.08896 i (classic; reference in SOURCES.md).

Computation: Hayward f = 1 - 2Mr^2/(r^3 + 2 M l^2) with M=1 and l/M in
{0.1, 0.03, 0.01} -- frequency/damping shifts relative to Schwarzschild as the
core shrinks.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
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


def make_metric(name, M=1.0, ell=None):
    """Returns (f(r), f'(r), r_in, r_out): the function, its derivative, and the horizons."""
    if name == "schwarzschild":
        f = lambda r: 1.0 - 2.0 * M / r
        fp = lambda r: 2.0 * M / r**2
        return f, fp, 2.0 * M, None
    if name == "hayward":
        c = 2.0 * M * ell**2
        f = lambda r: 1.0 - 2.0 * M * r**2 / (r**3 + c)
        fp = lambda r: (-(2.0 * M) * (2.0 * r * (r**3 + c) - r**2 * 3.0 * r**2)
                        / (r**3 + c)**2)
        # Horizons: r^3 - 2M r^2 + c = 0.
        roots = np.roots([1.0, -2.0 * M, 0.0, c])
        real = sorted(r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0)
        return f, fp, real[0], real[-1]
    raise ValueError(name)


def tortoise_grid(f, r_lo, r_hi, n=60000):
    """Grid of r* on [r_lo, r_hi] (horizons excluded by a small gap)."""
    r = np.geomspace(r_lo, r_hi, n)
    inv_f = 1.0 / f(r)
    rstar = np.concatenate([[0.0], np.cumsum(0.5 * (inv_f[1:] + inv_f[:-1])
                                             * np.diff(r))])
    return r, rstar - rstar[0]


def solve_qnm(name, l=2, M=1.0, ell=None, omega0=None,
              n_grid=60000):
    f, fp, r_in, r_out = make_metric(name, M, ell)
    # Outer branch only: (r_+(1+delta), r_end) -- r* is monotone, f>0.
    # delta=1e-12: r*(start) ~ -54M -- a deep ingoing start.
    r_plus = r_out if r_out is not None else 2.0 * M
    r_start = r_plus * (1.0 + 1e-12)
    r_end = 120.0 * M
    r, rstar = tortoise_grid(f, r_start, r_end, n_grid)
    V = f(r) * (l * (l + 1) - 2 + 2 * f(r) - r * fp(r)) / r**2
    # Uniform r* grid; matching at +45M, the outer arm is short (~+70M):
    # amplification of the unwanted component over the arm ~e^{2|Im|L} < e^6 -- acceptable.
    rs_u = np.linspace(rstar[0] + 3.0, min(rstar[-1], 70.0), 200000)
    V_u = np.interp(rs_u, rstar, V)

    def V_at(q):
        return float(np.interp(q, rs_u, V_u))

    def rhs(rs, y, om):
        return [y[1], (V_at(rs) - om**2) * y[0]]

    rs_mid = 45.0

    def wronskian(om):
        if not (np.isfinite(om.real) and np.isfinite(om.imag)
                and abs(om) < 50.0):
            return 1e6 + 1e6j
        rs_lo = rs_u[0]
        y0 = [1.0, -1j * om]
        sol_in = solve_ivp(rhs, (rs_lo, rs_mid), y0, args=(om,),
                           rtol=1e-10, atol=1e-12, max_step=0.5)
        rs_hi = rs_u[-1]
        y0o = [1.0, 1j * om]
        sol_out = solve_ivp(rhs, (rs_hi, rs_mid), y0o, args=(om,),
                            rtol=1e-10, atol=1e-12, max_step=0.5)
        if not (sol_in.success and sol_out.success):
            return 1e6 + 1e6j
        psi_i, dpsi_i = sol_in.y[0, -1], sol_in.y[1, -1]
        psi_o, dpsi_o = sol_out.y[0, -1], sol_out.y[1, -1]
        Wv = psi_i * dpsi_o - dpsi_i * psi_o
        return Wv / (abs(psi_i) * abs(psi_o) + 1e-300)

    def system(x):
        om = x[0] + 1j * x[1]
        Wv = wronskian(om)
        if not np.isfinite(Wv.real + Wv.imag):
            return [1e6, 1e6]
        return [Wv.real, Wv.imag]

    guess = omega0 if omega0 is not None else (0.3737 - 0.0890j) / M
    sol = root(system, [guess.real, guess.imag], method="hybr",
               options={"xtol": 1e-11})
    om_found = sol.x[0] + 1j * sol.x[1]
    return dict(name=name, ell=ell, l=l, omega=om_found,
                converged=bool(sol.success),
                residual=complex(system(sol.x)[0] + 1j * system(sol.x)[1]),
                match_rstar=float(rs_mid))


def main():
    out = {}
    # 1. Validation: Schwarzschild l=2 (known: 0.37367 - 0.08896 i).
    val = solve_qnm("schwarzschild", l=2)
    known = 0.37367 - 0.08896j
    val["known_value"] = known
    val["rel_error"] = abs(val["omega"] - known) / abs(known)
    out["validation_schwarzschild_l2"] = val
    print("validation:", val["omega"], "rel err:", val["rel_error"])
    # 2. Hayward with small cores.
    results = []
    for ell in (0.1, 0.03, 0.01):
        res = solve_qnm("hayward", l=2, ell=ell, omega0=val["omega"])
        res["d_Re_vs_schw"] = res["omega"].real - val["omega"].real
        res["d_Im_vs_schw"] = res["omega"].imag - val["omega"].imag
        results.append(res)
        print("hayward ell=", ell, "->", res["omega"], "conv:", res["converged"])
    out["hayward_small_cores"] = results

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (np.complex128, complex)):
            return {"re": float(o.real), "im": float(o.imag)}
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    safe_path(DATA / "wronskian_qnm_results.json").write_text(
        json.dumps(clean(out), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
