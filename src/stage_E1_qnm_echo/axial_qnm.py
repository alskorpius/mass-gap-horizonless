"""E1 (goal "verifiable physics"): axial perturbations of the horizonless branch.

Part 1 -- symbolic derivation of the axial potential for the metric class
ds^2 = -f(r) dt^2 + dr^2/f(r) + r^2 dOmega^2 with an ARBITRARY m(r)
(f = 1 - 2 m(r)/r). Axial ansatz l=1:
  g_{t phi} = h0(r) sin^2(theta) e^{-i omega t},  g_{r phi} = h1(r) sin(theta) e^{-i omega t}.
Linearized Einstein equations -> 2nd-order ODE for the master function.
Validation: at f = 1 - 2M/r the potential must match Regge-Wheeler
V = f [ l(l+1)/r^2 - 6M/r^3 ] to machine precision; generalization to l: replace
the angular eigenvalue 2 -> beta = l(l+1) (checked against Schwarzschild
at beta = 6).

Part 2 -- time-domain solver d^2 Psi/dt^2 = d^2 Psi/dr*^2 - V Psi,
r* = dr/f, leapfrog scheme; boundaries: Schwarzschild -- ingoing/outgoing
Sommerfeld; horizonless star -- Psi|r=0 = 0 (regularity, V ~ beta/r^2).
Validation: fundamental Schwarzschild QNM, axial l=2:
omega M = 0.37367 - 0.08896 i (classic; reference in SOURCES.md).

Part 3 -- horizonless configurations of the stage-5 family (n=6, eps_c=1,
M/M_ext in {0.95, 0.995}): echo train, extraction of the dominant modes
(matrix pencil/Prony and Fourier), comparison of the echo period with the
geometric-optics estimate tau_echo = 2 Int dr/f (stage 5).

Units: G=c=1. Run: python axial_qnm.py (part1 first, then the dynamics).
"""

import json
from pathlib import Path

import numpy as np
import sympy as sp
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


def safe_path(target) -> Path:
    resolved = Path(target).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path outside project root: {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# Part 1: axial potential (symbolic)
# ---------------------------------------------------------------------------
def derive_axial_potential(verbose=True):
    t, r, th, ph = sp.symbols("t r theta phi", real=True, positive=True)
    f = sp.Function("f")(r)
    h0 = sp.Function("h0")(r)
    h1 = sp.Function("h1")(r)
    w = sp.Symbol("omega", real=True)
    st = sp.sin(th)
    g = sp.Matrix([[-f, 0, 0, h0 * st**2 * sp.exp(-sp.I * w * t)],
                   [0, 1 / f, 0, h1 * st * sp.exp(-sp.I * w * t)],
                   [0, 0, r**2, 0],
                   [0, 0, 0, r**2 * st**2]])
    coords = (t, r, th, ph)
    gi = g.inv()
    N = 4
    gam = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(N)]
    for a in range(N):
        for b in range(N):
            for c in range(N):
                gam[a][b][c] = sp.simplify(sp.expand(
                    sum(gi[a, d] * (sp.diff(g[d, b], coords[c])
                                    + sp.diff(g[d, c], coords[b])
                                    - sp.diff(g[b, c], coords[d]))
                        for d in range(N)) / 2))

    def ricci_full(bb, dd):
        expr = sum(sp.diff(gam[k][bb][dd], coords[k])
                   - sp.diff(gam[k][bb][k], coords[dd]) for k in range(N))
        for k in range(N):
            for lam in range(N):
                expr += gam[k][k][lam] * gam[lam][bb][dd] \
                        - gam[k][bb][lam] * gam[lam][k][dd]
        return sp.expand(expr)

    R_full = sp.MutableDenseMatrix(N, N, lambda i, j: 0)
    for i in range(N):
        for j in range(N):
            R_full[i, j] = ricci_full(i, j)
    ginv = sp.simplify(g.inv())
    scalar_full = sp.simplify(sum(ginv[i, j] * R_full[i, j] for i in range(N) for j in range(N)))
    G_tp = sp.expand(R_full[0, 3] - sp.Rational(1, 2) * g[0, 3] * scalar_full)
    G_rp = sp.expand(R_full[1, 3] - sp.Rational(1, 2) * g[1, 3] * scalar_full)
    # Linearization: h -> eps*h, expand in eps, take the first order.
    eps = sp.Symbol("eps")

    def linearize(expr):
        e1 = expr.subs({h0: eps * h0, h1: eps * h1}, simultaneous=True)
        e1 = sp.expand(sp.series(e1, eps, 0, 2).removeO())
        return sp.simplify(sp.expand(e1).coeff(eps, 1))

    G_tp_lin = linearize(G_tp)
    G_rp_lin = linearize(G_rp)
    # Cancel common angular/time factors.
    G_tp_red = sp.simplify(sp.expand(G_tp_lin / (sp.exp(-sp.I * w * t) * st)))
    G_rp_red = sp.simplify(sp.expand(G_rp_lin / (sp.exp(-sp.I * w * t) * st)))
    # h0 algebraically from G_{t phi}=0 (contains h0 and h1, but not
    # derivatives of h1?), substitute into G_{r phi}=0.
    sol_h0 = sp.solve(sp.Eq(G_tp_red, 0), h0)
    if not sol_h0:
        raise RuntimeError("h0 has no algebraic solution")
    eq_r = sp.simplify(G_rp_red.subs(h0, sol_h0[0]))
    # Master function: Psi = h1 * r / f (standard RW form for this class).
    Psi = sp.Function("Ps")(r)
    expr = sp.expand(eq_r.subs(h1, Psi * f / r, simultaneous=True))
    # Collect the coefficients of Psi'', Psi', Psi.
    a2 = sp.simplify(expr.coeff(sp.Derivative(Psi, r, 2)))
    a1 = sp.simplify(expr.coeff(sp.Derivative(Psi, r)))
    rest = sp.simplify(sp.expand(expr - a2 * sp.Derivative(Psi, r, 2)
                                 - a1 * sp.Derivative(Psi, r)))
    a0 = sp.simplify(rest / Psi)
    # Canonicalization: Psi'' + P Psi' + Q Psi = 0; eliminating the first
    # derivative gives the effective potential u'' + (Q - P'/2 - P^2/4) u = 0 (in r);
    # switching to r* = dr/f adds a division by f^2, and the w^2 term is already in Q.
    P, Q = sp.simplify(a1 / a2), sp.simplify(a0 / a2)
    V_r = sp.simplify(Q - sp.diff(P, r) / 2 - P**2 / 4)
    V = sp.simplify(V_r / f**2)
    # Schwarzschild check.
    M = sp.Symbol("M", positive=True)
    V_schw = sp.simplify(V.subs(f, 1 - 2 * M / r))
    V_rw = sp.simplify((1 - 2 * M / r) * (2 / r**2 - 6 * M / r**3))
    check = sp.simplify(V_schw - V_rw)
    if verbose:
        print("V_axial =", sp.sstr(V))
        print("Schwarzschild check (l=1, RW):", sp.sstr(check))
    return V, check


if __name__ == "__main__":
    V, check = derive_axial_potential()
