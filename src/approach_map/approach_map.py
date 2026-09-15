"""Common framework for comparing regular black hole candidates (stage 1b).

All candidates are written in a single class of static spherically symmetric
metrics (G=c=1, lengths in units of L0):

    ds^2 = -f(x) dt^2 + dx^2/f(x) + R(x)^2 dOmega^2,   f = F(R(x)).

Families: Schwarzschild, Hayward, Bardeen, Dymnikova (via F(R) at R=x),
the Simpson-Visser black-bounce (F=1-2m/R, R=sqrt(x^2+a^2)), and pocket v01
(Hayward F, non-monotonic R(x) from src/baseline/pocket_model.py).

The Einstein tensor formulas are taken from src/baseline/pocket_model.py
(independently checked there via Christoffel symbols); here they are applied
to arbitrary F(R), R(x). Notation: S=R''/R, B=f'R'/(2R),
D=(1-f R'^2)/R^2, a=f''/2:
    G^t_t = 2fS+2B-D,  G^x_x = 2B-D,  G^th_th = fS+2B+a,
    K = 4(a^2+2B^2+2C^2+D^2), C=fS+B;  8pi T(n,n) = -2S (n=-d_x, EF map).
Density and pressures are eigenvalues; for f<0 the timelike direction swaps
(t <-> x), as in the baseline. This is a prescribed metric, not dynamics.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "baseline"))
import pocket_model  # noqa: E402  (baseline, unmodified)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCHW_SHADOW = 3.0 * np.sqrt(3.0)  # b_c/m for Schwarzschild


# ---------------------------------------------------------------- F(R) families
def F_schwarzschild(R, m):
    return 1 - 2 * m / R, 2 * m / R**2, -4 * m / R**3


def F_hayward(R, m, ell):
    c0 = 2 * m * ell**2
    den = R**3 + c0
    F = 1 - 2 * m * R**2 / den
    FR = 2 * m * R * (R**3 - 2 * c0) / den**2
    FRR = -4 * m * (R**6 - 7 * c0 * R**3 + c0**2) / den**3
    return F, FR, FRR


def F_bardeen(R, m, g):
    q = R**2 + g**2
    F = 1 - 2 * m * R**2 * q**-1.5
    FR = 2 * m * R * (R**2 - 2 * g**2) * q**-2.5
    FRR = 2 * m * (-2 * R**4 + 11 * R**2 * g**2 - 2 * g**4) * q**-3.5
    return F, FR, FRR


def F_dymnikova(R, m, rstar):
    """M(R) = m (1 - exp(-R^3/rstar^3)); de Sitter core with 8pi rho(0)=6m/rstar^3."""
    s = rstar**3
    E = np.exp(-R**3 / s)
    M = -m * np.expm1(-R**3 / s)
    M1 = 3 * m * R**2 * E / s
    M2 = m * E * (6 * R / s - 9 * R**4 / s**2)
    F = 1 - 2 * M / R
    FR = 2 * M / R**2 - 2 * M1 / R
    FRR = -4 * M / R**3 + 4 * M1 / R**2 - 2 * M2 / R
    return F, FR, FRR


def F_triple_root(R, m, r_minus, r_plus, b2):
    """Carballo-Rubio et al. 2205.13556: F = N/D, N=(R-r_-)^3 (R-r_+), D = N + 2 m R^3 + b2 R^2.
    r_- is the triple root (kappa_- = 0), r_+ is simple; ADM mass m; de Sitter core with 8pi rho(0) = 3 b2/(r_-^3 r_+).
    Regularity requires D > 0 for all R >= 0 (checked in checks.py)."""
    P = np.polynomial.polynomial
    N = P.polymul(P.polypow([-r_minus, 1.0], 3), [-r_plus, 1.0])          # coefficients in ascending order of degree
    Dc = P.polyadd(N, [0.0, 0.0, b2, 2 * m])
    n, n1, n2 = P.polyval(R, N), P.polyval(R, P.polyder(N)), P.polyval(R, P.polyder(N, 2))
    d, d1, d2 = P.polyval(R, Dc), P.polyval(R, P.polyder(Dc)), P.polyval(R, P.polyder(Dc, 2))
    F = n / d
    FR = (n1 * d - n * d1) / d**2
    FRR = (n2 * d - n * d2) / d**2 - 2 * d1 * (n1 * d - n * d1) / d**3
    return F, FR, FRR


def one_minus_F_triple_root(R, m, r_minus, r_plus, b2):
    P = np.polynomial.polynomial
    N = P.polymul(P.polypow([-r_minus, 1.0], 3), [-r_plus, 1.0])
    Dc = P.polyadd(N, [0.0, 0.0, b2, 2 * m])
    return (2 * m * R**3 + b2 * R**2) / P.polyval(R, Dc)


# ---------------------------------------------------------------- R(x) families
def R_identity(x, **_):
    x = np.asarray(x, dtype=float)
    return x, np.ones_like(x), np.zeros_like(x)


def R_bounce(x, a):
    x = np.asarray(x, dtype=float)
    R = np.sqrt(x**2 + a**2)
    return R, x / R, a**2 / R**3


ONE_MINUS_F = {
    F_schwarzschild: lambda R, m: 2 * m / R,
    F_hayward: lambda R, m, ell: 2 * m * R**2 / (R**3 + 2 * m * ell**2),
    F_bardeen: lambda R, m, g: 2 * m * R**2 * (R**2 + g**2) ** -1.5,
    F_dymnikova: lambda R, m, rstar: -2 * m * np.expm1(-R**3 / rstar**3) / R,
    F_triple_root: one_minus_F_triple_root,
}
ONE_MINUS_RP2 = {
    R_identity: lambda x, **_: np.zeros_like(np.asarray(x, dtype=float)),
    R_bounce: lambda x, a: a**2 / (np.asarray(x, dtype=float) ** 2 + a**2),
}


@dataclass
class Family:
    name: str
    label: str
    F: callable
    Fkw: dict
    Rfun: callable = R_identity
    Rkw: dict = field(default_factory=dict)
    x_min: float = 1e-3
    x_max: float = 12.0
    notes: str = ""

    @property
    def m(self):
        return self.Fkw["m"]

    def geometry(self, x):
        R, Rp, Rpp = self.Rfun(x, **self.Rkw)
        F, FR, FRR = self.F(R, **self.Fkw)
        f = F
        fp = FR * Rp
        fpp = FRR * Rp**2 + FR * Rpp
        # D=(1-f R'^2)/R^2 without subtracting nearly equal numbers near the center:
        # 1 - f R'^2 = (1 - R'^2) + (1-f) R'^2, where 1-f and 1-R'^2 are given explicitly.
        one_minus_f = ONE_MINUS_F[self.F](R, **self.Fkw)
        one_minus_Rp2 = ONE_MINUS_RP2[self.Rfun](x, **self.Rkw)
        with np.errstate(divide="ignore", invalid="ignore"):
            D = (one_minus_Rp2 + one_minus_f * Rp**2) / R**2
        return dict(R=R, Rp=Rp, Rpp=Rpp, f=f, fp=fp, fpp=fpp, D=D)

    def grid(self, n_core=3000, n_far=3000):
        lo = self.x_min if self.x_min > 0 else 1e-4 * self.m
        head = np.array([0.0]) if self.x_min == 0 else np.array([])
        return np.unique(np.r_[head, np.geomspace(lo, 0.5 * self.m, n_core),
                               np.linspace(0.5 * self.m, self.x_max, n_far)])


class PocketFamily(Family):
    """Pocket v01: geometry taken directly from the baseline (without modifying the code)."""

    def __init__(self, params: pocket_model.Parameters, label=None):
        self.p = params
        super().__init__(name="pocket", label=label or f"pocket a={params.amplitude} b={params.b:.4g}",
                         F=None, Fkw=dict(m=params.m), x_min=0.0, x_max=10.0 * params.m)

    def geometry(self, x):
        g = pocket_model.geometry(x, self.p)
        return dict(R=g["R"], Rp=g["Rp"], Rpp=g["Rpp"], f=g["f"], fp=g["fp"], fpp=g["fpp"], D=g["D"], B=g["B"])

    def grid(self, **_):
        return pocket_model.profile(self.p)


# ---------------------------------------------------------------- tensors
def tensors(fam: Family, x):
    x = np.asarray(x, dtype=float)
    g = fam.geometry(x)
    R, Rp, Rpp, f, fp, fpp = (g[k] for k in ("R", "Rp", "Rpp", "f", "fp", "fpp"))
    with np.errstate(divide="ignore", invalid="ignore"):
        S = Rpp / R
        B = g["B"] if "B" in g else fp * Rp / (2 * R)
        D = g["D"] if "D" in g else (1 - f * Rp**2) / R**2
    a = 0.5 * fpp
    C = f * S + B
    Gt = 2 * f * S + 2 * B - D
    Gx = 2 * B - D
    Ga = f * S + 2 * B + a
    K = 4 * (a * a + 2 * B * B + 2 * C * C + D * D)
    ricci = -(Gt + Gx + 2 * Ga)
    timelike_t = f >= 0
    rho = np.where(timelike_t, -Gt, -Gx)          # 8*pi*rho
    pr = np.where(timelike_t, Gx, Gt)             # 8*pi*p_r
    pt = Ga                                       # 8*pi*p_perp
    # roundoff error level: terms that G^mu_nu is built from
    noise = 1e-12 * (np.abs(D) + np.abs(B) + np.abs(a) + np.abs(f * S)) + 1e-14
    return dict(x=x, **g, rho8pi=rho, pr8pi=pr, pt8pi=pt, K=K, Ricci=ricci, noise=noise,
                NEC8pi=-2 * S,
                nec_r=rho + pr, nec_t=rho + pt,
                sec=rho + pr + 2 * pt,             # 8pi(rho + p_r + 2 p_perp) = "rho+3p" anisotropic
                dec_r=rho - np.abs(pr), dec_t=rho - np.abs(pt))


# ---------------------------------------------------------------- structure
def horizons(fam: Family, x=None):
    """All zeros of f on the grid (refined with brentq), with surface gravity f'/2."""
    x = fam.grid() if x is None else x
    x = x[x > 0] if isinstance(fam, PocketFamily) else x
    f = fam.geometry(x)["f"]
    out = []
    s = np.sign(f)
    for i in np.nonzero(s[:-1] * s[1:] < 0)[0]:
        xr = brentq(lambda t: float(fam.geometry(t)["f"]), x[i], x[i + 1], xtol=1e-14)
        out.append(xr)
    out += [float(xi) for xi in x[s == 0]]          # zero of f exactly at a grid node
    res = []
    for xr in sorted(out):
        if res and abs(xr - res[-1]["x"]) < 1e-9:
            continue
        g = fam.geometry(xr)
        res.append(dict(x=xr, R=float(g["R"]), kappa=0.5 * float(g["fp"]), area=4 * np.pi * float(g["R"]) ** 2))
    return res


def photon_sphere(fam: Family, x_outer):
    """Outer unstable photon sphere: the outermost local maximum of h=f/R^2 for x>x_outer.
    Returns (x_ph, R_ph, b_c) with b_c=1/sqrt(h_max) — critical impact parameter (shadow radius).
    None if there is no local maximum (a horizonless object without a photon sphere)."""
    lo = max(x_outer * 1.0001, 1e-3 * fam.m)
    xs = np.linspace(lo, 8 * fam.m + x_outer, 4000)
    g = fam.geometry(xs)
    h = g["f"] / g["R"] ** 2
    idx = np.nonzero((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]))[0] + 1
    if idx.size == 0:
        return None
    i = int(idx[-1])
    hn = lambda t: -float(fam.geometry(t)["f"] / fam.geometry(t)["R"] ** 2)
    res = minimize_scalar(hn, bounds=(xs[i - 1], xs[i + 1]), method="bounded", options=dict(xatol=1e-12))
    g = fam.geometry(res.x)
    return dict(x=float(res.x), R=float(g["R"]), b_c=float(1 / np.sqrt(-res.fun)))


def proper_volume(fam: Family, x_lo, x_hi):
    """4pi int R^2/sqrt(f) dx on the static slice; only where f>0."""
    val, err = quad(lambda t: 4 * np.pi * float(fam.geometry(t)["R"] ** 2 / np.sqrt(fam.geometry(t)["f"])),
                    x_lo, x_hi, limit=200)
    return val, err


def refine_max(fam: Family, key, x, arr):
    """Refine the maximum of quantity key near the grid maximum."""
    i = int(np.nanargmax(arr))
    lo, hi = x[max(i - 1, 0)], x[min(i + 1, len(x) - 1)]
    if hi <= lo:
        return float(x[i]), float(arr[i])
    res = minimize_scalar(lambda t: -float(tensors(fam, t)[key]), bounds=(lo, hi), method="bounded",
                          options=dict(xatol=1e-13))
    return float(res.x), float(-res.fun)


def characterize(fam: Family):
    """Unified summary of a candidate (all numbers in units of L0, G=c=1, 8pi absorbed into the source)."""
    x = fam.grid()
    T = tensors(fam, x)
    m = fam.m
    hs = horizons(fam, x)
    out = dict(name=fam.name, label=fam.label, m=m, n_horizons=len(hs), horizons=hs)
    x_ph = None
    if hs:
        outer, inner = hs[-1], hs[0]
        out.update(R_plus=outer["R"], kappa_plus=outer["kappa"], T_H=outer["kappa"] / (2 * np.pi),
                   T_H_over_schw=outer["kappa"] / (2 * np.pi) / (1 / (8 * np.pi * m)),
                   area_plus_over_schw=outer["area"] / (16 * np.pi * m**2))
        if len(hs) >= 2:
            out.update(R_minus=inner["R"], kappa_minus=inner["kappa"])
        ph = photon_sphere(fam, outer["x"])
        if ph:
            out.update(x_ph=ph["x"], R_ph=ph["R"], b_c=ph["b_c"], b_c_over_schw=ph["b_c"] / (SCHW_SHADOW * m))
            x_ph = ph["x"]
        # proper volume of the static region inside the inner horizon (f>0), if it exists
        if len(hs) >= 2 and fam.geometry(0.5 * inner["x"])["f"] > 0:
            V, Verr = proper_volume(fam, fam.x_min if not isinstance(fam, PocketFamily) else 0.0, inner["x"] * (1 - 1e-9))
            out.update(core_volume=V, core_volume_err=Verr)
    else:
        ph = photon_sphere(fam, fam.x_min)
        if ph:
            out.update(x_ph=ph["x"], R_ph=ph["R"], b_c=ph["b_c"], b_c_over_schw=ph["b_c"] / (SCHW_SHADOW * m))
        else:
            out["photon_sphere"] = "none"
    # curvature
    xK, Kmax = refine_max(fam, "K", x, T["K"])
    out.update(K_max=Kmax, x_K_max=xK, K_center=float(T["K"][0]), x_first=float(x[0]),
               K_max_over_schw_at_R=Kmax / (48 * m**2 / float(fam.geometry(xK)["R"]) ** 6))
    # source and energy conditions
    for key in ("rho8pi", "nec_r", "nec_t", "sec", "dec_r", "dec_t"):
        i = int(np.nanargmin(T[key]))
        out[f"min_{key}"] = float(T[key][i])
        out[f"x_min_{key}"] = float(x[i])
    out["rho8pi_center"] = float(T["rho8pi"][0])
    with np.errstate(divide="ignore", invalid="ignore"):
        out["w_r_center"] = float(T["pr8pi"][0] / T["rho8pi"][0]) if T["rho8pi"][0] != 0 else np.nan
        out["w_t_center"] = float(T["pt8pi"][0] / T["rho8pi"][0]) if T["rho8pi"][0] != 0 else np.nan
    tol = T["noise"] * 10 + 1e-9 * max(1.0, abs(out["rho8pi_center"]))
    out["NEC_violated"] = bool((T["nec_r"] < -tol).any() or (T["nec_t"] < -tol).any())
    out["WEC_violated"] = bool(out["NEC_violated"] or (T["rho8pi"] < -tol).any())
    out["SEC_violated"] = bool(out["NEC_violated"] or (T["sec"] < -tol).any())
    out["DEC_violated"] = bool(out["WEC_violated"] or (T["dec_r"] < -tol).any() or (T["dec_t"] < -tol).any())
    # where SEC (rho+3p) < 0: region of "repulsive" effective gravity
    neg = x[T["sec"] < -tol]
    out["sec_negative_range"] = [float(neg.min()), float(neg.max())] if neg.size else None
    # exterior deviations from Schwarzschild at the same m
    # df: deviation of g_tt; dgRR: deviation of the radial coefficient in areal radius,
    # g_RR = 1/(f R'^2), relative to the Schwarzschild 1/(1-2m/R) at the same m
    for r_over_m in (3.0, 6.0, 10.0):
        xr = _x_for_R(fam, r_over_m * m)
        g = fam.geometry(xr)
        fs = 1 - 2 * m / g["R"]
        out[f"df_at_R{int(r_over_m)}m"] = float(g["f"] - fs)
        out[f"dgRR_at_R{int(r_over_m)}m"] = float(fs / (g["f"] * g["Rp"] ** 2) - 1)
    return out, x, T


def _x_for_R(fam, R_target):
    if isinstance(fam, PocketFamily):
        return pocket_model.x_for_outer_branch(R_target, fam.p)
    if fam.Rfun is R_bounce:
        return float(np.sqrt(max(R_target**2 - fam.Rkw["a"] ** 2, 0.0)))
    return float(R_target)


# ---------------------------------------------------------------- factories
def make_families(m=1.0, ell=2 / 3, g=None, rstar=None, a=None, pocket=True):
    # Bardeen and Dymnikova: g^3 = r*^3 = 2 m ell^2 gives the same central density 8pi rho(0)=3/ell^2
    g = (2 * m * ell**2) ** (1 / 3) if g is None else g
    rstar = (2 * m * ell**2) ** (1 / 3) if rstar is None else rstar
    a = ell if a is None else a
    fams = [
        Family("schwarzschild", "Schwarzschild", F_schwarzschild, dict(m=m), notes="control; singular"),
        Family("hayward", f"Hayward ell={ell:.4g}", F_hayward, dict(m=m, ell=ell), notes="de Sitter core"),
        Family("bardeen", f"Bardeen g={ell:.4g}", F_bardeen, dict(m=m, g=ell), notes="de Sitter core; g=ell (density 6m/g^3)"),
        Family("bardeen_samerho", f"Bardeen g={g:.4g} (same rho_c)", F_bardeen, dict(m=m, g=g),
               notes="same central density as Hayward"),
        Family("dymnikova", f"Dymnikova r*={rstar:.4g}", F_dymnikova, dict(m=m, rstar=rstar), notes="vacuum-like core"),
        Family("simpson_visser", f"Simpson-Visser a={a:.4g}", F_schwarzschild, dict(m=m), R_bounce, dict(a=a),
               x_min=0.0, notes="black-bounce; throat at x=0"),
        Family("triple_root", "kappa_-=0 (2205.13556) r_-=0.5 r_+=2 b2=0.5625", F_triple_root,
               dict(m=m, r_minus=0.5, r_plus=2.0, b2=0.5625),
               notes="triple inner root: kappa_-=0; b2 tuned to 8pi rho_c = 6.75"),
    ]
    if pocket:
        fams.append(PocketFamily(pocket_model.Parameters(m=m, ell=ell)))
    return fams
