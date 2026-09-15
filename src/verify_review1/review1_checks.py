"""Response to external review #1 (items P1, P2, P5; P3 is assembled from
existing artifacts of both projects; P4/P6 -- as text in report.md).

P1: kappa_- and the mass-inflation e-fold time for the baseline the companion paper family
    (n=6, eps_c from the mass-gap window) at M = 3/10/30 M_sun.
P2: the sign of rho'(r) for the CR profile (our stage 7) and for the monotone
    the companion paper family (their code is imported READ-ONLY from ../the companion paper; the check
    runs on our own independent machinery: our own quadrature for m(r), our
    own horizons, our own finite-difference rho'). Output: NEC_t = -r rho'/2,
    signs, triple root.
P5: the unified formula ell^2 = ell_0^2 + lambda^2 M^2:
    ell_0(eps_c), M* = ell_0/lambda, the horizonless-branch boundary in terms
    of ell_0/M, the M=10 M_sun point (ell, r_-, r_+, kappa_-) in our family.

Units: G=c=1; lengths in km, masses in km (M_sun = 1.47670 km).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "CITATION.cff").is_file() and (candidate / "src").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("the companion paper root not found")


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


# The monotone triple-root family of the companion paper. The module is vendored
# in src/bhp2_family/ so that this check runs without the companion repository;
# it is byte-identical to the one published in inner-extremal-rbh.

C_KM_S = 299792.458
MSUN_KM = 1.47670
EPS_NUC_KM2 = 2.0793e-4  # nuclear saturation density in geometric units
N = 6.0


# ---------------- common: our n=6 family ----------------
def rc_of(M, eps_c):
    from scipy.special import gamma as gamma_fn
    return (M * N / (4.0 * np.pi * eps_c * gamma_fn(3.0 / N))) ** (1.0 / 3.0)


def m_of(r, rc, M, eps_c):
    from scipy.special import gammainc
    return M * gammainc(3.0 / N, (r / rc) ** N)


def f_of(r, rc, M, eps_c):
    return 1.0 - 2.0 * m_of(r, rc, M, eps_c) / r


def fp_of(r, rc, M, eps_c):
    rho = eps_c * np.exp(-((r / rc) ** N))
    return -8.0 * np.pi * r * rho + 2.0 * m_of(r, rc, M, eps_c) / r**2


def horizons_and_kappa(rc, M, eps_c):
    grid = np.unique(np.r_[np.geomspace(1e-6, rc * 0.02, 200),
                           np.linspace(rc * 0.02, 30.0 * M, 30000)])
    f = f_of(grid, rc, M, eps_c)
    zeros = []
    for j in range(len(grid) - 1):
        if f[j] * f[j + 1] < 0:
            zeros.append(brentq(lambda q: float(f_of(np.array([q]), rc, M, eps_c)[0]),
                                float(grid[j]), float(grid[j + 1]), xtol=1e-14))
    uniq = []
    for z in sorted(zeros):
        if not uniq or abs(z - uniq[-1]) > 1e-9:
            uniq.append(z)
    kaps = [float(fp_of(np.array([z]), rc, M, eps_c)[0]) / 2.0 for z in uniq]
    return uniq, kaps


# ---------------- P1 ----------------
def p1():
    rows = []
    for mult in (5.0, 10.0, 20.0):
        eps_c = EPS_NUC_KM2 * mult
        for m_msun in (3.0, 10.0, 30.0):
            M = m_msun * MSUN_KM
            rc = rc_of(M, eps_c)
            hz, kap = horizons_and_kappa(rc, M, eps_c)
            row = dict(eps_mult=mult, M_msun=m_msun, n_horizons=len(hz))
            if len(hz) >= 2:
                kminus = abs(kap[0])
                row.update(r_minus_km=hz[0], r_plus_km=hz[-1],
                           kappa_minus_km=kminus,
                           t_efold_ms=(1.0 / kminus) / C_KM_S * 1e3,
                           t_H_ms=(4 * np.pi / abs(kap[-1])) / C_KM_S * 1e3)
            rows.append(row)
    return rows


# ---------------- P2 ----------------
def cr_source():
    """CR metric (our stage 7): sympy derivatives -> rho(r)=m'/(4 pi r^2)."""
    import sympy as sp
    M, rm, rp, a2 = 1.0, 0.01, 2.0, 0.1
    r = sp.symbols("r", positive=True)
    num = (r - rm)**3 * (r - rp)
    den = num + 2 * M * r**3 + (a2 - 3 * rm * (rp + rm)) * r**2
    F = num / den
    m = r * (1 - F) / 2
    rho = sp.diff(m, r) / (4 * sp.pi * r**2)
    drho = sp.diff(rho, r)
    return (sp.lambdify(r, rho, "numpy"), sp.lambdify(r, drho, "numpy"),
            sp.lambdify(r, F, "numpy"), sp.lambdify(r, sp.diff(F, r), "numpy"),
            sp.lambdify(r, sp.diff(F, r, 2), "numpy"))


def cr_horizons_exact():
    import sympy as sp
    M, rm, rp, a2 = 1.0, 0.01, 2.0, 0.1
    r = sp.symbols("r", positive=True)
    num = (r - rm)**3 * (r - rp)
    den = num + 2 * M * r**3 + (a2 - 3 * rm * (rp + rm)) * r**2
    F = num / den
    return (float(F.subs(r, rm)), float(sp.diff(F, r).subs(r, rm)),
            float(sp.diff(F, r, 2).subs(r, rm)),
            float(F.subs(r, rp)), float(sp.diff(F, r).subs(r, rp)))


def bhp2_family():
    """the companion paper's monotone family (variant B, their s1*, d=s1*-1, scale=1/H(u_s)).
    Their modules are imported READ-ONLY; every check below is ours."""
    stability_dir = PROJECT_ROOT / "src" / "bhp2_family"
    sys.path.insert(0, str(stability_dir))
    spec = importlib.util.spec_from_file_location(
        "trm", stability_dir / "triple_root_monotone.py")
    trm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trm)
    s1_star, u_s, info = trm.find_merge_first(sigma_min=1.0)
    kw = info["kw"]
    d = s1_star - 1.0
    sig, s, mI, H, Hp = trm.profile(d, **kw)
    scale = 1.0 / float(np.interp(u_s, trm.U, H))
    fam = trm.MonotoneFamily(d, scale, kw=kw)
    return fam, dict(s1_star=s1_star, u_s=u_s, d=d, scale=scale,
                     sigma_min=float(sig.min()))


def audit_profile(rho_callable, r_grid, label):
    """Independent audit: our own m(r), our own horizons, our own rho'."""
    r = r_grid
    rho = np.maximum(rho_callable(r), 0.0)
    # our own mass quadrature
    m = np.concatenate([[0.0], 2.0 * np.pi * np.cumsum(
        np.diff(r) * (r[:-1]**2 * rho[:-1] + r[1:]**2 * rho[1:]))])
    f = 1.0 - 2.0 * m / r
    zeros = []
    for j in range(len(r) - 1):
        if f[j] * f[j + 1] < 0:
            zeros.append(brentq(lambda q: float(
                1.0 - 2.0 * m_of_interp(q, r, m) / q), float(r[j]), float(r[j + 1])))
    uniq = []
    for z in sorted(zeros):
        if not uniq or abs(z - uniq[-1]) > 1e-8:
            uniq.append(z)
    drho = np.gradient(rho, r)
    out = dict(label=label, n_zeros=len(uniq), r_zeros=[float(z) for z in uniq])
    # triple root: minimum |f| near R_-, f' and f'' there
    if uniq:
        r_minus = uniq[0]
        fm = float(1.0 - 2.0 * m_of_interp(r_minus, r, m) / r_minus)
        fp = np.gradient(f, r)
        fpp = np.gradient(fp, r)
        im = int(np.argmin(np.abs(r - r_minus)))
        out.update(f_at_rminus=fm, fprime_at_rminus=float(fp[im]),
                   fpp_at_rminus=float(fpp[im]))
    # NEC_t = -r rho'/2: minimum over r > cutoff
    mask = r > r[1]
    nec_t = -(r[mask] * drho[mask]) / 2.0
    out.update(min_nec_t=float(np.min(nec_t)),
               r_of_min_nec_t=float(r[mask][int(np.argmin(nec_t))]),
               max_rho_prime=float(np.max(drho[mask])),
               r_of_max_rho_prime=float(r[mask][int(np.argmax(drho[mask]))]),
               rho_min=float(np.min(rho[mask])), rho_max=float(np.max(rho)))
    # sign of rho outside the last horizon
    if uniq:
        mask_out = r > uniq[-1] * 1.05
        out.update(rho_outside_min=float(np.min(rho[mask_out])))
    return out, r, rho, drho


def m_of_interp(q, r_grid, m_grid):
    return float(np.interp(q, r_grid, m_grid))


# ---------------- P5 ----------------
def p5():
    lam = 0.27101577964329177
    from scipy.special import gamma as gamma_fn
    c_ext = 0.23136165195307468  # n=6 (stage 4)
    rows = {"ell0": [], "unified": []}
    for mult in (5.0, 10.0, 20.0):
        eps_c = EPS_NUC_KM2 * mult
        ell0 = float(np.sqrt(3.0 / (8.0 * np.pi * eps_c)))
        rows["ell0"].append(dict(eps_mult=mult, ell0_km=ell0,
                                 M_star_msun=ell0 / lam / MSUN_KM))
    # horizonless-branch boundary in terms of ell0/M:
    lam_crit = float(np.sqrt(3.0 / (8.0 * np.pi)) / c_ext)
    rows["lambda_crit_horizonless"] = lam_crit
    rows["bhp2_lambda_range"] = [0.13, 0.30]
    # the 10 M_sun point from the unified formula
    for mult in (5.0, 10.0, 20.0):
        eps_c = EPS_NUC_KM2 * mult
        ell0 = np.sqrt(3.0 / (8.0 * np.pi * eps_c))
        for m_msun in (10.0, 30.0):
            M = m_msun * MSUN_KM
            ell = float(np.sqrt(ell0**2 + (lam * M)**2))
            eps_eff = 3.0 / (8.0 * np.pi * ell**2)
            rc = rc_of(M, eps_eff)
            hz, kap = horizons_and_kappa(rc, M, eps_eff)
            rows["unified"].append(dict(eps_mult=mult, M_msun=m_msun,
                                        ell_km=ell, ell_over_M=ell / M,
                                        n_horizons=len(hz),
                                        r_minus_km=hz[0] if hz else None,
                                        r_plus_km=hz[-1] if hz else None,
                                        kappa_minus_km=abs(kap[0]) if kap else None,
                                        M_sqrt_eps=0.34549 * M / ell))
    return rows


def main():
    out = {}
    print("P1 ...")
    out["P1"] = p1()
    print(json.dumps(out["P1"], ensure_ascii=False, indent=1))
    print("P2: CR ...")
    rho_cr, drho_cr, F_cr, Fp_cr, Fpp_cr = cr_source()
    r_cr = np.unique(np.r_[np.geomspace(1e-5, 0.02, 1200),
                           np.linspace(0.02, 12.0, 12000)])
    # rho and rho' -- analytic (sympy), no sign clamp
    rho_v = rho_cr(r_cr)
    drho_v = drho_cr(r_cr)
    nec_t = -(r_cr * drho_v) / 2.0
    out["P2_CR"] = dict(
        label="CR (the companion paper, stage 7)",
        rho_max=float(np.max(rho_v)), rho_min=float(np.min(rho_v)),
        r_rho_min=float(r_cr[int(np.argmin(rho_v))]),
        max_rho_prime=float(np.max(drho_v)),
        r_max_rho_prime=float(r_cr[int(np.argmax(drho_v))]),
        min_nec_t=float(np.min(nec_t)), r_min_nec_t=float(r_cr[int(np.argmin(nec_t))]))
    out["P2_CR"]["horizons_exact"] = dict(
        r_minus=0.01, r_plus=2.0,
        F_at_rm=float(F_cr(np.array([0.01]))[0]),
        Fp_at_rm=float(Fp_cr(np.array([0.01]))[0]),
        Fpp_at_rm=float(Fpp_cr(np.array([0.01]))[0]),
        F_at_rp=float(F_cr(np.array([2.0]))[0]),
        Fp_at_rp=float(Fp_cr(np.array([2.0]))[0]))
    print(json.dumps(out["P2_CR"], ensure_ascii=False, indent=1))
    r1 = r_cr
    dr1 = drho_v
    print("P2: the companion paper monotone family ...")
    fam, meta = bhp2_family()
    meta["rho_c"] = float(fam.rho_c)
    meta["R1"] = float(fam.R1)
    meta["ell_over_M"] = float(np.sqrt(3.0 / (8.0 * np.pi * fam.rho_c)))
    r_b2 = np.unique(np.r_[np.geomspace(1e-4, 0.02 * fam.R1, 400),
                           np.linspace(0.02 * fam.R1, 12.0, 8000)])
    audit_b2, r2, rho2, dr2 = audit_profile(fam.rho, r_b2, "the companion paper monotone")
    out["P2_BHP2"] = dict(meta=meta, audit=audit_b2)
    print(json.dumps(out["P2_BHP2"], ensure_ascii=False, indent=1, default=str))
    # plot of rho'(r)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        axs[0].plot(r1, dr1 / np.max(np.abs(dr1)), label="CR (the companion paper, stage 7)")
        axs[0].axhline(0, color="k", lw=0.7)
        axs[0].set_xscale("log")
        axs[0].set_title(r"$\rho'(r)$, normalized: CR profile")
        axs[0].set_xlabel("r (units of M=1)"); axs[0].legend()
        axs[1].plot(r2 / fam.R1, dr2 / np.max(np.abs(dr2)), label="the companion paper monotone")
        axs[1].axhline(0, color="k", lw=0.7)
        axs[1].set_xscale("log")
        axs[1].set_title(r"$\rho'(r)$, normalized: the companion paper profile (R/R1)")
        axs[1].set_xlabel("R/R1"); axs[1].legend()
        fig.tight_layout()
        fig.savefig(safe_path(FIGS / "rho_prime_profiles.png"), dpi=150)
    except Exception as exc:  # the plot is not critical
        out["plot_error"] = str(exc)
    print("P5 ...")
    out["P5"] = p5()
    print(json.dumps(out["P5"], ensure_ascii=False, indent=1, default=str))
    safe_path(DATA / "review1_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")


if __name__ == "__main__":
    main()
