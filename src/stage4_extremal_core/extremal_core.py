"""Stage 4: extremal configurations of the vacuum-like core — a single
horizon with zero surface gravity.

Motivation: for the stage-3 two-horizon regular black holes, the inner
horizon has kappa_- != 0 — this is the driver of mass inflation
(Poisson-Israel; Carballo-Rubio et al., 2205.13556). If the family contains a
configuration with a single degenerate horizon (f touches zero: f=0, f'=0),
then kappa=0 and the exponential driver disappears.

Generalization of the stage-3 family to two parameters at fixed mass:
eps(r) = eps_c exp(-(r/rc)^n), rc is fitted from the given total mass:
M = 4 pi eps_c rc^3 Gamma(3/n)/n. The parameter n controls the sharpness of
the core transition. As n grows (fixed M), horizons appear through a
degenerate configuration n_ext(M): n < n_ext -- no horizons; n = n_ext -- one
degenerate horizon; n > n_ext -- two horizons with kappa_- != 0.

For each model: number of horizons, r_-, r_+, kappa_+/-, K(0) (de Sitter
check 512 pi^2 eps_c^2/3), K_max, photon sphere, and the critical impact
parameter b_crit against the Schwarzschild value 3 sqrt(3) M. Cross-check:
n=6 reproduces the stage-3 numbers.

Limitations (honestly): static; the degenerate horizon is a boundary case
requiring fine-tuning of n; for degenerate horizons, Aretakis growth of
massless fields along the horizon is known (slow, polynomial); stability
against radial perturbations has not been checked; the microphysics of
eps_c is not derived.
"""

import csv
import io
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma as gamma_fn, gammainc
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    """The repository root is located via the CITATION.cff and src/ markers, without ".."."""
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
    """Normalize the path and guarantee that it lies within the project root."""
    resolved = Path(target).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path outside project root: {resolved}")
    return resolved


EPS_C = 1.0


def rc_for_mass(M, n):
    return (M * n / (4.0 * np.pi * EPS_C * gamma_fn(3.0 / n))) ** (1.0 / 3.0)


def eps_of_r(r, rc, n):
    return EPS_C * np.exp(-((r / rc) ** n))


def m_of_r(r, rc, n, M):
    return M * gammainc(3.0 / n, (r / rc) ** n)


def f_and_derivs(r, rc, n, M):
    eps = eps_of_r(r, rc, n)
    m = m_of_r(r, rc, n, M)
    f = 1.0 - 2.0 * m / r
    fp = -8.0 * np.pi * r * eps + 2.0 * m / r**2
    m1 = 4.0 * np.pi * r**2 * eps
    m2 = 8.0 * np.pi * r * eps + 4.0 * np.pi * r**2 * eps * (-n * r**(n - 1) / rc**n)
    fpp = -2.0 * m2 / r + 4.0 * m1 / r**2 - 4.0 * m / r**3
    return f, fp, fpp, eps, m


def kretzmann(fp, fpp, r, m):
    A = fpp / 2.0
    B = fp / (2.0 * r)
    D = 2.0 * m / r**3
    return 4.0 * (A**2 + 4.0 * B**2 + D**2)


def grid_for(rc, M):
    return np.unique(np.r_[np.geomspace(1e-6, rc * 0.01, 100),
                           np.linspace(rc * 0.01, 3.0 * rc, 4000),
                           np.linspace(3.0 * rc, max(20.0 * rc, 12.0 * M), 6000)])


def horizon_structure(rc, n, M):
    r = grid_for(rc, M)
    f, fp, fpp, eps, m = f_and_derivs(r, rc, n, M)
    fmin = float(np.min(f))
    r_fmin = float(r[int(np.argmin(f))])
    zeros = []
    for j in range(len(r) - 1):
        if f[j] == 0.0:
            zeros.append(float(r[j]))
        elif f[j] * f[j + 1] < 0:
            zeros.append(brentq(lambda q: float(1.0 - 2.0 * m_of_r(q, rc, n, M) / q),
                                float(r[j]), float(r[j + 1]), xtol=1e-14))
    uniq = []
    for z in sorted(zeros):
        if not uniq or abs(z - uniq[-1]) > 1e-9:
            uniq.append(z)
    kappas = [float(0.5 * f_and_derivs(np.array([z]), rc, n, M)[1][0]) for z in uniq]
    r_center = np.array([1e-6])
    fp_c, fpp_c = f_and_derivs(r_center, rc, n, M)[1:3]
    m_c = f_and_derivs(r_center, rc, n, M)[4]
    return dict(f_min=fmin, r_fmin=r_fmin, horizons=uniq, kappas=kappas,
                K_center=float(kretzmann(fp_c, fpp_c, r_center, m_c)[0]),
                K_max=float(np.max(kretzmann(fp, fpp, r, m))),
                r_grid=r, f_grid=f)


def find_M_ext(n, tol=1e-13):
    """Critical mass M_ext(n): bisection over M, f_min(M) changes sign."""
    def g(M):
        rc = rc_for_mass(M, n)
        return horizon_structure(rc, n, M)["f_min"]
    lo, hi = 1e-4, 10.0
    if not (g(lo) > 0 > g(hi)):
        return None
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def find_next(M, lo, hi, tol=1e-12):
    """Root of f_min(n)=0 by bisection over n: boundary of horizon formation."""
    def g(nn):
        rc = rc_for_mass(M, nn)
        return horizon_structure(rc, nn, M)["f_min"]
    glo, ghi = g(lo), g(hi)
    if not (glo > 0 > ghi):
        return None, None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if abs(gm) < tol or hi - lo < 1e-14:
            return mid, gm
        if gm > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), g(0.5 * (lo + hi))


def photon_observation(rc, n, M, r_outer):
    """Photon sphere outside the outermost horizon (or outside the core) and b_crit."""
    def h(q):
        f, fp, _, _, _ = f_and_derivs(np.array([q]), rc, n, M)
        return float(q * fp[0] - 2.0 * f[0])
    lo = max(r_outer * (1.0 + 1e-9), 1e-4 * max(rc, M))
    hi = 30.0 * M
    hlo, hhi = h(lo), h(hi)
    if not (np.isfinite(hlo) and np.isfinite(hhi)) or hlo * hhi > 0:
        return None, None
    r_ph = brentq(h, lo, hi, xtol=1e-13)
    f_ph = float(f_and_derivs(np.array([r_ph]), rc, n, M)[0][0])
    return r_ph, r_ph / np.sqrt(f_ph)


def audit_mass(M, n_values):
    rows = []
    for n in n_values:
        rc = rc_for_mass(M, n)
        st = horizon_structure(rc, n, M)
        r_ph, b_crit = photon_observation(rc, n, M, max(st["horizons"], default=0.0))
        rows.append(dict(M=M, n=n, rc=rc, f_min=st["f_min"], r_fmin=st["r_fmin"],
                         n_horizons=len(st["horizons"]),
                         r_minus=st["horizons"][0] if st["horizons"] else np.nan,
                         r_plus=st["horizons"][-1] if st["horizons"] else np.nan,
                         kappa_minus=st["kappas"][0] if st["kappas"] else np.nan,
                         kappa_plus=st["kappas"][-1] if st["kappas"] else np.nan,
                         K_center=st["K_center"], K_max=st["K_max"],
                         r_photon=r_ph if r_ph else np.nan,
                         b_crit=b_crit if b_crit else np.nan,
                         b_crit_schw=3.0 * np.sqrt(3.0) * M,
                         shadow_dev=(b_crit - 3.0 * np.sqrt(3.0) * M)
                                    / (3.0 * np.sqrt(3.0) * M) if b_crit else np.nan))
    return rows


def main():
    summary = {}
    all_rows = []
    # 1) Critical masses (extremal configurations) for several profile sharpnesses.
    for n in (2.0, 6.0, 40.0):
        M_ext = find_M_ext(n)
        entry = dict(n=n, M_ext=M_ext, c_ext=M_ext * np.sqrt(EPS_C))
        if M_ext is not None:
            rc = rc_for_mass(M_ext, n)
            st = horizon_structure(rc, n, M_ext)
            kappa_touch = float(f_and_derivs(np.array([st["r_fmin"]]), rc, n, M_ext)[1][0]) / 2
            entry.update(rc_ext=rc, n_horizons_found=len(st["horizons"]),
                         r_touch=st["r_fmin"], kappa_at_touch=kappa_touch,
                         f_min=st["f_min"], K_center=st["K_center"],
                         K_max=st["K_max"], compactness_2M_over_rc=2 * M_ext / rc,
                         two_M_over_r_touch=2 * M_ext / st["r_fmin"])
        summary[f"n={n}"] = entry
    # 2) Snapshots of the neighborhood of the extremal configuration at n=6.
    n0 = 6.0
    M_ext = summary[f"n={n0}"]["M_ext"]
    for tag, M in (("below", M_ext * 0.98), ("ext", M_ext), ("above", M_ext * 1.02)):
        rc = rc_for_mass(M, n0)
        st = horizon_structure(rc, n0, M)
        r_ph, b_crit = photon_observation(rc, n0, M, max(st["horizons"], default=0.0))
        kappa_touch = float(f_and_derivs(np.array([st["r_fmin"]]), rc, n, M)[1][0]) / 2
        row = dict(tag=tag, n=n0, M=M, rc=rc, f_min=st["f_min"],
                   n_horizons=len(st["horizons"]),
                   r_minus=st["horizons"][0] if st["horizons"] else None,
                   r_plus=st["horizons"][-1] if st["horizons"] else None,
                   kappa_minus=st["kappas"][0] if st["kappas"] else None,
                   kappa_plus=st["kappas"][-1] if st["kappas"] else None,
                   kappa_at_fmin=kappa_touch, r_fmin=st["r_fmin"],
                   K_center=st["K_center"], K_max=st["K_max"],
                   r_photon=r_ph, b_crit=b_crit,
                   b_crit_schw=3 * np.sqrt(3) * M)
        summary[f"snapshot_n6_{tag}"] = row
        all_rows.append(row)
    # 3) Cross-check with stage 3: n=6, M=0.3545247944545191.
    M3 = 0.3545247944545191
    rc3 = rc_for_mass(M3, 6.0)
    st3 = horizon_structure(rc3, 6.0, M3)
    summary["cross_check_stage3_n6"] = dict(
        M=M3, rc=rc3, rc_stage3=0.4570927728269982,
        K_center=st3["K_center"], K_center_expected=512 * np.pi**2 / 3.0,
        n_horizons=len(st3["horizons"]), K_max=st3["K_max"],
        K_max_stage3=2038.021567827425)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(all_rows[0].keys()),
                            lineterminator="\r\n", extrasaction="ignore")
    writer.writeheader()
    for r in all_rows:
        writer.writerow(r)
    safe_path(DATA / "extremal_snapshots.csv").write_text(
        buf.getvalue(), encoding="utf-8", newline="")
    safe_path(DATA / "stage4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
