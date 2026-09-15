"""Stage 5: observable properties of the horizonless branch of the core family.

Context: a typical (not fine-tuned) object above the critical compactness
M sqrt(eps_c) > c_ext has an inner horizon with kappa_- != 0 (the
mass inflation problem, stage 4). The horizonless branch M sqrt(eps_c) < c_ext
is a candidate for a self-consistent horizonless alternative to a BH. Stability
under radial perturbations cannot be determined without source microphysics
(the inverse problem does not specify the response to perturbations; the EOS
p_r = -eps would give an imaginary sound speed — this is an interpretational
trap, see the report). So here we compute only quantities that depend on
geometry alone:

  - photon spheres (all roots of r f'(r) = 2 f(r)),
  - compactness 2M/R_99 (R_99 is the radius enclosing 99% of the mass),
  - echo time tau_echo = 2 Integral_0^{r_ph,outer} dr / f(r)
    (in static-slice time; diverges as f_min -> 0),
  - for the neighboring two-horizon branch — Hawking temperature T_H = kappa_+/2pi
    to compare mass-loss channels.

Regarding the user's hypothesis: the horizonless branch has no standard
Hawking channel (no horizon, no kappa). If mass loss is
driven by matter converting to an ultradense phase (the user's
hypothesis), the scale is set by eps_c, not M — the mass dependence
differs from T_H ~ 1/M. This difference is a potential testable
signature; the mechanism itself is not modeled here.
"""

import csv
import io
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
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


def rc_for_mass(M):
    return (M * N / (4.0 * np.pi * EPS_C * gamma_fn(3.0 / N))) ** (1.0 / 3.0)


def eps_of_r(r, rc):
    return EPS_C * np.exp(-((r / rc) ** N))


def m_of_r(r, rc, M):
    return M * gammainc(3.0 / N, (r / rc) ** N)


def f_of(r, rc, M):
    return 1.0 - 2.0 * m_of_r(r, rc, M) / r


def fprime(r, rc, M):
    eps = eps_of_r(r, rc)
    m = m_of_r(r, rc, M)
    return -8.0 * np.pi * r * eps + 2.0 * m / r**2


def photon_spheres(rc, M):
    r = np.unique(np.r_[np.geomspace(1e-6, rc * 0.02, 200),
                        np.linspace(rc * 0.02, 12.0 * M, 12000)])

    def h(q):
        return float(q * fprime(np.array([q]), rc, M)[0]
                     - 2.0 * f_of(np.array([q]), rc, M)[0])
    roots = []
    hv = np.array([h(q) for q in r])
    for j in range(len(r) - 1):
        if hv[j] * hv[j + 1] < 0:
            roots.append(brentq(h, float(r[j]), float(r[j + 1]), xtol=1e-13))
    return roots


def echo_time(rc, M, r_ph_outer):
    value, err = quad(lambda q: 1.0 / float(f_of(np.array([q]), rc, M)[0]),
                      1e-9, r_ph_outer, epsabs=1e-10, epsrel=1e-10, limit=500)
    return 2.0 * value, 2.0 * err


def one_model(M):
    rc = rc_for_mass(M)
    fmin_grid = np.linspace(rc * 0.05, 4.0 * rc, 8000)
    fmin = float(np.min(f_of(fmin_grid, rc, M)))
    r99 = float(rc * gammaincinv(3.0 / N, 0.99))
    ps = photon_spheres(rc, M)
    row = dict(M=M, rc=rc, f_min=fmin,
               R99=r99, compactness_2M_over_R99=2.0 * M / r99,
               n_photon_spheres=len(ps),
               r_photon_outer=ps[-1] if ps else np.nan,
               r_photon_inner=ps[0] if len(ps) > 1 else np.nan)
    if ps:
        tau, tau_err = echo_time(rc, M, ps[-1])
        row.update(echo_time=tau, echo_time_err=tau_err,
                   echo_time_over_2M=tau / (2.0 * M))
    else:
        row.update(echo_time=np.nan, echo_time_err=np.nan,
                   echo_time_over_2M=np.nan)
    return row


def main():
    # Critical mass for n=6 from stage 4.
    M_ext = 0.23136165195307468
    Ms = np.r_[np.linspace(0.04, M_ext * 0.95, 12),
               M_ext * np.array([0.97, 0.99, 0.995, 0.999, 0.9999])]
    rows = [one_model(float(M)) for M in Ms]
    # Two-horizon neighbors: T_H = kappa_+/2pi.
    for tag, M in (("1.001x", M_ext * 1.001), ("1.01x", M_ext * 1.01),
                   ("1.05x", M_ext * 1.05)):
        rc = rc_for_mass(M)
        r = np.linspace(rc * 0.05, 4.0 * rc, 8000)
        f = f_of(r, rc, M)
        zeros = []
        for j in range(len(r) - 1):
            if f[j] * f[j + 1] < 0:
                zeros.append(brentq(lambda q: float(f_of(np.array([q]), rc, M)[0]),
                                    float(r[j]), float(r[j + 1]), xtol=1e-13))
        kappas = [float(fprime(np.array([z]), rc, M)[0]) / 2.0 for z in zeros]
        rows.append(dict(M=M, rc=rc, f_min=float(np.min(f)),
                         R99=np.nan, compactness_2M_over_R99=np.nan,
                         n_photon_spheres=0, r_photon_outer=np.nan,
                         r_photon_inner=np.nan, echo_time=np.nan,
                         echo_time_err=np.nan, echo_time_over_2M=np.nan,
                         tag=tag, n_horizons=len(zeros),
                         r_plus=zeros[-1] if zeros else np.nan,
                         kappa_plus=kappas[-1] if kappas else np.nan,
                         T_Hawking=(kappas[-1] / (2.0 * np.pi)) if kappas else np.nan))
    for row in rows[:len(Ms)]:
        row["tag"] = "horizonless"
        row.update(n_horizons=0, r_plus=np.nan, kappa_plus=np.nan,
                   T_Hawking=np.nan)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()),
                            lineterminator="\r\n", extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    safe_path(DATA / "horizonless_scan.csv").write_text(
        buf.getvalue(), encoding="utf-8", newline="")
    summary = dict(M_ext_n6=M_ext,
                   note="echo diverges as f_min->0; T_H of the two-horizon cases for comparison",
                   rows=[{k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                          for k, v in r.items()} for r in rows])
    safe_path(DATA / "stage5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    hdr = (f"{'tag':12s} {'M':>10} {'f_min':>10} {'2M/R99':>8} {'n_ph':>4} "
           f"{'tau_echo/2M':>12} {'T_H':>10}")
    print(hdr)
    for r in rows:
        print(f"{r['tag']:12s} {r['M']:10.6f} {r['f_min']:10.3e} "
              f"{r['compactness_2M_over_R99'] if r['compactness_2M_over_R99'] == r['compactness_2M_over_R99'] else float('nan'):8.4f} "
              f"{r['n_photon_spheres']:4d} "
              f"{r['echo_time_over_2M'] if r['echo_time_over_2M'] == r['echo_time_over_2M'] else float('nan'):12.4f} "
              f"{r['T_Hawking'] if r['T_Hawking'] == r['T_Hawking'] else float('nan'):10.4g}")


if __name__ == "__main__":
    main()
