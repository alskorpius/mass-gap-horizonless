"""Parallel mini-stage P2: gravitational-wave echo bands of the
horizonless branch (n=6, stage 3-5 family) for realistic masses.

Question (user): how do our results look in the context of gravitational
waves -- are echoes observable, for which masses, and in which bands.

Method: echo time tau = 2 Int_0^{r_ph,outer} dr / f(r) (geometric-optics
estimate, as in stage 5) for models M/M_ext in {0.88, 0.95, 0.995, 0.9999};
conversion to seconds: tau_phys = tau_geo/(M_geo) * (G M / c^3) -- in
geometric units of mass M, length = M, time = M, hence
tau_phys = (tau/(2M)) * 2GM/c^3. Masses: 1.4, 10, 30, 60, 500, 4.3e6,
6.5e9 M_sun. Bands: LIGO 10-5e3 Hz, LISA 1e-4-0.1 Hz.

Honest caveats: geometric optics, not the full QNM spectrum (stage 5);
echo amplitude not estimated (depends on the core's reflection mechanism --
the microphysics is not built); branch stability is not proven.
"""

import csv
import io
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma as gamma_fn, gammainc
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
M_EXT = 0.23136165195307468  # stage 4, n=6
G_OVER_C3_PER_MSUN = 4.9254909e-6  # GM_sun/c^3, s


def rc_for_mass(M):
    return (M * N / (4.0 * np.pi * EPS_C * gamma_fn(3.0 / N))) ** (1.0 / 3.0)


def m_of_r(r, rc, M):
    return M * gammainc(3.0 / N, (r / rc) ** N)


def f_of(r, rc, M):
    return 1.0 - 2.0 * m_of_r(r, rc, M) / r


def fprime(r, rc, M):
    eps = EPS_C * np.exp(-((r / rc) ** N))
    return -8.0 * np.pi * r * eps + 2.0 * m_of_r(r, rc, M) / r**2


def photon_spheres(rc, M):
    r = np.unique(np.r_[np.geomspace(1e-6, rc * 0.02, 200),
                        np.linspace(rc * 0.02, 12.0 * M, 12000)])

    def h(q):
        return float(q * fprime(np.array([q]), rc, M)[0]
                     - 2.0 * f_of(np.array([q]), rc, M)[0])
    hv = np.array([h(q) for q in r])
    roots = []
    for j in range(len(r) - 1):
        if hv[j] * hv[j + 1] < 0:
            roots.append(brentq(h, float(r[j]), float(r[j + 1]), xtol=1e-13))
    return roots


def tau_echo_over_2M(M_over_ext):
    M = M_EXT * M_over_ext
    rc = rc_for_mass(M)
    ps = photon_spheres(rc, M)
    if not ps:
        return None, None
    value, _ = quad(lambda q: 1.0 / float(f_of(np.array([q]), rc, M)[0]),
                    1e-9, ps[-1], epsabs=1e-10, epsrel=1e-10, limit=500)
    return 2.0 * value / (2.0 * M), ps


def band_of(nu_hz):
    if 10.0 <= nu_hz <= 5e3:
        return "LIGO/Virgo/KAGRA"
    if 1e-4 <= nu_hz < 0.1:
        return "LISA"
    if nu_hz < 1e-4:
        return "below LISA (PTA scales)"
    return "above LIGO"


def main():
    ratios = [0.88, 0.95, 0.995, 0.9999]
    masses_msun = [1.4, 10.0, 30.0, 60.0, 500.0, 4.3e6, 6.5e9]
    rows = []
    for ratio in ratios:
        tau_ratio, ps = tau_echo_over_2M(ratio)
        if tau_ratio is None:
            continue
        for m_msun in masses_msun:
            tau_s = tau_ratio * 2.0 * G_OVER_C3_PER_MSUN * m_msun
            nu_hz = 1.0 / tau_s
            rows.append(dict(M_over_Mext=ratio, tau_over_2M=tau_ratio,
                             M_msun=m_msun, tau_echo_s=tau_s,
                             nu_echo_hz=nu_hz, band=band_of(nu_hz)))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()),
                            lineterminator="\r\n", extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    safe_path(DATA / "echo_bands.csv").write_text(
        buf.getvalue(), encoding="utf-8", newline="")
    hdr = (f"{'M/Mext':>8} {'M [Msun]':>10} {'tau_echo':>12} {'nu_echo':>12} {'band':>24}")
    print(hdr)
    for r in rows:
        tau_str = (f"{r['tau_echo_s']:.3e} s")
        print(f"{r['M_over_Mext']:8.4f} {r['M_msun']:10.3g} {tau_str:>12} "
              f"{r['nu_echo_hz']:12.4g} {r['band']:>24}")
    summary = dict(
        note="geometric-optics estimate; amplitude and stability not assessed",
        tau_over_2M_by_ratio={str(rr): tau_echo_over_2M(rr)[0] for rr in ratios})
    safe_path(DATA / "echo_bands_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")


if __name__ == "__main__":
    main()
