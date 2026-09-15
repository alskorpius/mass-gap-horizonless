"""E1-B: echo towers of the horizonless branch -- time evolution of axial
perturbations on the stage-4-5 geometries (n=6, eps_c=1, M < M_ext).

Inner boundary -- a reflecting wall at the regular center
(V ~ l(l+1)/r^2 -> infinity): the cavity between the center and the outer
barrier (photon sphere ~3M) produces an echo train. Extracted quantities:
  - the echo period from the waveform (times between successive packets)
    and its comparison with the geometric-optics estimate tau_geo = 2 Int_0^{r_ph} dr/f
    (stage 5);
  - the comb of trapped modes: FFT spacing Delta_omega of the echo train
    against 2 pi / tau_geo;
  - amplitude ratios A_echo/A_prompt (input for E2 assuming unit
    reflectivity of the wall; the reflectivity parameter is handled there).

Axial potential V = f[(l(l+1) - 2 + 2f - r f')/r^2] (checked), l=2.
Leapfrog scheme, Sommerfeld on the right, Dirichlet on the left (the wall).
"""

import json
from pathlib import Path

import numpy as np
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
N_PROF = 6.0
M_EXT = 0.23136165195307468


def star_metric(M):
    rc = (M * N_PROF / (4.0 * np.pi * EPS_C * gamma_fn(3.0 / N_PROF))) ** (1.0 / 3.0)

    def f(r):
        return 1.0 - 2.0 * M * gammainc(0.5, (r / rc) ** N_PROF) / r

    def fp(r):
        # f' = -8 pi r eps + 2 m / r^2, eps = eps_c e^{-(r/rc)^n}
        m = M * gammainc(0.5, (r / rc) ** N_PROF)
        eps = EPS_C * np.exp(-((r / rc) ** N_PROF))
        return -8.0 * np.pi * r * eps + 2.0 * m / r**2

    return f, fp, rc


def build_star_grid(M, window=60.0, n=12001):
    f, fp, rc = star_metric(M)
    r_tab = np.geomspace(1e-5, 4000.0, 600000)
    inv_f = 1.0 / f(r_tab)
    rs_tab = np.concatenate([[0.0], np.cumsum(0.5 * (inv_f[1:] + inv_f[:-1])
                                              * np.diff(r_tab))])
    rs = np.linspace(0.2, window, n)
    r_of_rs = np.interp(rs, rs_tab, r_tab)
    V = f(r_of_rs) * (6 - 2 + 2 * f(r_of_rs) - r_of_rs * fp(r_of_rs)) / r_of_rs**2
    return rs, V


def evolve(rs, V, t_max=140.0, r0=12.0, sigma=1.5, obs=16.0, cfl=0.45):
    drs = rs[1] - rs[0]
    dt = cfl * drs
    steps = int(t_max / dt)
    Psi = np.exp(-((rs - r0) ** 2) / (2 * sigma**2))
    Pi = -np.gradient(Psi, rs)
    Psi_old = Psi - dt * Pi
    i_obs = int(np.argmin(np.abs(rs - obs)))
    sig = [(0.0, Psi[i_obs])]
    for k in range(1, steps):
        lap = np.zeros_like(Psi)
        lap[1:-1] = (Psi[2:] - 2 * Psi[1:-1] + Psi[:-2]) / drs**2
        Psi_new = 2 * Psi - Psi_old + dt**2 * (lap - V * Psi)
        Psi_new[0] = 0.0                     # wall at the center (Dirichlet)
        Psi_new[-1] = Psi[-1] - dt / drs * (Psi[-1] - Psi[-2])  # Sommerfeld
        Psi_old, Psi = Psi, Psi_new
        sig.append(((k + 1) * dt, Psi[i_obs]))
    return np.array(sig)


def analyze(M, t_max=140.0):
    rs, V = build_star_grid(M)
    sig = evolve(rs, V, t_max=t_max)
    t, y = sig[:, 0], sig[:, 1]
    # Geometric-optics estimate (also a prior for the autocorrelation):
    # 2 Int_0^{r_ph} dr/f, r_ph -- the outer root of r f' - 2 f = 0.
    f, fp, rc = star_metric(M)
    rr = np.geomspace(1e-6, 30.0, 200000)
    h = rr * fp(rr) - 2 * f(rr)
    r_ph = None
    for j in range(len(rr) - 1):
        if h[j] * h[j + 1] < 0 and rr[j] > 2 * M:
            r_ph = float(rr[j])
            break
    tau_geo = None
    if r_ph:
        from scipy.integrate import quad
        val, _ = quad(lambda q: 1.0 / float(f(np.array([q]))[0]), 1e-9, r_ph,
                      limit=500)
        tau_geo = 2.0 * val
    # Echo period: power autocorrelation plus a subharmonic correction:
    # if the peak sits at ~0.5*tau_geo, double it (within-packet beating).
    dt = float(np.mean(np.diff(t)))
    power = (y - np.mean(y)) ** 2
    ac = np.correlate(power, power, mode="full")[len(power) - 1:]
    ac /= ac[0]
    lags = np.arange(len(ac)) * dt
    tau_wave = None
    if tau_geo:
        m = (lags > 0.5 * tau_geo) & (lags < 2.0 * tau_geo)
        if np.any(m):
            j = int(np.argmax(ac[m]))
            tau_wave = float(lags[m][j])
            if abs(tau_wave - 0.5 * tau_geo) < 0.15 * tau_geo:
                tau_wave *= 2.0
    # Envelope and packet amplitudes in windows of k*tau_geo.
    from scipy.ndimage import gaussian_filter1d
    env = gaussian_filter1d(np.abs(y), sigma=0.15 * (tau_geo or 10.0) / dt)
    amps, times = [], []
    t_prompt = None
    if tau_geo:
        # first arrival: the first envelope maximum before tau_geo
        m0 = t < tau_geo
        if np.any(m0):
            j0 = int(np.argmax(env[m0]))
            t_prompt = float(t[j0])
            amps.append(float(env[j0]))
            times.append(t_prompt)
            for k in range(1, 5):
                lo, hi = t_prompt + k * tau_geo - 0.25 * tau_geo, \
                         t_prompt + k * tau_geo + 0.25 * tau_geo
                mk = (t >= lo) & (t <= hi)
                if np.any(mk):
                    jk = int(np.argmax(env[mk]))
                    amps.append(float(env[jk]))
                    times.append(float(t[np.where(mk)[0][jk]]))
    ratios = [float(a / amps[0]) for a in amps[1:]] if amps else []
    # Final period: median of the packet intervals (most reliable),
    # if there are >= 3 packets.
    gaps = np.diff([x for x in times])
    good_gaps = [g for g in gaps if abs(g - tau_geo) < 0.35 * tau_geo] if tau_geo else []
    tau_packets = float(np.median(good_gaps)) if len(good_gaps) >= 2 else None
    # FFT comb of the echo train (after the prompt).
    delta_omega = None
    if tau_geo and times:
        i1 = np.searchsorted(t, times[0] + 0.6 * tau_geo)
        seg = y[i1:]
        if len(seg) > 50:
            w = np.hanning(len(seg))
            F = np.abs(np.fft.rfft(seg * w))
            fr = np.fft.rfftfreq(len(seg), d=dt)
            pk = []
            for j in range(2, len(F) - 1):
                if F[j] > 0.1 * np.max(F) and F[j] >= F[j - 1] and F[j] > F[j + 1]:
                    pk.append(2 * np.pi * fr[j])
            pk = sorted(pk)
            deltas = np.diff(pk)
            delta_omega = float(np.median(deltas)) if len(deltas) >= 2 else None
    return dict(M=M, M_over_Mext=M / M_EXT, f_min=float(np.min(f(rr))),
                tau_echo_waveform=tau_wave, tau_geo=tau_geo,
                tau_packets=tau_packets,
                tau_ratio=((tau_packets or tau_wave) / tau_geo
                           if tau_geo and (tau_packets or tau_wave) else None),
                delta_omega_comb=(2 * np.pi / tau_packets if tau_packets else None),
                delta_omega_expected=(2 * np.pi / tau_geo if tau_geo else None),
                comb_spacing_rel=((2 * np.pi / tau_packets - 2 * np.pi / tau_geo)
                                  / (2 * np.pi / tau_geo)
                                  if (tau_packets and tau_geo) else None),
                packet_times=[float(x) for x in times],
                amplitude_ratios=ratios)


def main():
    out = {}
    for frac in (0.88, 0.95, 0.995):
        res = analyze(M_EXT * frac, t_max=160.0)
        out[f"M/Mext={frac}"] = res
        print(json.dumps(res, ensure_ascii=False, default=float))
    safe_path(DATA / "echo_towers_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")


if __name__ == "__main__":
    main()
