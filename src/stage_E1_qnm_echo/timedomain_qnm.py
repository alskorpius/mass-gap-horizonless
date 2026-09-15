"""E1-A (time domain): axial-sector QNMs from evolving the wave equation.

Method: d^2 Psi/dt^2 = d^2 Psi/dr*^2 - V(r*) Psi on a uniform r* grid;
leapfrog (2nd order); Sommerfeld boundaries; a Gaussian pulse as the initial
data; the ringing at the observer is decomposed with the matrix-pencil
(Prony) method.

Potential: V = f[(l(l+1) - 2 + 2f - r f')/r^2] (checked against the companion paper and the
RW limit). Geometries: Schwarzschild (validation: l=2 fundamental
0.37367-0.08896 i) and Hayward with small cores (shifts relative to
Schwarzschild -- where the companion paper's WKB is unreliable). The outer branch (r>r_+)
is used, where r* is monotone; the core enters only through the profile
f(r) on that branch.
"""

import json
from pathlib import Path

import numpy as np
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
    if name == "schwarzschild":
        f = lambda r: 1.0 - 2.0 * M / r
        fp = lambda r: 2.0 * M / r**2
        return f, fp, 2.0 * M
    if name == "hayward":
        c = 2.0 * M * ell**2
        f = lambda r: 1.0 - 2.0 * M * r**2 / (r**3 + c)
        fp = lambda r: -2.0 * M * (2.0 * r * (r**3 + c) - 3.0 * r**4) / (r**3 + c)**2
        roots = np.roots([1.0, -2.0 * M, 0.0, c])
        real = sorted(r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0)
        return f, fp, real[-1]
    raise ValueError(name)


def build_grid(name, M=1.0, ell=None, n=8001, window=240.0):
    f, fp, r_plus = make_metric(name, M, ell)
    # Table r*(r): grid logarithmic in the offset r-r_+ (the throat is
    # traversed uniformly in r*: with r-r_+=r_+e^u we have dr/f ~ du/kappa).
    u = np.linspace(np.log(1e-10), np.log(4000.0 * M - r_plus), 1500000)
    r_tab = r_plus + np.exp(u)
    inv_f = 1.0 / f(r_tab)
    rs_tab = np.concatenate([[0.0], np.cumsum(0.5 * (inv_f[1:] + inv_f[:-1])
                                              * np.diff(r_tab))])
    rs = np.linspace(0.5, window, n)
    r_of_rs = np.interp(rs, rs_tab, r_tab)
    V = f(r_of_rs) * (6 - 2 + 2 * f(r_of_rs) - r_of_rs * fp(r_of_rs)) / r_of_rs**2
    return rs, V


def evolve(rs, V, t_max=380.0, r0=70.0, sigma=2.5, obs=85.0, cfl=0.45):
    drs = rs[1] - rs[0]
    dt = cfl * drs
    steps = int(t_max / dt)
    Psi = np.exp(-((rs - r0) ** 2) / (2 * sigma**2))
    Pi = -np.gradient(Psi, rs)          # inject a left-moving pulse
    Psi_old = Psi - dt * Pi             # first step (forward Euler in time)
    i_obs = int(np.argmin(np.abs(rs - obs)))
    sig = [(0.0, Psi[i_obs])]
    for n in range(1, steps):
        lap = np.zeros_like(Psi)
        lap[1:-1] = (Psi[2:] - 2 * Psi[1:-1] + Psi[:-2]) / drs**2
        Psi_new = (2 * Psi - Psi_old + dt**2 * (lap - V * Psi))
        # Sommerfeld along the characteristics: on the left a left-moving wave
        # leaves (dPsi/dt = +dPsi/dr*), on the right a right-moving one (= -dPsi/dr*).
        Psi_new[0] = Psi[0] + dt / drs * (Psi[1] - Psi[0])
        Psi_new[-1] = Psi[-1] - dt / drs * (Psi[-1] - Psi[-2])
        Psi_old, Psi = Psi, Psi_new
        t = (n + 1) * dt
        sig.append((t, Psi[i_obs]))
    return np.array(sig)


def matrix_pencil(dt, y, L=8, keep=2):
    """Dominant exponentials y_n ~ sum a_k z_k^n (matrix pencil)."""
    y = np.asarray(y, dtype=complex)
    N = len(y)
    H1 = np.array([[y[i + j] for j in range(L)] for i in range(N - L - 1)])
    H2 = np.array([[y[i + 1 + j] for j in range(L)] for i in range(N - L - 1)])
    _, _, Vh = np.linalg.svd(H1)
    Vk = Vh[:keep].conj().T
    M = np.linalg.pinv(H1 @ Vk) @ (H2 @ Vk)
    z = np.linalg.eigvals(M)
    Vand = np.vstack([z**k for k in range(N)])
    a, *_ = np.linalg.lstsq(Vand, y, rcond=None)
    return list(zip(z, a))


def extract_mode(times, values, env_window=(60.0, 105.0), fft_window=(90.0, 280.0)):
    """Two transparent estimators (after matrix-pencil failures -- noted):
    Re omega -- the quadratically interpolated FFT peak of the fft_window
    (no mean subtraction: the late tail shifts the DC component);
    Im omega -- the slope of the log-envelope over the early env_window,
    where the fundamental mode dominates (the tail is still small)."""
    t, y = np.asarray(times), np.asarray(values)
    i1, i2 = np.searchsorted(t, fft_window[0]), np.searchsorted(t, fft_window[1])
    seg, tt = y[i1:i2], t[i1:i2]
    w = np.hanning(len(seg))
    F = np.abs(np.fft.rfft(seg * w))
    fr = np.fft.rfftfreq(len(seg), d=float(np.mean(np.diff(tt))))
    k = int(np.argmax(F))
    fpk = fr[k]
    if 0 < k < len(F) - 1:
        den = F[k - 1] - 2 * F[k] + F[k + 1]
        if den != 0:
            fpk = fr[k] + 0.5 * (F[k - 1] - F[k + 1]) / den * (fr[1] - fr[0])
    omega_re = 2 * np.pi * fpk
    j1, j2 = np.searchsorted(t, env_window[0]), np.searchsorted(t, env_window[1])
    env = np.abs(y[j1:j2])
    omega_im = float(np.polyfit(t[j1:j2], np.log(env + 1e-300), 1)[0])
    return complex(omega_re, omega_im), None


def main():
    out = {}
    rs, V = build_grid("schwarzschild")
    sig = evolve(rs, V)
    t, y = sig[:, 0], sig[:, 1]
    om, all_modes = extract_mode(t, y)
    known = 0.37367 - 0.08896j
    out["validation_schwarzschild_l2"] = dict(
        omega=dict(re=om.real, im=om.imag), known=dict(re=known.real, im=known.imag),
        rel_error=float(abs(om - known) / abs(known)))
    print("validation:", om, "rel err:", abs(om - known) / abs(known))
    hay = []
    prev = om
    for ell in (0.1, 0.03, 0.01):
        rs, V = build_grid("hayward", ell=ell)
        sig = evolve(rs, V)
        t, y = sig[:, 0], sig[:, 1]
        om_h, _ = extract_mode(t, y)
        hay.append(dict(ell=ell, omega=dict(re=om_h.real, im=om_h.imag),
                        d_Re=float(om_h.real - om.real),
                        d_Im=float(om_h.imag - om.imag)))
        print("hayward ell=", ell, "->", om_h)
    out["hayward_small_cores"] = hay
    safe_path(DATA / "timedomain_qnm_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
