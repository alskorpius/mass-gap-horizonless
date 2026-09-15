"""Preprint: figure generation from experiment artifacts (no new calculations,
except regenerating the waveform for fig. 2 — the same code as echo_towers)."""

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ROOT = REPO / "data"
SRC = REPO / "src"


def fig_qnm():
    d = json.load(open(ROOT / "stage_E1_qnm_echo/timedomain_qnm_results.json",
                       encoding="utf-8"))
    ell = np.array([r["ell"] for r in d["hayward_small_cores"]])
    dRe = np.abs([r["d_Re"] for r in d["hayward_small_cores"]])
    # audit by the companion paper (matrix pencil): dRe/Re = +8.8e-4 at l/M=0.1,
    # law dRe/Re = +0.087 (l/M)^2; old FFT points — superseded
    corr = json.load(open(ROOT / "corrections_e1_e5/corrections_results.json",
                          encoding="utf-8"))
    dRe_aud = corr["e1a"]["accepted_from_companion_audit"]["dRe"]
    ell_aud = np.array([0.1, 0.03, 0.01])
    law = 0.37367 * 0.087 * ell_aud**2
    fig, ax = plt.subplots(figsize=(5.0, 3.6), layout="constrained")
    ax.loglog(ell, dRe, "o", mfc="none", color="#999999",
              label=r"FFT-peak (superseded: bias $>$ signal)")
    ax.loglog(ell_aud, law, ":", color="#555555",
              label=r"$+0.087\,(\ell/M)^2$ (audited)")
    ax.loglog([0.1], [dRe_aud], "o", color="#3265aa", ms=8,
              label=r"matrix pencil, $\Delta\mathrm{Re}\,\omega_M$")
    ax.set_xlabel(r"$\ell/M$")
    ax.set_ylabel(r"$|\Delta\omega_M|$ (fundamental, $l=2$ axial)")
    ax.legend(fontsize=8)
    fig.savefig(HERE / "fig_qnm_shifts.pdf")
    fig.savefig(HERE / "fig_qnm_shifts.png", dpi=160)
    plt.close(fig)


def fig_echo():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "et", SRC / "stage_E1_qnm_echo/echo_towers.py")
    et = importlib.util.module_from_spec(spec)
    sys.modules["et"] = et
    spec.loader.exec_module(et)
    M = et.M_EXT * 0.95
    rs, V = et.build_star_grid(M)
    sig = et.evolve(rs, V, t_max=90.0)
    t, y = sig[:, 0], sig[:, 1]
    from scipy.ndimage import gaussian_filter1d
    tau = json.load(open(ROOT / "stage_E1_qnm_echo/echo_towers_results.json",
                         encoding="utf-8"))["M/Mext=0.95"]["tau_geo"]
    dt = float(np.mean(np.diff(t)))
    env = gaussian_filter1d(np.abs(y), sigma=0.15 * tau / dt)
    i0 = int(np.argmax(env[t < tau]))
    t0 = t[i0]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    ax.plot(t, y, lw=0.6, color="#3265aa", alpha=0.8)
    ax.plot(t, env, color="#d97818", lw=1.6, label="envelope")
    for k in range(1, 4):
        ax.axvline(t0 + k * tau, color="#777777", ls=":", lw=0.9)
    ax.set_xlabel(r"$t$ (units of $\varepsilon_c=1$)")
    ax.set_ylabel(r"$\Psi$ at observer")
    ax.set_title(r"Echo train, $M/M_{\rm ext}=0.95$; "
                 r"$\tau_{\rm echo}^{\rm geom}=%.3f$" % tau, fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(HERE / "fig_echo_train.pdf")
    fig.savefig(HERE / "fig_echo_train.png", dpi=160)
    plt.close(fig)


def fig_stability():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "s3", SRC / "stage_E3_shell_stability/shell_stability.py")
    s3 = importlib.util.module_from_spec(spec)
    sys.modules["s3"] = s3
    spec.loader.exec_module(s3)
    M = s3.M_I + 0.5
    kappa = 0.5
    roots = s3.equilibrium_points(M, kappa, s3.A_int, 2 * M + 0.05, 4 * M)
    R0 = roots[0]
    mu0 = s3.mu_static(R0, M, s3.A_int)
    Rs = R0 + np.linspace(-0.6, 0.6, 800)
    mus = mu0 * (Rs / R0) ** (-2 * kappa)
    V = np.array([s3.V_eff(r, M, m, s3.A_int) for r, m in zip(Rs, mus)])
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3), layout="constrained")
    axs[0].plot(Rs, V, color="#3265aa", lw=1.8)
    axs[0].axhline(0, color="#333333", lw=0.7)
    axs[0].axvline(R0, color="#d97818", ls=":", lw=1.4)
    axs[0].set_xlabel(r"$R$")
    axs[0].set_ylabel(r"$V_{\rm eff}(R)$")
    axs[0].set_title(r"Equilibrium $R_*= %.2f$ is a maximum ($\kappa=0.5$)"
                     % R0, fontsize=9)
    d = json.load(open(ROOT / "stage_E3_shell_stability/shell_stability_results.json",
                       encoding="utf-8"))
    kaps, vpps = [], []
    for k, r in d["stall_family"].items():
        kap = float(k.split("kappa=")[1])
        if r and kap > 0:
            kaps.append(kap)
            vpps.append(r[0]["v_double_prime"])
    o = np.argsort(kaps)
    axs[1].plot(np.array(kaps)[o], np.array(vpps)[o], "o", color="#a92648")
    M2 = [r[0]["v_double_prime"] for k, r in d["stall_family"].items()
          if r and k.startswith("M=1.8")]
    axs[1].plot(np.array(kaps)[o][:3], M2, "s", mfc="none", color="#3265aa")
    axs[1].axhline(0, color="#333333", lw=0.7)
    axs[1].set_xlabel(r"$\kappa = dP/d\sigma$")
    axs[1].set_ylabel(r"$v''(R_*)$")
    axs[1].set_title(r"All equilibria unstable (circles: $M=1.34$; boxes: $M=1.84$)",
                     fontsize=9)
    fig.savefig(HERE / "fig_stability.pdf")
    fig.savefig(HERE / "fig_stability.png", dpi=160)
    plt.close(fig)


def fig_gamma_c():
    """Curve Gamma_c(R*): the causal stability window (nonlinear_shell_eos)."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "ne", SRC / "nonlinear_shell_eos/nonlinear_eos.py")
    ne = importlib.util.module_from_spec(spec)
    sys.modules["ne"] = ne
    spec.loader.exec_module(ne)
    M = ne.M_I + 0.5
    Rs, Gcs, Pis = [], [], []
    for R0 in np.linspace(2 * M + 0.1, 5.5, 60):
        r = ne.reduced_coefficients(R0, M)
        if r["Gamma_c"] is not None:
            Rs.append(R0)
            Gcs.append(r["Gamma_c"])
            Pis.append(r["Pi_eq"])
    fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
    ax.plot(Rs, Gcs, color="#3265aa", lw=2, label=r"$\Gamma_c(R_*)$")
    ax.plot(Rs, Pis, color="#d97818", lw=1.6, ls="--", label=r"$\Pi_{\rm eq}(R_*)$")
    ax.axhspan(0, 1, color="#77aa77", alpha=0.15, label=r"causal $0\leq\Gamma\leq1$")
    ax.axhline(0, color="#333333", lw=0.7)
    ax.set_xlabel(r"$R_*$")
    ax.set_ylabel(r"$dP/d\sigma$ at equilibrium")
    ax.set_title(r"Stability window: $\Gamma>\Gamma_c$ and $\Gamma\leq1$"
                 r" $\Rightarrow$ stable for $R_*\gtrsim3.5M$", fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(HERE / "fig_gamma_c.pdf")
    fig.savefig(HERE / "fig_gamma_c.png", dpi=160)
    plt.close(fig)


def fig_ergo():
    """Map of ergoregions (a, R*) + echo spin multiplet."""
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.4), layout="constrained")
    chis = np.linspace(0.01, 0.99, 60)
    Rfr = np.linspace(1.0, 2.2, 80)  # R*/2M
    Z = np.zeros((len(chis), len(Rfr)))
    for i, chi in enumerate(chis):
        a = chi
        r_plus = 1 + np.sqrt(1 - a**2)
        for j, rf in enumerate(Rfr):
            Rstar = 2 * rf
            thetas = np.linspace(0, np.pi, 91)
            Sig = Rstar**2 + a**2 * np.cos(thetas)**2
            gtt = -(1 - 2 * Rstar / Sig)
            Z[i, j] = 1 if np.max(gtt) > 0 else 0
    axs[0].imshow(Z, origin="lower", aspect="auto", extent=[1.0, 2.2, 0, 1],
                  cmap="RdYlGn_r", vmin=0, vmax=1, alpha=0.85)
    axs[0].axvline(1.0, color="k", lw=1)
    chis_fine = np.linspace(0.01, 0.99, 200)
    r_plus = 1 + np.sqrt(1 - chis_fine**2)
    axs[0].plot(r_plus / 2, chis_fine, color="k", lw=1.2)
    axs[0].set_xlabel(r"$R_*/2M$")
    axs[0].set_ylabel(r"$\chi=a/M$")
    axs[0].set_title("ergoregion (red) vs none (green); black: $r_+$", fontsize=9)
    # spin multiplet
    M_geo = 30 * 4.925e-6
    Rstar = 2.4
    for chi, col in ((0.7, "#3265aa"), (0.9, "#d97818")):
        a = chi
        Om = a / (Rstar**2 + a**2) / (2 * np.pi * M_geo)
        f0 = 1.0 / (18.7 * 2 * M_geo)
        ms = np.arange(-2, 3)
        axs[1].scatter(f0 + ms * Om, np.zeros_like(ms), marker="o", s=50,
                       color=col, label=fr"$\chi={chi}$")
    axs[1].set_xlabel("Hz")
    axs[1].set_yticks([])
    axs[1].set_title(r"Echo comb $m$-multiplet, $30\,M_\odot$, $R_*=1.2\cdot2M$",
                     fontsize=9)
    axs[1].legend(fontsize=8)
    fig.savefig(HERE / "fig_ergo_spin.pdf")
    fig.savefig(HERE / "fig_ergo_spin.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    HERE.mkdir(parents=True, exist_ok=True)
    fig_qnm()
    fig_echo()
    fig_stability()
    fig_gamma_c()
    fig_ergo()
    print("figures done")
