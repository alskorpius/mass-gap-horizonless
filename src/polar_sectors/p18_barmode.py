"""P18.B: bar mode as a source of ε for GW spin-down of the corner remnant.

Remnant: M = 2.7-2.9 Msun, R = 10.8 km, P0 = 1 ms, I = 0.35 M R^2.
(1) β = T/|W|: T = (1/2) I Ω^2; W from the Newtonian estimate with a GR
    correction: W ≈ -(3/5) G M^2/R x (GR factor ~0.6-0.9 for compactness
    M/R~0.18); we take a range.
(2) Thresholds: secular (CFS) bar mode β_s ~ 0.14 (uniform rotation), lower
    for differential rotation (~0.09-0.12); dynamical β_d ~ 0.27.
(3) r-modes: energy at saturation α~1e-4-1e-2.
(4) Retrospective: short signal (seconds-tens of seconds), h_rss
    against the O2 limit 2.1e-22 (short search, Abbott+2017).
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
    raise FileNotFoundError("project root not found")


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


C = 2.998e10
G = 6.674e-8
M_SUN = 1.989e33


def beta_TW(M_solar, R_km, P_ms, k_I=0.35, W_gr_factor=0.75):
    """β = T/|W|; W_Newt = (3/5)GM^2/R; GR factor 0.6-0.9, we take 0.75
    (Lai-Shapiro-type correction for compactness ~0.18)."""
    M_g = M_solar * M_SUN
    R_cm = R_km * 1e5
    I = k_I * M_g * R_cm**2
    Om = 2 * np.pi / (P_ms * 1e-3)
    T = 0.5 * I * Om**2
    W = (3.0 / 5.0) * G * M_g**2 / R_cm * W_gr_factor
    return T / abs(W), T, abs(W)


def main():
    out = {}

    # ---- (1) β for the corner remnant ----
    rows = []
    for M in (2.7, 2.8, 2.9):
        for P in (1.0, 1.3, 2.0):
            for wf in (0.6, 0.75, 0.9):
                b, T, W = beta_TW(M, 10.8, P, W_gr_factor=wf)
                rows.append(dict(M=M, P_ms=P, W_gr_factor=wf,
                                beta=float(b), T_erg=float(T), W_erg=float(W)))
    b_fast = [r for r in rows if r["P_ms"] == 1.0]
    print("β (P=1 ms):",
          {f"M={r['M']}": f"{r['beta']:.3f}" for r in b_fast})
    b_min = min(r["beta"] for r in b_fast)
    b_max = max(r["beta"] for r in b_fast)
    out["beta"] = dict(rows_P1=b_fast, range=[float(b_min), float(b_max)],
                        beta_secular=0.14, beta_dynamic=0.27,
                        beta_secular_diff_rot="(0.09-0.12)",
                        statement=f"β(P=1ms) = {b_min:.3f}-{b_max:.3f} "
                                  f"({'ABOVE' if b_min > 0.14 else 'BELOW'} "
                                  f"the secular threshold 0.14 at the minimum)")
    # β at the threshold frequency (β ∝ P^-2 => β=0.14 requires a SMALLER P):
    b_mid = b_fast[len(b_fast) // 2]["beta"]
    b_crit_P = 1.0 * np.sqrt(b_mid / 0.14)
    out["beta"]["P_at_beta014_ms"] = float(b_crit_P)
    print(f"    β=0.14 requires P ≈ {b_crit_P:.2f} ms (faster than 1 ms)")

    # ---- (2) bar mode: energy and time ----
    E_rot = 4.5e52
    M_g = 2.8 * M_SUN
    R_cm = 10.8e5
    I = 0.35 * M_g * R_cm**2
    # secular bar mode: grows over tau ~ 100-1000 periods (CFS
    # via viscosity); radiates until saturation at ε_sat ~ β-β_crit
    b0, _, _ = beta_TW(2.8, 10.8, 1.0, W_gr_factor=0.75)
    eps_sat_bar = max(b0 - 0.14, 0.0)  # ε at saturation ~ excess over β
    Om0 = 2 * np.pi / 1e-3
    # growth time of the secular instability: τ ~ (τ_visc)(Ωτ_visc)^{...}; parametrically
    # the literature gives minutes to hours; we take a range
    for t_growth_s, tag in [(1e2, "minutes"), (1e3, "hundreds of s"),
                            (1e4, "hours")]:
        # radiated energy up to saturation: E ~ E_rot x (excess β)/β
        frac_bar = eps_sat_bar / b0 if b0 > 0 else 0
        E_bar = E_rot * frac_bar
        # L_GW at ε_sat_bar
        L = (32.0 / 5.0) * G / C**5 * I**2 * (eps_sat_bar)**2 * Om0**6
        t_rad = E_bar / L if L > 0 else np.inf
        print(f"    growth {tag}: ε_sat={eps_sat_bar:.3f}, E_bar={E_bar:.2e} "
              f"erg, t_rad={t_rad:.1e} s, L={L:.2e} erg/s")
        out.setdefault("barmode", []).append(
            dict(growth_time_s=t_growth_s, tag=tag,
                 eps_sat=float(eps_sat_bar), E_erg=float(E_bar),
                 L_GW=float(L), t_radiate_s=float(t_rad)))
    out["barmode_statement"] = (
        f"β(P=1ms)={b0:.3f} {'>' if b0 > 0.14 else '<'} 0.14: the bar mode is "
        f"{'SECULARLY UNSTABLE' if b0 > 0.14 else 'absent'} "
        f"for the nominal profile; ε_sat~{eps_sat_bar:.3f}")

    # ---- (3) r-modes ----
    rows_r = []
    for alpha in (1e-4, 1e-3, 1e-2):
        # E_rmode ~ alpha^2 x Ω^2 I x (calibration 1e-3): standard
        # E ~ 1e-6 alpha^2 M R^2 Ω^2 ... we use Lai-Shapiro:
        E_rm = 1.1e-6 * alpha**2 * M_g * R_cm**2 * Om0**2  # erg
        rows_r.append(dict(alpha=alpha, E_erg=float(E_rm)))
        print(f"    r-mode α={alpha:.0e}: E={E_rm:.2e} erg")
    out["rmodes"] = rows_r
    E_rm_max = rows_r[-1]["E_erg"]
    rm_orders = float(np.log10(E_rot / E_rm_max))
    out["rmodes_statement"] = (
        f"r-modes at typical saturation α~1e-4-1e-2 give "
        f"<={E_rm_max:.2e} erg -- ~{rm_orders:.0f} orders of magnitude below "
        f"E_rot -- INSUFFICIENT")

    # ---- (4) retrospective of the short signal ----
    d_cm = 40 * 3.086e24
    f_GW = 2 * Om0 / (2 * np.pi)
    def h_rss_from_E(E, f):
        return np.sqrt(E * G / (np.pi**2 * C**3 * d_cm**2 * f**2))
    h_rss_lim = 2.1e-22
    rows4 = []
    for tag, E in [("bar mode (excess β)", E_rot * eps_sat_bar / b0),
                   ("full GW spin-down", E_rot),
                   ("r-mode α=1e-2", 1.1e-6 * 1e-4 * M_g * R_cm**2 * Om0**2)]:
        h = h_rss_from_E(E, f_GW)
        rows4.append(dict(source=tag, E_erg=float(E),
                          h_rss=float(h), ratio_to_limit=float(h / h_rss_lim)))
        print(f"    {tag}: h_rss={h:.2e} = {h/h_rss_lim:.2f}x the limit")
    out["retro_short"] = dict(rows=rows4, limit=h_rss_lim,
                              f_GW_Hz=float(f_GW))

    # ---- summary table of channels ----
    out["channel_table"] = [
        dict(channel="magnetic ε (dipole)", energy="~0",
             time="--", status="ε_mag=6e-10 ≪ 4e-3 -- insufficient"),
        dict(channel="bar mode (secular)",
             energy=f"{E_rot*eps_sat_bar/b0:.1e} erg" if b0 > 0.14 else "none",
             time="minutes-hours (growth)",
             status=("OPENS a GW channel without a toroid" if b0 > 0.14 else
                     f"β={b0:.3f}<0.14 -- NO INSTABILITY")),
        dict(channel="r-modes (α<=1e-2)",
             energy=f"<={E_rm_max:.2e} erg", time="--",
             status=f"~{rm_orders:.0f} orders of magnitude smaller -- insufficient"),
        dict(channel="toroid B>=6e18 G", energy="full E_rot",
             time="τ_GW~τ_sd", status="requires a field near the virial limit"),
    ]

    out["verdict"] = (
        f"β = T/|W| for the corner remnant at P=1 ms: {b_min:.3f}-{b_max:.3f} "
        f"(range of the GR factor W) -- THE WHOLE RANGE IS BELOW the secular "
        f"threshold 0.14 (the dynamical threshold 0.27 even more so); β=0.14 requires "
        f"P ≈ {b_crit_P:.2f} ms. The bar mode IS ABSENT for all "
        f"profiles considered. r-modes at saturation α<=1e-2 give "
        f"<=2.8e43 erg -- 9 orders of magnitude below E_rot -- INSUFFICIENT. "
        f"THE ONLY GW channel -- magnetic deformation from an internal "
        f"toroid >=6e18 G ≈ (1.5x) the virial limit. Bottom line: the corner branch "
        f"survives only for internal fields at the level of the virial "
        f"limit -- recorded as an exclusion condition in the abstract")
    safe_path(DATA / "p18_barmode.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print("verdict recorded")


if __name__ == "__main__":
    main()
