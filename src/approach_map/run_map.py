"""Stage 1b: approach map. Unified characterization of candidates and parameter sweeps.

Run: python src/approach_map/run_map.py [--output DIR]
Output: candidates.json / candidates.csv (summary), profiles/<name>.csv (profiles),
sweep_*.csv (sweeps), *.png (plots). All lengths in L0, G=c=1, source in units of 8pi.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import approach_map as am  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SUMMARY_KEYS = ["label", "n_horizons", "R_plus", "R_minus", "kappa_plus", "kappa_minus", "T_H_over_schw",
                "area_plus_over_schw", "b_c", "b_c_over_schw", "df_at_R3m", "df_at_R6m", "df_at_R10m",
                "dgRR_at_R3m", "dgRR_at_R10m",
                "K_center", "K_max", "x_K_max", "K_max_over_schw_at_R", "core_volume",
                "rho8pi_center", "w_r_center", "w_t_center", "min_rho8pi", "min_nec_r", "min_nec_t", "min_sec",
                "sec_negative_range", "NEC_violated", "WEC_violated", "SEC_violated", "DEC_violated"]


def write_csv(path, rows, keys):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def clean(d):
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}


def sweep(make, values, pname):
    rows = []
    for v in values:
        fam = make(v)
        try:
            s, _, _ = am.characterize(fam)
        except Exception as exc:  # keep failed cases, don't drop them
            s = dict(label=fam.label, error=repr(exc))
        s[pname] = float(v)
        rows.append(clean(s))
        print(f"  {fam.label}: horizons={s.get('n_horizons')} Kmax={s.get('K_max')}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(Path(__file__).resolve().parents[2] / "data" / "approach_map"))
    args = ap.parse_args()
    out = Path(args.output)
    (out / "profiles").mkdir(parents=True, exist_ok=True)

    # 1. first-round candidates at common m=1, ell=2/3 (as in v01), same central density
    m, ell = 1.0, 2 / 3
    fams = am.make_families(m=m, ell=ell)
    summaries, profiles = [], {}
    for fam in fams:
        s, x, T = am.characterize(fam)
        summaries.append(clean(s))
        profiles[fam.name] = (x, T)
        keys = ["x", "R", "f", "rho8pi", "pr8pi", "pt8pi", "nec_r", "nec_t", "sec", "K", "Ricci", "NEC8pi"]
        write_csv(out / "profiles" / f"{fam.name}.csv", [dict(zip(keys, row)) for row in zip(*[T[k] for k in keys])], keys)
        print(f"{fam.label}: horizons={s['n_horizons']} K_max={s['K_max']:.4g} NEC_viol={s['NEC_violated']} SEC_viol={s['SEC_violated']}")
    (out / "candidates.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    write_csv(out / "candidates.csv", summaries, ["name"] + SUMMARY_KEYS)

    # 2. sweeps over the regularization parameter: exterior invariance vs interior cost
    ell_grid = np.r_[np.geomspace(0.01, 0.5, 12), np.linspace(0.55, 0.769, 6)]
    print("sweep hayward ell")
    sw_h = sweep(lambda v: am.Family("hayward", f"Hayward ell={v:.4g}", am.F_hayward, dict(m=m, ell=v)), ell_grid, "ell")
    print("sweep bardeen g (same central density: g^3=2 m ell^2)")
    sw_b = sweep(lambda v: am.Family("bardeen", f"Bardeen g={v:.4g}", am.F_bardeen, dict(m=m, g=v)),
                 (2 * m * ell_grid**2) ** (1 / 3), "g")
    print("sweep dymnikova r* (same central density)")
    sw_d = sweep(lambda v: am.Family("dymnikova", f"Dymnikova r*={v:.4g}", am.F_dymnikova, dict(m=m, rstar=v)),
                 (2 * m * ell_grid**2) ** (1 / 3), "rstar")
    print("sweep simpson-visser a")
    a_grid = np.r_[np.geomspace(0.01, 1.0, 10), np.linspace(1.2, 1.95, 5), [2.0, 2.2, 3.0]]
    sw_sv = sweep(lambda v: am.Family("simpson_visser", f"Simpson-Visser a={v:.4g}", am.F_schwarzschild, dict(m=m),
                                      am.R_bounce, dict(a=v), x_min=0.0), a_grid, "a")
    print("sweep triple-root r_- (r_+=2, b2 fixed to 8pi rho_c=6.75 at each r_-)")
    sw_tr = sweep(lambda v: am.Family("triple_root", f"kappa_-=0 r_-={v:.4g}", am.F_triple_root,
                                      dict(m=m, r_minus=v, r_plus=2.0, b2=6.75 * v**3 * 2.0 / 3)),
                  [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5], "r_minus")
    write_csv(out / "sweep_triple_root_rminus.csv", sw_tr, ["r_minus"] + SUMMARY_KEYS + ["error"])
    print("sweep pocket amplitude (b=1/12) and b (amplitude=5)")
    sw_pa = sweep(lambda v: am.PocketFamily(am.pocket_model.Parameters(m=m, ell=ell, amplitude=v)),
                  [0.0, 0.5, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0], "amplitude")
    sw_pb = sweep(lambda v: am.PocketFamily(am.pocket_model.Parameters(m=m, ell=ell, b=v)),
                  [1 / 24, 1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4, 1 / 3], "b")
    for name, rows, pname in (("hayward_ell", sw_h, "ell"), ("bardeen_g", sw_b, "g"), ("dymnikova_rstar", sw_d, "rstar"),
                              ("simpson_visser_a", sw_sv, "a"), ("pocket_amplitude", sw_pa, "amplitude"), ("pocket_b", sw_pb, "b")):
        write_csv(out / f"sweep_{name}.csv", rows, [pname] + SUMMARY_KEYS + ["error"])

    make_plots(out, fams, profiles, summaries, dict(hayward=sw_h, bardeen=sw_b, dymnikova=sw_d, simpson_visser=sw_sv))
    print("done ->", out)


def make_plots(out, fams, profiles, summaries, sweeps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(schwarzschild="k", hayward="C0", bardeen="C1", bardeen_samerho="C5", dymnikova="C2",
                  simpson_visser="C3", pocket="C4", triple_root="C6")
    labels = {f.name: f.label for f in fams}

    # A. profiles vs areal radius R: f, 8pi rho, SEC combination, K
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for name, (x, T) in profiles.items():
        R = np.asarray(T["R"], float)
        c = colors[name]
        axes[0, 0].plot(R, T["f"], c, label=labels[name])
        axes[0, 1].plot(R, T["rho8pi"], c)
        axes[1, 0].plot(R, T["sec"], c)
        axes[1, 1].plot(R, T["K"], c)
    axes[0, 0].set(ylabel="f", xlim=(0, 4), ylim=(-1.5, 1.05), title="Metric function f (zeros are Killing horizons)")
    axes[0, 0].axhline(0, color="gray", lw=0.5)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(ylabel="8π ρ", xlim=(0, 3), ylim=(-300, 30), title="Effective density (eigenvalue)")
    axes[0, 1].axhline(0, color="gray", lw=0.5)
    axes[1, 0].set(xlabel="R / L0", ylabel="8π(ρ + p_r + 2p_⊥)", xlim=(0, 3), ylim=(-100, 60),
                   title="\"rho+3p\": sign sets attraction/repulsion (SEC)")
    axes[1, 0].axhline(0, color="gray", lw=0.5)
    axes[1, 1].set(xlabel="R / L0", ylabel="K L0⁴", yscale="log", xlim=(0, 3), title="Kretschmann invariant")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle("Candidates at m=1, ell=2/3 (Hayward, Dymnikova, Bardeen-samerho: 8πρ_c=3/ell²); pocket v01 (α=5, b=1/12)")
    fig.tight_layout()
    fig.savefig(out / "candidates_profiles.png", dpi=130)
    plt.close(fig)

    # B. exterior deviations from Schwarzschild at the same mass
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (x, T) in profiles.items():
        if name == "schwarzschild":
            continue
        R = np.asarray(T["R"], float)
        mask = R > 1.5
        ax.plot(R[mask], np.abs(np.asarray(T["f"])[mask] - (1 - 2 / R[mask])), colors[name], label=labels[name])
    ax.set(xlabel="R / m", ylabel="|f − f_Schwarzschild|", yscale="log", xlim=(1.5, 12), title="Exterior deviation of the metric at the same mass m")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "exterior_deviation.png", dpi=130)
    plt.close(fig)

    # C. sweeps: exterior invariance (shadow, T_H) vs interior cost (K_max, kappa_-)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for name, rows in sweeps.items():
        rows = [r for r in rows if "error" not in r]
        if not rows:
            continue
        pname = {"hayward": "ell", "bardeen": "g", "dymnikova": "rstar", "simpson_visser": "a"}[name]
        # common axis: effective ell (same central density) or a for SV
        if name in ("bardeen", "dymnikova"):
            p = np.array([np.sqrt(r[pname] ** 3 / 2) for r in rows])
        else:
            p = np.array([r[pname] for r in rows])
        c = colors[name]
        g = lambda k: np.array([r.get(k) if r.get(k) is not None else np.nan for r in rows], float)
        axes[0, 0].plot(p, np.abs(g("b_c_over_schw") - 1), c + "o-", label=name)
        axes[0, 1].plot(p, np.abs(g("T_H_over_schw") - 1), c + "o-")
        axes[1, 0].plot(p, g("K_max"), c + "o-")
        axes[1, 1].plot(p, np.abs(g("kappa_minus")), c + "o-")
    axes[0, 0].set(xscale="log", yscale="log", ylabel="|b_c/b_c,Schw − 1|", title="Shadow radius shift")
    axes[0, 0].axhspan(0.05, 1, color="red", alpha=0.08, label="~5%: order of EHT precision (see SOURCES)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(xscale="log", yscale="log", ylabel="|T_H/T_H,Schw − 1|", title="Hawking temperature shift")
    axes[1, 0].set(xscale="log", yscale="log", xlabel="ell_eff / m  (a / m for Simpson-Visser)", ylabel="K_max L0⁴", title="Maximum curvature")
    axes[1, 1].set(xscale="log", yscale="log", xlabel="ell_eff / m  (a / m for Simpson-Visser)", ylabel="|κ_−| L0", title="Surface gravity of the inner horizon")
    for ax in axes.flat:
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Sweep over the regularization parameter: exterior invariance vs interior cost (m=1)")
    fig.tight_layout()
    fig.savefig(out / "sweep_exterior_vs_interior.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
