"""P19: the GW170817 remnant as an object of the geometric (tower) branch.

Setup : if the horizonless threshold lies above the remnant
mass, the geometric branch predicts a quiet horizonless remnant with no
matter and no field of its own. Four checks:
(1) rotational energy: the only channel is GW; a stationary axisymmetric
    object does not radiate at all -> compatible with the kilonova/afterglow?
(2) jet: BZ requires a horizon/ergosphere/anchor; disk nu-nubar annihilation;
    magnetic tower. Estimate of the would-be BZ power for a disk field of 1e15 G.
(3) ergoregion instability: is there an ergosphere for 2.7 Msun, R*≈10.3 km
    (branch throat), spin from P=1 ms; counterfactual window r+<R*<2M.
(4) discriminator: tower -> pure t^-2; corner -> plateau (P13). Hajela+2022.

Source artifacts: stage4_summary.json (c_ext(n)), stage5_summary.json
(throat rc/M), ergoregion_results.json (criterion R*<2M), p17_gwbudget.json
(h_rss chain), Abbott+2017 (O2 limit).
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
KM = 1e5           # cm in a km
M_SUN_KM = 1.4766  # GM/c^2 in km per Msun
M_SUN_S = 4.9255e-6  # GM/c^3 in s per Msun

EPS_NUC_KM2 = 2.0e-4  # km^-2; consistent with the preprint: ε_c=5ε_nuc <-> ℓ0≈10.9 km
M_GRAV = 2.7          # Msun, gravitational mass of the remnant
P_MS = 1.0
K_I = 0.35            # reference moment of inertia I=k M R^2 (a limitation!)
AGE_YR = 9.1          # remnant age: Aug 2017 -> Sept 2026


def main():
    out = {"assumptions": {
        "M_grav_Msun": M_GRAV, "P_ms": P_MS, "k_I": K_I,
        "eps_nuc_km2": EPS_NUC_KM2,
        "I_note": "I=0.35MR^2 -- a reference value (fluid star); the moment of "
                  "inertia of the throat geometry was not computed -- it affects only "
                  "χ and E_rot, not the conclusions about the ergosphere",
    }}
    M_km = M_GRAV * M_SUN_KM
    M_g = M_GRAV * M_SUN
    Om = 2 * np.pi / (P_MS * 1e-3)

    # ---- (0) horizonless threshold for the remnant ----
    s4 = json.load(open(safe_path(
        PROJECT_ROOT / "data" / "stage4_extremal_core" / "stage4_summary.json"),
        encoding="utf-8"))
    c_ext = {int(float(k.split("=")[1])): s4[k]["c_ext"] for k in s4
             if k.startswith("n=")}
    l0 = lambda ec: np.sqrt(3.0 / (8.0 * np.pi * ec))  # noqa: E731
    thr_rows = []
    for n in sorted(c_ext):
        for ec_mult in (5.0, 29.0):
            ec = ec_mult * EPS_NUC_KM2
            mext_km = c_ext[n] / np.sqrt(ec)
            thr_rows.append(dict(n=n, eps_c_eps_nuc=ec_mult,
                                 M_ext_Msun=float(mext_km / M_SUN_KM),
                                 ell0_km=float(l0(ec))))
    # horizonless condition for 2.7 Msun: ε_c <= (c_ext/M_km)^2
    cond_rows = []
    for n in sorted(c_ext):
        ec_max = (c_ext[n] / M_km) ** 2
        cond_rows.append(dict(n=n, eps_c_max_eps_nuc=float(ec_max / EPS_NUC_KM2),
                              ell0_min_km=float(l0(ec_max))))
    out["threshold"] = dict(rows=thr_rows, condition_2p7=cond_rows,
                            reviewer_band="4.9-6.8 Msun at ℓ0≈5-11 km "
                                          "(our n=6->4.9 / n≈4->6.8 / "
                                          "n=2->8.4 at ε_c=5ε_nuc)",
                            statement=(
                                "2.7 Msun is horizonless for ε_c ≲ 17 ε_nuc "
                                "(n=6, ℓ0≳6 km) through 49 ε_nuc (n=2): most "
                                "of the allowed band 5-29 ε_nuc; at the "
                                "sharp edge (ε_c≳17, n=6) the remnant would have "
                                "been two-horizon"))
    print("thresholds:", [(r["n"], r["eps_c_eps_nuc"], round(r["M_ext_Msun"], 2))
                       for r in thr_rows])
    print("condition 2.7:", [(r["n"], round(r["eps_c_max_eps_nuc"], 1))
                            for r in cond_rows])

    # ---- (1) geometry: branch throat and ergosphere ----
    s5 = json.load(open(safe_path(
        PROJECT_ROOT / "data" / "stage5_horizonless_observables" / "stage5_summary.json"),
        encoding="utf-8"))
    mext6 = c_ext[6] / np.sqrt(5.0 * EPS_NUC_KM2)  # km, ε_c=5 ε_nuc, n=6
    # stage5 rows are in code units (ε_c=1): M/M_ext is dimensionless
    mm = np.array([r["M"] / s5["M_ext_n6"] for r in s5["rows"]])
    rcm = np.array([r["rc"] / r["M"] for r in s5["rows"]])
    frac = M_km / mext6
    rc_over_M = float(np.interp(frac, mm, rcm))
    R_star_km = rc_over_M * M_km
    R_star_cm = R_star_km * KM
    ratio = R_star_km / (2 * M_km)
    chi = K_I * M_g * R_star_cm**2 * Om * C / (G * M_g**2)
    E_rot = 0.5 * K_I * M_g * R_star_cm**2 * Om**2
    ergo = json.load(open(safe_path(
        PROJECT_ROOT / "data" / "ergoregion_rotation" / "ergoregion_results.json"),
        encoding="utf-8"))
    out["geometry"] = dict(
        eps_c_choice="5 ε_nuc (soft edge of the band)", M_over_Mext=float(frac),
        rc_over_M=rc_over_M, R_star_km=float(R_star_km),
        R_star_over_2M=float(ratio), chi=float(chi),
        E_rot_erg=float(E_rot),
        criterion_ergo=str(ergo.get("criterion", "R*<2M (shell/Kerr)")),
        ergosphere="NONE at any spin: smooth tower f(r)>0 everywhere "
                   "(no ergosurface by construction) AND R*/2M="
                   f"{ratio:.2f}>1 (criterion for a shell realization)")
    print(f"throat: R*={R_star_km:.1f} km = {ratio:.2f}*2M; χ={chi:.3f}; "
          f"E_rot={E_rot:.2e} erg")

    # ---- (2) jet ----
    r_g_cm = M_km * KM
    B0 = 1e15
    Phi = 2 * np.pi * r_g_cm**2 * B0
    kappa_bz = 0.05
    P_bz = kappa_bz * Phi**2 * Om**2 / (6 * np.pi * C)
    L_need = 3e49 / 1.7
    B_need = B0 * np.sqrt(L_need / P_bz)
    e_nunu = (1e-4, 1e-3)
    L_nu = 1e52
    e_jet_low = e_nunu[0] * L_nu * 1.0
    e_jet_high = e_nunu[1] * L_nu * 1.0
    out["jet"] = dict(
        BZ_wouldbe=dict(P_at_1e15_erg_s=float(P_bz), B_needed_G=float(B_need),
                        note="normalization P=(κ/6πc)Φ²Ω², κ=0.05, Φ=2π r_g²B; "
                             "Ω=2π/P; the ENERGETICS is sufficient for "
                             f"B_disk≳{B_need:.1e} G (within the MRI band "
                             "1e15-1e6 of Kiuchi+2018)"),
        mechanism="ABSENT for a purely geometric object: no "
                  "horizon (nothing to thread field lines onto), no ergosphere "
                  "(no negative-energy orbits), no material "
                  "surface (no anchor for the disk field)",
        nunu_annihilation=dict(E_erg=[float(e_jet_low), float(e_jet_high)],
                               note="η_νν~1e-4-1e-3 x L_ν~1e52 erg/s x 1 s "
                                    "(stated normalization, not the primary source); "
                                    "<3e49 and uncollimated"),
        magnetic_tower="an anchor is needed -- absent",
        shell_fork="in the shell realization (stage9-10) there is material "
                   "in a shell -> an anchor exists -> the channel opens; but this is no "
                   "longer 'without matter', and then the P17 budget applies",
        verdict="A BRANCH PROBLEM (not an exclusion): the energetics of every "
                "mechanism is <= what is needed; pure geometry has no anchor")
    print(f"jet: P_BZ(1e15)={P_bz:.2e} erg/s, B_need={B_need:.2e} G; "
          f"νν: {e_jet_low:.0e}-{e_jet_high:.0e} erg")

    # ---- (3) ergoregion instability ----
    M_geom_s = M_GRAV * M_SUN_S
    tau_max_s = 1e15 * M_geom_s      # Mω_I=1e-15
    tau_min_s = 1e6 * M_geom_s       # Mω_I=1e-6
    age_s = AGE_YR * 3.156e7
    mw_age = (1.0 / age_s) * M_geom_s           # Mω_I giving τ=age
    f_GW = 2000.0
    d_cm = 40 * 3.086e24
    h_rss = np.sqrt(E_rot * G / (np.pi**2 * C**3 * d_cm**2 * f_GW**2))
    out["ergoinstability"] = dict(
        applicable=False,
        reason="there is no ergosphere (f>0; R*/2M>1) -> the Friedman "
               "instability does not apply; trivially compatible with 9 years",
        counterfactual_window=dict(
            tau_range_s=[float(tau_min_s), float(tau_max_s)],
            tau_range_human="13 s - 422 years (Franzin+2022 frequencies "
                            "Mω_I∈[1e-15,1e-6], rescaled to 2.7 Msun)",
            age_yr=AGE_YR, straddles=bool(tau_min_s < age_s < tau_max_s),
            mw_I_at_age=float(mw_age),
            note="within the window the outcome is undetermined (some modes "
                 "would already have grown); a spin reset would give "
                 "h_rss=1.3e-23=0.06xO2 -- untestable"),
        h_rss_if_spindown=float(h_rss))
    print(f"ergo: not applicable; counterfactual τ∈[{tau_min_s:.0e}, "
          f"{tau_max_s:.0e}] s vs age {age_s:.1e} s")

    # ---- (4) combined prediction and discriminator ----
    out["discriminator"] = dict(
        tower_prediction="quiet remnant: the late-time X-ray excess = pure "
                         "kilonova afterglow t^-2, remnant contribution = 0; "
                         "the blue component is the neutrino wind of the HOT "
                         "DISK (~1e51 erg, seconds), not of the object",
        corner_prediction="plateau/excess after ~1000 days (P13, L_sd∼3-6e39) "
                          "-- now on the disfavored corner branch",
        hajela2022="the ~1200-1500 day excess is statistically marginal; "
                   "t^-2 falls within the error bars -> the data today are closer "
                   "to a pure afterglow (tower); persistence of the excess "
                   "would point to an engine (corner-like)",
        open_caveat="the Y_e microphysics of the wind without remnant neutrinos was not "
                    "modeled (disk only)")
    out["verdict"] = (
        "The geometric branch PREDICTS a quiet horizonless remnant for "
        "GW170817 (2.7 Msun < the threshold 4.9-8.4 Msun at ε_c=5 ε_nuc; condition "
        "ε_c≲17 ε_nuc for n=6). Checks: (1) COMPATIBLE -- E_rot="
        f"{E_rot:.1e} erg is locked in (a stationary axisymmetric object with no "
        "matter and no field does not radiate); the kilonova/afterglow require "
        "nothing of the remnant, the blue component is the disk ν-wind. (2) PROBLEM -- "
        f"the jet: would-be BZ {P_bz:.1e} erg/s at B=1e15 G is sufficient for "
        f"B≳{B_need:.1e} G, but pure geometry has no anchor; νν~1e48-1e49 "
        "<3e49 and does not collimate. (3) COMPATIBLE -- there is no ergosphere "
        "at any spin (f>0; R*/2M=1.29); the Friedman instability "
        "does not apply; in the counterfactual window τ=13s-422yr against 9 years -- "
        "the outcome is undetermined, the radiation is untestable (0.06xO2). (4) "
        "DISCRIMINATOR -- tower: pure t^-2; corner (disfavored): plateau; "
        "Hajela+2022 today is closer to t^-2. Bottom line: the branch passes checks 1, 3, 4 and "
        "carries one problem (the jet); the remnant's status -- 'quiet tower' -- "
        "is recorded in the preprint")
    safe_path(DATA / "p19_tower_remnant.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print("verdict recorded")


if __name__ == "__main__":
    main()
