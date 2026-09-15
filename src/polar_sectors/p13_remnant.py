"""P13: additional constraints on the surviving remnant of GW170817 .

Remnant model on the corner branch: M=2.7-2.9 Msun, R=10.8 km, I=0.35MR^2.
Dipole spin-down: L0 = B^2 R^6 Omega^4/(6c^3); tau = E_rot/L0;
L_sd(t) = L0/(1+t/tau)^2.

P13.1 Jet: spin-down channel luminosity L_sd at quiet B versus required jet
      energy (GRB 170817A: E_iso~3e46; structured jet:
      E_K,true ~ (0.3-1)e50 erg) and the proto-magnetar threshold B~1e15.
P13.2 Composition: Mdot_wind x t_irr versus the blue budget ~0.02 Msun.
P13.4 Prediction: L_sd(day 1200-1500) versus the Chandra excess
      ~ (3-6)e39 erg/s; B window from the excess and upper limits.
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
R_CM = 10.8e5
K_I = 0.35


def L0_erg_s(B_g, P_ms):
    Om = 2 * np.pi / (P_ms * 1e-3)
    return B_g**2 * R_CM**6 * Om**4 / (6 * C**3)


def Erot(M_solar, P_ms):
    I = K_I * M_solar * 1.989e33 * R_CM**2
    Om = 2 * np.pi / (P_ms * 1e-3)
    return 0.5 * I * Om**2


def Lsd(B_g, P_ms, M_solar, t_s):
    L0 = L0_erg_s(B_g, P_ms)
    tau = Erot(M_solar, P_ms) / L0
    return L0 / (1 + t_s / tau) ** 2


def main():
    out = {}
    M, P = 2.8, 1.0
    E = Erot(M, P)
    print(f"E_rot(M={M}, P={P} ms) = {E:.2e} erg")

    # ---- P13.1: jet ----
    E_iso = 3e46                     # GRB 170817A (from literature abstracts)
    theta = 0.15                     # half-angle ~8-10 deg
    E_true = E_iso * theta**2 / 2
    E_jet_true = 3e49                # structured jet: (0.3-1)e50 (conservative)
    for B in (1e10, 1e11, 1e12):
        L = L0_erg_s(B, P)
        E_17 = L * 1.7
        print(f"quiet B={B:.0e}: L_sd={L:.2e}, over 1.7 s => {E_17:.2e} erg "
              f"vs jet {E_jet_true:.0e} (deficit {E_jet_true/E_17:.0e})")
    # proto-magnetar launch: needs L_sd >= E_iso/duration ~ 1e46/1s
    B_jet = 1e15 * np.sqrt(1e46 / L0_erg_s(1e15, P) * 1.0)
    print(f"B for L_sd=1e46 (jet over ~1 s): {B_jet:.2e} G")
    out["p1_jet"] = dict(
        E_rot=E, E_iso=E_iso, E_true_beamed=E_true, E_jet_true=E_jet_true,
        quiet_deficit={f"B={b:.0e}": float(E_jet_true / (L0_erg_s(b, P) * 1.7))
                       for b in (1e10, 1e11, 1e12)},
        B_for_jet=B_jet,
        verdict="branch (a) 'quiet from birth' CONTRADICTS the jet (deficit "
                "≥4-6 orders of magnitude); survival is possible only on branch (b) "
                "of ultra-early spin-down B≳1e15 G (proto-magnetar "
                "jet launch, Metzger+2011-class)")

    # ---- branch (b): which B gives quiescence by day ~1000? ----
    day = 86400.0
    for B in (1e15, 3e15, 10**15.5, 1e16):
        L1000 = Lsd(B, P, M, 1000 * day)
        tau = E / L0_erg_s(B, P)
        print(f"B={B:.1e}: tau={tau/86400:.2f} d, L_sd(1000d)={L1000:.2e} erg/s")
    # threshold: L_sd(1000d) < 3e39
    from scipy.optimize import brentq
    f = lambda lb: np.log10(Lsd(10**lb, P, M, 1000 * day)) - np.log10(3e39)
    lb_q = brentq(f, 15, 17, xtol=1e-6)
    B_quiet1000 = 10**lb_q
    print(f"B_quiet(L_sd(1000d)<3e39) = {B_quiet1000:.2e} G")
    out["branch_b"] = dict(B_quiet_1000d=B_quiet1000,
                           statement="branch (b): B >= 2.4e15 G -- the jet "
                                     "launches (L0 ~ 9e49 erg/s, energy "
                                     "reserve for the jet 3e49 erg); by day "
                                     "1000 L_sd<3e39")

    # ---- P13.2: composition ----
    Mdot_lo, Mdot_hi = 1e-3, 1e-2         # Msun/s
    M_blue = 0.02                          # blue budget
    out["p2_composition"] = dict(
        Mdot_wind=(Mdot_lo, Mdot_hi), M_blue=M_blue,
        t_irr_max_s=(M_blue / Mdot_hi, M_blue / Mdot_lo),
        verdict="the neutrino wind of the surviving remnant must shut off within "
                "t ≲ 2-20 s (otherwise it overruns the blue budget and erodes "
                "the red lanthanide component); the remnant cools and the wind "
                "dies out at ~10-30 s -- marginal, NOT an exclusion, but a tight "
                "constraint on the cooling phase")

    # ---- P13.3: origin of the dipole ----
    out["p3_dipole"] = dict(
        merger_amplification="KH/MRI amplify the SMALL-SCALE field to "
                             "1e15-1e16 G within ms (Kiuchi+2018; Price-Rosswog "
                             "2006 -- from abstracts)",
        requirement="branch (b) needs a coherent DIPOLE of the same magnitude",
        status="open: simulations give small-scale amplification; whether "
               "the dipole reaches this level is not established; recorded as an assumption")

    # ---- P13.4: late-time excess ----
    d_cm = 40 * 3.086e24                # 40 Mpc
    area = 4 * np.pi * d_cm**2
    F_excess = 3e-14                    # erg/cm^2/s (Chandra, days ~1200-1500)
    L_excess = area * F_excess
    print(f"Chandra excess ~3e-14 => L ~ {L_excess:.2e} erg/s")
    # B window: L_sd(1200d) in [3e39, 6e39] and L_sd(1500d) < 6e39 (t^-2 decay slower than the afterglow)
    lo = brentq(lambda lb: np.log10(Lsd(10**lb, P, M, 1200 * day)) - np.log10(3e39), 15, 17)
    hi = brentq(lambda lb: np.log10(Lsd(10**lb, P, M, 1200 * day)) - np.log10(6e39), 15, 17)
    out["p4_prediction"] = dict(
        L_excess_obs=L_excess,
        B_window=(10**lo, 10**hi),
        statement="the corner remnant with B~(%.1f-%.1f)e15 G gives L_sd(1200d)="
                  "(3-6)e39 erg/s -- EXACTLY the band of the Chandra excess; "
                  "decay ~t^-2 (like the afterglow) but with a flat floor at "
                  "t>>tau; distinguished from the kilonova afterglow by "
                  "spectrum (harder, PWN-type) and no correlation with ejecta mass" % (10**lo / 1e15, 10**hi / 1e15),
        upper_limits="if late-time (day >1500) upper limits come in "
                     "<1e39 -- the window narrows from above; current limits "
                     "are not comparably deep (the excess is contested)")

    # summary
    out["verdict"] = (
        "P13.1: branch (a) quiet-from-birth CONTRADICTS the jet -- over 1.7 s "
        "2.6e39-2.6e43 erg is available (B=1e10-1e12) against the required "
        "~3e49 (structured jet) -- deficit 1e6-1e10; jet launch "
        "requires L_sd≳1e46 => B≳2.6e13 (proto-magnetar class of "
        "successful jets: B~1e15-1e16). Survival -- only branch (b) "
        "of ultra-early spin-down: B>=2.4e15 G (quiet by day 1000). "
        "P13.2: NOT an exclusion, a tight constraint -- the wind phase ≲2-20 s "
        "(blue budget 0.02 Msun against mdot 1e-3-1e-2 Msun/s; the red "
        "component is preserved). P13.3: dipole coherence at the level of "
        "1e15.5-1e16 after KH/MRI amplification (small-scale, up to 1e15-1e16) "
        "-- an OPEN assumption. P13.4: POSITIVE PREDICTION: a remnant "
        "with B=(1.4-2.0)e15 G gives L_sd(1200-1500 d)=(3-6)e39 erg/s -- "
        "the band of the late Chandra excess; B>=2.4e15 -- hidden (L<3e39 "
        "at all epochs). Range allowed by the late-time data: "
        "L_sd(1200d) <= 6e39 erg/s")

    safe_path(DATA / "p13_remnant.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(json.dumps(out["verdict"], ensure_ascii=False))


if __name__ == "__main__":
    main()
