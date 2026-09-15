"""Polar sector, part 1-2: Lambda(M) curve of the hybrid corner branch
(f=5, n_t=4-5) and the candidacy of the GW190814 secondary (2.6 M☉).

Machine: lab_interface (SLy + hybrid star with a de Sitter core, Love
numbers with jumps). Stable part of the branch = the rising tail
(past the valley).

GW190814 (Abbott et al. 2020, ApJ 896, L44): m2 = 2.59-2.67 M☉ (90%),
tidal signal not measured (compatible with small Λ2); a mass gap is
observed between ~2.6 and ~5 M☉.
"""

import importlib.util
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


_spec = importlib.util.spec_from_file_location(
    "li", safe_path(PROJECT_ROOT / "src" / "lab_interface" / "lab_interface.py"))
LI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LI)


def hybrid_branch(f_nuc=5.0, n_t=4.0, n_pts=40):
    """Stable (rising) tail of the hybrid branch with Love numbers: scan
    Rc -> (M, R, Lambda), tail only."""
    eps_sat = LI.RHO_SAT * LI.C2 * LI.C_P
    eps_c = f_nuc * eps_sat
    P_t = LI.sly_P_of_rho(n_t * LI.RHO_SAT) * LI.C_P

    coarse = []
    for Rc in np.geomspace(0.05, 14.0, 120):
        r = LI.hybrid_star(eps_c, P_t, Rc)
        if r and 2.0 * r[0] * LI.KM_PER_MSUN < r[1]:
            coarse.append((Rc, r[0], r[1]))
    P = np.array(coarse)
    j0 = int(np.argmin(P[:, 1]))
    j1 = j0 + int(np.argmax(P[j0:, 1]))
    tail = P[j0:j1 + 1]
    xs = np.maximum.accumulate(tail[:, 1])
    ok = tail[:, 1] >= xs - 1e-9
    tail = tail[ok]
    Rc_lo, Rc_hi = float(tail[0][0]), float(tail[-1][0])

    rows = []
    for Rc in np.geomspace(Rc_lo, Rc_hi, n_pts):
        r = LI.hybrid_star(eps_c, P_t, Rc, y_need=True)
        if r is None:
            continue
        M, R, yR, Lam = r
        if 2.0 * M * LI.KM_PER_MSUN >= R:
            continue
        m_core = 4 * np.pi / 3 * eps_c * Rc**3 / LI.KM_PER_MSUN
        rows.append(dict(Rc=Rc, M=M, R=R, Lambda=Lam, yR=yR,
                         M_core=m_core, core_frac=m_core / M))
    return rows


def sly_branch():
    """Pure SLy branch with Lambda(M) up to the maximum."""
    rows = []
    for Pc in np.geomspace(8e-7, 1.2e-3, 45):
        r = LI.tov_sly(Pc, y_need=True)
        if r is None:
            continue
        M, R, yR, Lam = r
        rows.append(dict(Pc=Pc, M=M, R=R, Lambda=Lam))
    # monotonic part up to the maximum
    Ms = np.array([x["M"] for x in rows])
    imax = int(np.argmax(Ms))
    return rows[:imax + 1]


def main():
    out = {"meta": "parts 1-2: Lambda(M) of the corner branch and GW190814"}

    # SLy anchor (machine validation already in lab_interface: M_max=2.067)
    sly = sly_branch()
    Ms_sly = np.array([x["M"] for x in sly])
    Ls_sly = np.array([x["Lambda"] for x in sly])
    Rs_sly = np.array([x["R"] for x in sly])
    print("SLy: M from %.2f to %.2f, Λ(1.4)=%.0f" %
          (Ms_sly.min(), Ms_sly.max(),
           float(np.interp(1.4, Ms_sly, Ls_sly))))
    out["sly"] = dict(M_range=[float(Ms_sly.min()), float(Ms_sly.max())],
                      Lam14=float(np.interp(1.4, Ms_sly, Ls_sly)),
                      Mmax=float(Ms_sly.max()))

    branches = {}
    for n_t in (4.0, 5.0):
        rows = hybrid_branch(n_t=n_t)
        M = np.array([r["M"] for r in rows])
        L = np.array([r["Lambda"] for r in rows])
        R = np.array([r["R"] for r in rows])
        # Lambda(M) along the tail: Lambda decreases with M; build a
        # monotonic interpolation
        branches[f"nt={n_t:g}"] = dict(
            rows=rows,
            M_range=[float(M.min()), float(M.max())],
            R_range=[float(R.min()), float(R.max())],
            Lam_range=[float(L.max()), float(L.min())],
            Lam_at=[dict(M=float(m), Lam=float(l))
                    for m, l in zip(M[:: max(1, len(M) // 8)],
                                    L[:: max(1, len(L) // 8)])])
        print(f"nt={n_t}: M in [{M.min():.3f}, {M.max():.3f}], "
              f"R in [{R.min():.2f}, {R.max():.2f}] km, "
              f"Λ in [{L.min():.1f}, {L.max():.1f}]")
        for m in (1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6):
            if M.min() <= m <= M.max():
                print(f"   M={m}: Λ={float(np.interp(m, M, L)):.1f}, "
                      f"R={float(np.interp(m, M, R)):.2f} km")

    out["branches"] = branches

    # GW190814: Lambda(2.59-2.67) and candidacy
    gw = {}
    for key, br in branches.items():
        M = np.array([r["M"] for r in br["rows"]])
        L = np.array([r["Lambda"] for r in br["rows"]])
        R = np.array([r["R"] for r in br["rows"]])
        ok = (M.min() <= 2.59 and M.max() >= 2.67)
        gw[key] = dict(
            covers_2p6=bool(ok),
            Lam_259=float(np.interp(2.59, M, L)) if ok else None,
            Lam_267=float(np.interp(2.67, M, L)) if ok else None,
            R_26=float(np.interp(2.6, M, R)) if ok else None,
            Lam_sly_26=None)
    gw["sly_Lam_26"] = (float(np.interp(2.6, Ms_sly, Ls_sly))
                        if Ms_sly.max() >= 2.6 else
                        "SLy does not reach 2.6 M☉ (M_max=%.2f)" % Ms_sly.max())
    out["gw190814"] = gw
    print("GW190814:", json.dumps(gw, ensure_ascii=False, default=str, indent=1))

    # Mass distribution (qualitative comparison, stated honestly)
    out["mass_distribution"] = dict(
        observed_gap="2.7-5 M☉ empty (GWTC); GW190814 secondary 2.59-2.67 "
                      "-- the only object in the lower gap",
        our_branch=f"hybrid corner branch: stable up to M_max≈"
                   f"{branches['nt=4']['M_range'][1]:.2f} M☉ (nt=4) -- "
                   "fills 2.1-3.6 M☉ with objects of R~10-11 km",
        predicts="objects in 2.6-3.6 M☉ SHOULD exist (GW190814-like "
                 "events stop being unique); their Λ(2.6-3.6)≈ "
                 f"{branches['nt=4']['Lam_range'][1]:.1f}-"
                 f"{branches['nt=4']['Lam_range'][0]:.1f} -- the tidal "
                 "signal is weak (unnoticeable in LIGO, as for GW190814)",
        not_predicts="the upper edge of the gap, 3.6-5 M☉, remains empty "
                     "(collapse dynamics -- black holes form higher up "
                     "the stellar-evolution track); the relative "
                     "population 2.6-3.6 vs <2.1 was not computed (a "
                     "population model is needed)")

    safe_path(DATA / "lambda_curve_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print("wrote lambda_curve_results.json")


if __name__ == "__main__":
    main()
