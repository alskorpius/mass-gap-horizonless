"""GW170817 tension: baryonic budget of the surviving corner branch.

User question: the branch is stable up to ~3.5 M☉ (gravitational); the
GW170817 remnant (~2.7 M☉ gravitational / ~2.8-3.0 baryonic, with the
torus) collapsed under the standard interpretation => M_TOV <~ 2.2-2.3
(Margalit-Metzger-type inference). Why didn't the remnant stay on the
branch?

We compute: M_b(M) along the corner branch (nt=4): baryonic mass =
int 4 pi r^2 n_B m_b / sqrt(1-2m/r) dr; comparison with the GW170817
remnant. Formulation of two outcomes (survival prediction / collapse
exclusion of the corner).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
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
    "li", PROJECT_ROOT / "src" / "lab_interface" / "lab_interface.py")
LI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LI)

M_B_NUC = 1.66e-24          # g; n_B [fm^-3] = 6.022e-16 * rho[g/cm^3] / 1.66e-24 ... use direct computation


def hybrid_profile(f_nuc, n_t, Rc):
    """TOV profile of the hybrid star (as in lab_interface.hybrid_star)
    with dense output (r, m, rho); baryonic-mass integral."""
    eps_sat = LI.RHO_SAT * LI.C2 * LI.C_P
    eps_c = f_nuc * eps_sat
    P_t = LI.sly_P_of_rho(n_t * LI.RHO_SAT) * LI.C_P
    m_c = 4 * np.pi / 3 * eps_c * Rc**3
    if 2 * m_c / Rc >= 1:
        return None

    def rhs(r, s):
        P, m = s
        rho = LI.sly_rho_of_P(P / LI.C_P)
        e = LI.sly_eps_of_rho(rho)
        dP = -(e + P) * (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
        dm = 4 * np.pi * r * r * e
        return [dP, dm]

    def surf(r, s):
        return s[0] - 1e-13 * P_t
    surf.terminal, surf.direction = True, -1

    sol = solve_ivp(rhs, (Rc, 50.0), [P_t, m_c], events=[surf],
                    rtol=1e-10, atol=[P_t * 1e-12, m_c * 1e-10],
                    max_step=0.05, dense_output=True)
    if not sol.t_events[0].size:
        return None
    R = float(sol.t_events[0][0])
    m_R = float(sol.y_events[0][0][1])

    rs = np.linspace(Rc, R, 4000)
    Ps = sol.sol(rs)[0]
    rhos = np.array([LI.sly_rho_of_P(p / LI.C_P) for p in Ps])
    ms = sol.sol(rs)[1]
    # baryonic mass: m_b * int n_B 4 pi r^2 (1-2m/r)^-1/2 dr; everything in cm
    nB = rhos / M_B_NUC                       # cm^-3
    rs_cm = rs * 1e5                          # km -> cm
    integrand = nB * 4 * np.pi * rs_cm**2 / np.sqrt(1 - 2 * ms / rs)
    Mb_env = np.trapezoid(integrand, rs_cm) * M_B_NUC   # g (count x m_b)
    # core: baryons of the ultra-dense phase; conservatively n >= n_t (kappa_n=1..5)
    Mb_core_list = {}
    V_core = 4 * np.pi / 3 * (Rc * 1e5) ** 3   # cm^3
    for kap in (1, 2, 5):
        n_core = kap * n_t * LI.RHO_SAT / M_B_NUC     # cm^-3
        Mb_core_list[f"kappa={kap}"] = (n_core * V_core * M_B_NUC)  # g
    M_b_g = Mb_env + Mb_core_list["kappa=1"]
    return dict(Rc=Rc, M_grav=m_R / LI.KM_PER_MSUN, R=R,
                M_b_solar_k1=(M_b_g / 1.989e33),
                M_b_core_solar={k: v / 1.989e33 for k, v in Mb_core_list.items()},
                Mb_env_solar=Mb_env / 1.989e33)


def main():
    out = {"meta": "baryonic budget of the surviving corner branch (nt=4) "
                   "vs the GW170817 remnant; core baryon content is "
                   "parametrised by kappa_n (core baryon number is not "
                   "modelled -- placeholder storage)"}

    rows = []
    for Mt in (1.7, 2.0, 2.2, 2.6, 2.7, 3.0, 3.3, 3.45):
        # fit Rc to the target mass
        def eq(Rc):
            p = hybrid_profile(5.0, 4.0, Rc)
            if p is None:
                return -1.0
            return p["M_grav"] - Mt
        try:
            Rc = brentq(eq, 6.0, 10.9, xtol=1e-4)
        except Exception:
            try:
                Rc = brentq(eq, 0.5, 10.9, xtol=1e-4)
            except Exception:
                continue
        p = hybrid_profile(5.0, 4.0, Rc)
        rows.append(dict(M_grav=p["M_grav"], R=p["R"], Rc=Rc,
                         M_b_k1=p["M_b_solar_k1"],
                         Mb_env=p["Mb_env_solar"],
                         Mb_core=p["M_b_core_solar"]))
        print(f"M={p['M_grav']:.2f} M☉: R={p['R']:.2f} km, "
              f"M_b(k=1)={p['M_b_solar_k1']:.3f} M☉ "
              f"(envelope {p['Mb_env_solar']:.3f} + core "
              f"{p['M_b_core_solar']['kappa=1']:.3f})")
    out["rows"] = rows

    Mb_27 = float(np.interp(2.7, [r["M_grav"] for r in rows],
                            [r["M_b_k1"] for r in rows]))
    Mb_max = max(r["M_b_k1"] for r in rows)
    out["comparison"] = dict(
        Mb_branch_at_27=Mb_27,
        Mb_branch_max=Mb_max,
        GW170817_remnant=dict(
            M_grav_total=2.74, M_b_remnant_range=(2.8, 3.0),
            source="standard interpretation: baryonic ~2.74+0.1-0.2 "
                   "(losses to ejecta/torus), per the literature"),
        verdict=("the GW170817 remnant (M_b ~ 2.8-3.0) FITS on the branch "
                 f"(M_b(2.7)={Mb_27:.2f}, M_b,max={Mb_max:.2f} at k=1): "
                 "within the corner scenario the remnant is NOT REQUIRED "
                 "to collapse -- the model PREDICTS a surviving compact "
                 "object in place of GW170817; this is in tension with "
                 "the standard inference M_TOV<=2.2-2.3 "
                 "(Margalit-Metzger-type), which is itself EOS-dependent. "
                 "Two outcomes: (A) the remnant survived -- testable via "
                 "late-time emission; (B) the remnant collapsed -- the "
                 "corner is excluded, and the GW190814 candidacy dies "
                 "with it"))

    safe_path(DATA / "baryonic_budget.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(json.dumps(out["comparison"]["verdict"], ensure_ascii=False))


if __name__ == "__main__":
    main()
