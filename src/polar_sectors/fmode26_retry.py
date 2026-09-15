"""Retry of the f-mode search at 2.59-2.67 M☉ (per the user's remark):
different scan grids, different windows, different NM starting points;
report on all attempts. If the root is not localised, we record the
failure (no number is claimed)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

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
sys.path.insert(0, str(PROJECT_ROOT / "src" / "third_party"))
sys.path.insert(0, str(HERE))

from relmodpy.star import Star                    # noqa: E402
from relmodpy.mode import Mode                    # noqa: E402
from scipy.optimize import brentq, minimize       # noqa: E402
from fmode import HybridAnalogEOS, SLyEOS, mass_km, HZ_PER_KM  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location(
    "li", PROJECT_ROOT / "src" / "lab_interface" / "lab_interface.py")
LI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LI)


def main():
    eps_sat = LI.RHO_SAT * LI.C2 * LI.C_P
    eps_c = 5.0 * eps_sat
    P_t = LI.sly_P_of_rho(4.0 * LI.RHO_SAT) * LI.C_P

    out = {"meta": "repeated f-mode attempts at 2.59-2.67 M☉, different grids"}
    for G1 in (1e3, 1e4, 1e5):
        eos = HybridAnalogEOS(eps_c, P_t, Gamma1_core=G1)
        for msun in (2.59, 2.63, 2.67):
            try:
                pc = brentq(lambda p: Star(eos, p).M - mass_km(msun),
                            1e-6, 1e-2, xtol=1e-12)
            except Exception as exc:
                print(f"G1={G1:g}, M={msun}: TOV fit did not converge: {str(exc)[:50]}")
                continue
            s = Star(eos, pc)
            M = s.M
            mode = Mode(s)
            attempts = []
            # three scan grids + different NM starting points
            for (lo, hi, n) in [(0.05, 0.16, 18), (0.10, 0.24, 30), (0.14, 0.20, 25)]:
                wrs = np.linspace(lo / M, hi / M, n)
                best_w, best_v = None, np.inf
                for wr in wrs:
                    try:
                        v = abs(mode.spectrum(2, wr - 1e-6j))
                    except Exception:
                        continue
                    if v < best_v:
                        best_w, best_v = wr, v
                if best_w is None:
                    attempts.append(dict(window=(lo, hi), scan="all points failed"))
                    continue
                for x0 in ([best_w * M, 5e-5], [best_w * M, 2e-4],
                           [best_w * M, 1e-3]):
                    def abs_ain(x):
                        try:
                            return abs(mode.spectrum(2, (x[0] + 1j * x[1]) / M))
                        except Exception:
                            return 1e6
                    res = minimize(abs_ain, x0=x0, method="Nelder-Mead",
                                   options=dict(xatol=1e-6, fatol=1e-10,
                                                maxiter=300))
                    f_kHz = res.x[0] / M * HZ_PER_KM / 1e3
                    attempts.append(dict(window=(lo, hi), x0_im=x0[1],
                                         wM=res.x[0], Im=res.x[1],
                                         f_kHz=f_kHz, absAin=res.fun))
            out.setdefault("attempts", {})[f"G1={G1:g},M={msun}"] = attempts
            best_att = min(attempts, key=lambda a: a.get("absAin", 1e9)
                           if isinstance(a, dict) and "absAin" in a else 1e9)
            status = ("CONVERGES" if best_att.get("absAin", 1) < 1e-5 else
                      "NOT LOCALISED")
            print(f"G1={G1:g}, M={msun}: best |Ain|="
                  f"{best_att.get('absAin', float('nan')):.2e} -> {status}"
                  + (f", f={best_att['f_kHz']:.3f} kHz"
                     if "f_kHz" in best_att else ""))

    safe_path = DATA / "fmode26_retry.json"
    safe_path.write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                    default=float), encoding="utf-8")
    print("wrote fmode26_retry.json")


def safe_path(target) -> Path:
    resolved = Path(target).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path outside project root: {resolved}")
    return resolved


if __name__ == "__main__":
    main()
