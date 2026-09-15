"""Part 3-4: f-modes (polar sector, l=2) -- SLy and the hybrid branch.

Machine: RelModPy (Gittins, MIT; vendored in src/third_party/relmodpy) --
Detweiler-Lindblom 1985 equations (H1, K, W, X), matched to Zerilli per
Andersson-Kokkotas-Schutz 1995, Muller's method. Our contribution is the
EOS adapters:
(a) SLy (piecewise polytropic, as in lab_interface);
(b) hybrid-analogue EOS: p>P_t -> epsilon=eps_c (incompressible core --
a de Sitter analogue: the m(r) profile matches EXACTLY, the envelope is
consistent in the Gamma1->inf limit; the caveat is noted).

Validations:
(A) RelModPy's own validation anchor (polytrope, f-mode: Re(wM)=0.171);
(B) SLy f-mode(1.4 M_sun) ~ 1.9-2.0 kHz (literature anchor).

Units: G=c=1, lengths in km; pressure/density in km^-2.
f[Hz] = omega[km^-1]/(2 pi t_km), t_km = G*M_sun/c^3 in seconds per km...
here: 1/c = 3.3356e-6 s/km => f = omega/(2 pi 3.3356e-6).
"""

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


def safe_path(target) -> Path:
    resolved = Path(target).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path outside project root: {resolved}")
    return resolved


sys.path.insert(0, str(safe_path(PROJECT_ROOT / "src" / "third_party")))

from relmodpy.eos import EOS, EnergyPolytrope          # noqa: E402
from relmodpy.star import Star                          # noqa: E402
from relmodpy.mode import Mode                          # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_spec = importlib.util.spec_from_file_location(
    "li", safe_path(PROJECT_ROOT / "src" / "lab_interface" / "lab_interface.py"))
LI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LI)

HZ_PER_KM = 1.0 / (2 * np.pi * 3.3356e-6)     # f[Hz] = omega * HZ_PER_KM
C_P = LI.C_P                                  # km^-2 per (dyn/cm^2)


class SLyEOS(EOS):
    """Piecewise polytropic SLy (as in lab_interface); Gamma1 = segment
    Gamma."""

    def epsilon(self, p):
        rho = LI.sly_rho_of_P(p / C_P)         # g/cm^3
        return LI.sly_eps_of_rho(rho)          # already km^-2

    def Gamma(self, p):
        rho = LI.sly_rho_of_P(p / C_P)
        K, g = LI._segment(rho)
        return g

    def Gamma1(self, p):
        return self.Gamma(p)


class HybridAnalogEOS(EOS):
    """Analogue of the hybrid star: p>P_t -- dense phase eps_c
    (incompressible), p<=P_t -- SLy. The core's m(r) profile matches the
    de Sitter case exactly; the difference is only in p(r) inside the
    core (TOV fluid vs p=-eps_c) and in the core's eigenmodes (the
    analogue has extra stiff core modes -- far in frequency from the
    envelope f-mode)."""

    def __init__(self, eps_c_km2, P_t_km2, Gamma1_core=1e4):
        self.eps_c = eps_c_km2
        self.P_t = P_t_km2
        self.Gamma1_core = Gamma1_core
        self.sly = SLyEOS()

    def epsilon(self, p):
        if p > self.P_t:
            return self.eps_c
        return self.sly.epsilon(p)

    def Gamma(self, p):
        return 3.0

    def Gamma1(self, p):
        if p > self.P_t:
            return self.Gamma1_core
        return self.sly.Gamma1(p)


def fmode(star, ell=2, wM_lo=0.05, wM_hi=0.16, n_scan=18):
    """f-mode: (1) scan |Ain(omega)| along Im=-1e-6 -- localise the
    minimum; (2) 2D minimisation of |Ain|^2 over (Re omega, Im omega)
    with Nelder-Mead (the code's Muller root finder diverges for
    imprecise starting points -- dropped after diagnosis). The root
    Ain=0 is the QNM; damping corresponds to Im>0 (convention, cf.
    validation anchor A). Validated against anchor A: (0.17084, 6.19e-5)
    vs (0.171, 6.19e-5)."""
    from scipy.optimize import minimize
    M = star.M
    mode = Mode(star)
    wrs = np.linspace(wM_lo / M, wM_hi / M, n_scan)
    best_w, best_v = None, np.inf
    for wr in wrs:
        try:
            v = abs(mode.spectrum(ell, wr - 1e-6j))
        except Exception:
            continue
        if v < best_v:
            best_w, best_v = wr, v
    if best_w is None:
        raise RuntimeError("spectrum scan failed")

    def abs_ain(x):
        try:
            return abs(mode.spectrum(ell, (x[0] + 1j * x[1]) / M))
        except Exception:
            return 1e6

    res = minimize(abs_ain, x0=[best_w * M, 5e-5], method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-10, maxiter=300))
    if res.fun > 1e-5:
        raise RuntimeError(f"minimum |Ain| did not reach zero: {res.fun:.2e}")
    return (res.x[0] + 1j * res.x[1]) / M


def validation_A():
    """RelModPy validation anchor: polytrope pc=5.52e-3, f-mode
    Re(wM)=0.171, Im(wM)=+6.19e-5 (damping for Im>0 in the code's
    convention)."""
    poly = EnergyPolytrope(n=1, K=100.0)
    s = Star(poly, 5.52e-3)
    w = fmode(s, wM_lo=0.14, wM_hi=0.20, n_scan=21)
    ref = 0.171
    got = w.real * s.M
    print(f"validation A: Re(wM)={got:.4f} vs 0.171 "
          f"(dev. {abs(got - ref) / ref * 100:.2f}%), Im(wM)={w.imag * s.M:.2e}")
    assert abs(got - ref) / ref < 0.02, "anchor A not passed"
    assert w.imag > 0, "damping sign (convention)"
    return dict(got=got, ref=ref, rel=abs(got - ref) / ref,
                ImwM=float(w.imag * s.M))


def pc_for_mass(eos, M_target, lo, hi):
    from scipy.optimize import brentq

    def eq(pc):
        s = Star(eos, pc)
        return s.M - M_target

    return float(brentq(eq, lo, hi, xtol=1e-12, rtol=1e-12))


def mass_km(msun):
    return msun * 1.476625


def main():
    out = {"meta": "f-modes via RelModPy (LD85 + AKS95 + Muller), "
                   "EOS adapters are our own"}

    out["validation_A"] = validation_A()

    # SLy: f-modes at several masses; anchor 1.4 M_sun ~ 1.8-2.1 kHz
    sly_rows = []
    for msun in (1.4, 1.8, 2.0):
        pc = pc_for_mass(SLyEOS(), mass_km(msun), 1e-6, 1e-3)
        s = Star(SLyEOS(), pc)
        w = fmode(s)
        row = dict(M_solar=msun, M_km=s.M, R_km=s.R,
                   f_kHz=float(w.real * HZ_PER_KM / 1e3),
                   Im_sign_stable=bool(w.imag > 0),
                   tau_s=float(1.0 / (w.imag * HZ_PER_KM)) if w.imag > 0 else None)
        sly_rows.append(row)
        print(f"SLy {msun} M_sun: f={row['f_kHz']:.3f} kHz, "
              f"tau={row['tau_ms'] if False else (row['tau_s'] if row['tau_s'] else float('nan')):.2f} s, "
              f"stable={row['Im_sign_stable']}")
    f14 = [r for r in sly_rows if r["M_solar"] == 1.4][0]["f_kHz"]
    out["validation_B"] = dict(
        f_kHz_14=f14, anchor_range_kHz=(1.8, 2.1),
        passed=bool(1.8 <= f14 <= 2.1),
        note="literature anchor: f-modes of SLy-like EOS at 1.4 M_sun "
             "fall in 1.8-2.1 kHz (Andersson-Kokkotas 1998; "
             "Kokkotas-Ruoff 2001 -- from abstracts, status noted in "
             "SOURCES); Im(omega)>0 = damping (RelModPy convention, "
             "cross-checked against anchor A)")
    assert out["validation_B"]["passed"], "anchor B not passed"
    out["sly_fmodes"] = sly_rows

    # Hybrid-analogue branch: f=5, n_t=4 (the corner)
    eps_sat = LI.RHO_SAT * LI.C2 * LI.C_P
    eps_c = 5.0 * eps_sat
    P_t = LI.sly_P_of_rho(4.0 * LI.RHO_SAT) * C_P
    hyb_rows = []
    for Gamma1_core in (1e3, 1e5):
        eos = HybridAnalogEOS(eps_c, P_t, Gamma1_core=Gamma1_core)
        try:
            pc = pc_for_mass(eos, mass_km(2.6), 1e-6, 1e-2)
            s = Star(eos, pc)
            w = fmode(s)
            hyb_rows.append(dict(Gamma1_core=Gamma1_core,
                                 M_solar=s.M / 1.476625, R_km=s.R,
                                 f_kHz=float(w.real * HZ_PER_KM / 1e3),
                                 stable=bool(w.imag > 0),
                                 tau_s=float(1.0 / (w.imag * HZ_PER_KM))
                                 if w.imag > 0 else None))
            print(f"hybrid G1={Gamma1_core:g}: M={s.M / 1.476625:.2f}, "
                  f"f={hyb_rows[-1]['f_kHz']:.3f} kHz, stable="
                  f"{hyb_rows[-1]['stable']}")
        except Exception as exc:
            hyb_rows.append(dict(Gamma1_core=Gamma1_core, error=str(exc)[:80]))
            print(f"hybrid G1={Gamma1_core:g}: error {str(exc)[:60]}")
    out["hybrid_26"] = hyb_rows

    # f-modes along the branch (Gamma1_core=1e4)
    eos = HybridAnalogEOS(eps_c, P_t, Gamma1_core=1e4)
    branch = []
    for msun in (1.8, 2.2, 2.6, 3.0):
        try:
            pc = pc_for_mass(eos, mass_km(msun), 1e-6, 1e-2)
            s = Star(eos, pc)
            w = fmode(s)
            branch.append(dict(M_solar=msun, R_km=s.R,
                               f_kHz=float(w.real * HZ_PER_KM / 1e3),
                               stable=bool(w.imag > 0),
                               tau_s=float(1.0 / (w.imag * HZ_PER_KM))
                               if w.imag > 0 else None))
            print(f"branch {msun} M☉: f={branch[-1]['f_kHz']:.3f} kHz, "
                  f"stable={branch[-1]['stable']}")
        except Exception as exc:
            branch.append(dict(M_solar=msun, error=str(exc)[:80]))
            print(f"branch {msun}: error {str(exc)[:60]}")
    out["hybrid_branch_fmodes"] = branch

    safe_path(DATA / "fmode_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print("wrote fmode_results.json")


if __name__ == "__main__":
    main()
