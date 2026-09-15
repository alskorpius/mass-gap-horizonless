"""Independent checks of the approach_map framework (not a repeat of the same formulas).

1. Analytic derivatives F_R, F_RR, R', R'' against centered finite differences.
2. Schwarzschild: G=0, K=48 m^2/R^6.
3. Hayward/Bardeen/Dymnikova: 8pi rho(0) -> 3/ell_eff^2, p=-rho at the center.
4. Covariant conservation of the source in the region f>0:
   p_r' + (2R'/R)(p_r - p_perp) + (f'/(2f))(rho + p_r) = 0   (finite differences in p_r).
5. Pocket v01: agreement with baseline pocket_model.tensors on its grid.
6. Simpson-Visser: 8pi T(n,n) at the throat = -2 a^2/R^4|_{x=0} = -2/a^2 (from -2R''/R).
Run: python src/approach_map/checks.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import approach_map as am  # noqa: E402

records = []


def record(name, err, tol):
    ok = bool(err <= tol)
    records.append(dict(check=name, error=float(err), tolerance=tol, passed=ok))
    print(f"{'OK ' if ok else 'FAIL'} {name}: err={err:.3e} tol={tol:.1e}")


fams = am.make_families()
by = {f.name: f for f in fams}

# 1. derivatives
for fam in fams:
    xs = np.array([0.05, 0.3, 0.9, 1.7, 3.0, 6.0])
    if fam.name == "simpson_visser":
        xs = np.array([0.0, 0.05, 0.3, 0.9, 1.7, 3.0])
    h = 1e-5
    g0 = fam.geometry(xs)
    gp, gm = fam.geometry(xs + h), fam.geometry(xs - h)
    fp_fd = (gp["f"] - gm["f"]) / (2 * h)
    fpp_fd = (gp["f"] - 2 * g0["f"] + gm["f"]) / h**2
    Rp_fd = (gp["R"] - gm["R"]) / (2 * h)
    Rpp_fd = (gp["R"] - 2 * g0["R"] + gm["R"]) / h**2
    record(f"{fam.name}: f' vs FD", np.max(np.abs(fp_fd - g0["fp"]) / (1 + np.abs(g0["fp"]))), 1e-7)
    record(f"{fam.name}: f'' vs FD", np.max(np.abs(fpp_fd - g0["fpp"]) / (1 + np.abs(g0["fpp"]))), 1e-4)
    record(f"{fam.name}: R' vs FD", np.max(np.abs(Rp_fd - g0["Rp"]) / (1 + np.abs(g0["Rp"]))), 1e-7)
    record(f"{fam.name}: R'' vs FD", np.max(np.abs(Rpp_fd - g0["Rpp"]) / (1 + np.abs(g0["Rpp"]))), 1e-4)

# 2. Schwarzschild
x = np.linspace(0.3, 8, 500)
T = am.tensors(by["schwarzschild"], x)
record("schwarzschild: G=0", max(np.max(np.abs(T["rho8pi"])), np.max(np.abs(T["pr8pi"])), np.max(np.abs(T["pt8pi"]))), 1e-12)
record("schwarzschild: K=48m^2/R^6", np.max(np.abs(T["K"] * x**6 / 48 - 1)), 1e-10)

# 3. de Sitter centers
ell = by["hayward"].Fkw["ell"]
m = by["hayward"].Fkw["m"]
for name, ell_eff in (("hayward", ell), ("bardeen", np.sqrt(by["bardeen"].Fkw["g"] ** 3 / (2 * m))),
                      ("dymnikova", np.sqrt(by["dymnikova"].Fkw["rstar"] ** 3 / (2 * m)))):
    T0 = am.tensors(by[name], 1e-3)
    record(f"{name}: 8pi rho(0) = 3/ell_eff^2", abs(T0["rho8pi"] / (3 / ell_eff**2) - 1), 1e-4)
    record(f"{name}: p_perp/rho(0) = -1", abs(T0["pt8pi"] / T0["rho8pi"] + 1), 1e-4)
    Tx = am.tensors(by[name], x)
    ok = np.abs(Tx["rho8pi"]) > 1e-200
    record(f"{name}: p_r/rho = -1 (everywhere, R=x)", np.max(np.abs(Tx["pr8pi"][ok] / Tx["rho8pi"][ok] + 1)), 1e-10)

# 4. conservation of the source in the regions f>0 (inside the inner and outside the outer horizon)
for fam in fams:
    hs = am.horizons(fam)
    segs = []
    if len(hs) >= 2:
        segs.append((max(fam.x_min, 1e-3), hs[0]["x"] * 0.98))
    if hs:
        segs.append((hs[-1]["x"] * 1.02, 6.0))
    else:
        segs.append((max(fam.x_min, 1e-3), 6.0))
    worst = 0.0
    for lo, hi in segs:
        xs = np.linspace(lo, hi, 400)
        h = 1e-6 * max(1.0, lo)
        Tm, T0, Tp = am.tensors(fam, xs - h), am.tensors(fam, xs), am.tensors(fam, xs + h)
        dpr = (Tp["pr8pi"] - Tm["pr8pi"]) / (2 * h)
        res = dpr + 2 * T0["Rp"] / T0["R"] * (T0["pr8pi"] - T0["pt8pi"]) + T0["fp"] / (2 * T0["f"]) * (T0["rho8pi"] + T0["pr8pi"])
        scale = 1 + np.abs(dpr) + np.abs(T0["pr8pi"]) * 2 * np.abs(T0["Rp"] / T0["R"])
        worst = max(worst, float(np.max(np.abs(res) / scale)))
    record(f"{fam.name}: conservation (f>0)", worst, 1e-5)

# 5. pocket: agreement with baseline
pk = by["pocket"]
xs = pk.grid()
Ta = am.tensors(pk, xs)
Tb = am.pocket_model.tensors(xs, pk.p)
for key in ("K", "epsilon8pi", "radial_pressure8pi", "transverse_pressure8pi", "NEC8pi"):
    ka = {"epsilon8pi": "rho8pi", "radial_pressure8pi": "pr8pi", "transverse_pressure8pi": "pt8pi"}.get(key, key)
    A, Bv = np.asarray(Ta[ka], float), np.asarray(Tb[key], float)
    mask = np.isfinite(A) & np.isfinite(Bv)
    record(f"pocket: {key} vs baseline", np.max(np.abs(A[mask] - Bv[mask]) / (1 + np.abs(Bv[mask]))), 1e-9)

# 6. Simpson-Visser: throat
sv = by["simpson_visser"]
a = sv.Rkw["a"]
T0 = am.tensors(sv, 0.0)
record("simpson_visser: 8pi T(n,n)|throat = -2/a^2", abs(T0["NEC8pi"] / (-2 / a**2) - 1), 1e-12)
record("simpson_visser: rho+p_r|throat = -2|f|/a^2", abs(T0["nec_r"] - (-2 * abs(T0["f"]) / a**2)), 1e-12)

# 7. Triple root (2205.13556): F(r_-)=0, F'(r_-)=0 (kappa_-=0), F(r_+)=0, F'(r_+)!=0, D>0, de Sitter core
tr = by["triple_root"]
kw = tr.Fkw
F_, FR_, _ = am.F_triple_root(np.array([kw["r_minus"], kw["r_plus"]]), **kw)
record("triple_root: F(r_-)=F(r_+)=0", np.max(np.abs(F_)), 1e-12)
record("triple_root: F'(r_-)=0 (kappa_-=0)", abs(FR_[0]), 1e-12)
record("triple_root: F'(r_+) != 0", 1.0 / max(abs(FR_[1]), 1e-300) * 1e-3, 1.0)
Rgrid = np.linspace(0, 50, 50001)
Pp = np.polynomial.polynomial
Ncoef = Pp.polymul(Pp.polypow([-kw["r_minus"], 1.0], 3), [-kw["r_plus"], 1.0])
Dval = Pp.polyval(Rgrid, Pp.polyadd(Ncoef, [0.0, 0.0, kw["b2"], 2 * kw["m"]]))
record("triple_root: D>0 on [0,50]", float(max(0.0, -Dval.min())), 0.0)
T0 = am.tensors(tr, 1e-6)   # first-order correction ~ 6.5 R, so the point is closer to the center
rho_c = 3 * kw["b2"] / (kw["r_minus"] ** 3 * kw["r_plus"])
record("triple_root: 8pi rho(0)=3 b2/(r_-^3 r_+)", abs(T0["rho8pi"] / rho_c - 1), 1e-4)
record("triple_root: 1-F -> 2m/R at large R", abs(am.one_minus_F_triple_root(1e4, **kw) * 1e4 / (2 * kw["m"]) - 1), 1e-2)

n_fail = sum(not r["passed"] for r in records)
out = Path(__file__).resolve().parents[2] / "data" / "approach_map" / "checks.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n{len(records) - n_fail}/{len(records)} checks passed -> {out}")
sys.exit(1 if n_fail else 0)
