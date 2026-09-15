"""Search for kappa_- = 0 (triple root of f) with a MONOTONICALLY decreasing density (WEC guaranteed).

We build rho(u) via the logarithmic slope sigma(u) = -d ln rho / d ln u >= 0:
    sigma(u) = 2 S(u; u1, w1) - d * exp(-((ln u - ln u2)/w2)^2) + 4 S(u; u3, w3),   S -- a sigmoid in ln u,
i.e. the slope grows from 0 (core) to 2 (halo ~R^-2), dips to 2 - d near u2, then rises to 6 (cutoff).
rho = exp(-int sigma d ln u) is monotone for sigma >= 0 (d <= 2). h = 2 m/R ∝ H(u) = (2/u) int s u^2 du.
h' ∝ q = 4 pi R^3 rho - m; q' = 4 pi R^2 (2 rho + R rho') = 4 pi R^2 rho (2 - sigma): H grows while sigma < 2.
A dip of sigma below 2 after rising above 2 gives an S-shaped H (max, then min, then growth) -- two inner
intersections with level 1 -> three inner horizons; when the max and min merge -- a stationary inflection -- a triple root.
Algorithm: for given (u1, u2, u3, w) we scan the dip depth d; find d*, at which the two stationary points of H
merge; the scale scale = 1/H(u_s) gives f = 1 - scale*H with a triple root; we check it with the full pipeline.
Run: python src/stability/triple_root_monotone.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "approach_map"))
import approach_map as am  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[2] / "data" / Path(__file__).resolve().parent.name

LU = np.linspace(np.log(1e-4), np.log(3e3), 60001)
U = np.exp(LU)


def sigma_of(d, u1=1.0, u2=4.0, u3=20.0, w1=0.25, w2=0.35, w3=0.25, s1=3.0, s_end=6.0):
    """Slope: 0 -> s1 (>2, q decreases through zero: maximum of H) -> dips to s1-d (<2 for d>s1-2: q grows through zero:
    minimum of H) -> s_end (cutoff: final maximum of H). d <= s1 keeps sigma >= 0 (monotonicity of rho)."""
    S = lambda uu, w: 1 / (1 + np.exp(-(LU - np.log(uu)) / w))
    return s1 * S(u1, w1) - d * np.exp(-((LU - np.log(u2)) / w2) ** 2) + (s_end - s1) * S(u3, w3)


def profile(d, **kw):
    sig = sigma_of(d, **kw)
    ln_rho = -cumulative_trapezoid(sig, LU, initial=0.0)
    s = np.exp(ln_rho)                                  # rho/rho_c, s(0)=1
    mI = cumulative_trapezoid(s * U**3, LU, initial=0.0)   # int s u^2 du = int s u^3 d ln u
    H = 2 * mI / U
    Hp = np.gradient(H, U)
    return sig, s, mI, H, Hp


def stationary_points(Hp, lo=0.3, hi=200.0):
    mask = (U > lo) & (U < hi)
    sg = np.sign(Hp[mask])
    idx = np.nonzero(sg[:-1] * sg[1:] < 0)[0]
    return U[mask][idx], mask


def find_merge(kw):
    """d*, at which the number of stationary points of H changes from 3 to 1 (merger of max/min)."""
    def n_stat(d):
        _, _, _, _, Hp = profile(d, **kw)
        return len(stationary_points(Hp)[0])
    ds = np.linspace(0.0, 2.99, 300)
    ns = [n_stat(d) for d in ds]
    change = [i for i in range(len(ds) - 1) if ns[i] != ns[i + 1]]
    if not change:
        return None, ns
    i = change[0]
    d_star = brentq(lambda d: n_stat(d) - 2, ds[i], ds[i + 1], xtol=1e-10) if False else 0.5 * (ds[i] + ds[i + 1])
    # refinement by bisection on the discrete indicator
    lo, hi = ds[i], ds[i + 1]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if n_stat(mid) == ns[i]:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), ns


class MonotoneFamily(am.Family):
    def __init__(self, d, scale, M=1.0, kw=None):
        kw = kw or {}
        sig, s, mI, H, Hp = profile(d, **kw)
        Itot = mI[-1]
        # h = 2m/R = 8 pi rho_c R1^2 (mI/u) = (8 pi rho_c R1^2 / 2) H  =>  h = scale*H for 8 pi rho_c R1^2 = 2 scale;
        # M = 4 pi rho_c R1^3 Itot = scale R1 Itot  =>  R1 = M/(scale Itot)
        R1 = M / (Itot * scale)
        rho_c = scale / (4 * np.pi * R1**2)
        self.R1, self.rho_c, self.sig, self.s = R1, rho_c, sig, s
        self._Rg, self._mg = U * R1, 4 * np.pi * rho_c * R1**3 * mI
        self._ln_rho = np.log(s)
        super().__init__("monotone_triple", f"monotone d={d:.4g} scale={scale:.4g}", None, dict(m=M), x_min=1e-3, x_max=12.0)

    def rho(self, R):
        return self.rho_c * np.exp(np.interp(np.log(np.asarray(R, float) / self.R1), LU, self._ln_rho))

    def drho(self, R):
        R = np.asarray(R, float)
        sig = np.interp(np.log(R / self.R1), LU, self.sig)
        return -sig * self.rho(R) / R

    def geometry(self, x):
        R = np.asarray(x, float)
        m = np.interp(R, self._Rg, self._mg)
        m1 = 4 * np.pi * R**2 * self.rho(R)
        m2 = 8 * np.pi * R * self.rho(R) + 4 * np.pi * R**2 * self.drho(R)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = 1 - 2 * m / R
            fp = 2 * m / R**2 - 2 * m1 / R
            fpp = -4 * m / R**3 + 4 * m1 / R**2 - 2 * m2 / R
            D = (2 * m / R) / R**2
        return dict(R=R, Rp=np.ones_like(R), Rpp=np.zeros_like(R), f=f, fp=fp, fpp=fpp, D=D)


def find_merge_first(sigma_min=1.0, base=None):
    """Scan over s1 (slope excess above 2) at fixed dip-bottom depth sigma_min = s1 - d:
    find s1*, at which the first maximum of H and the minimum merge (an ascending stationary inflection) --
    only the outer maximum survives. Returns s1*, u_s."""
    base = base or dict(u1=1.0, u2=4.0, u3=20.0, w1=0.25, w2=0.35, w3=0.25)

    def pts(s1):
        kw = dict(base, s1=s1)
        _, _, _, _, Hp = profile(s1 - sigma_min, **kw)
        return stationary_points(Hp)[0], kw
    lo, hi = 2.02, 3.5
    p_lo, _ = pts(lo)
    p_hi, _ = pts(hi)
    if len(p_hi) < 3 or len(p_lo) >= 3:
        return None, None, None
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        p, _ = pts(mid)
        if len(p) >= 3:
            hi = mid
        else:
            lo = mid
    p_hi, kw = pts(hi)
    p_lo, _ = pts(lo)
    # at hi -- three points; the merging pair is the first two (check: the third is close to the single point at lo)
    merged_first = abs(p_lo[-1] - p_hi[2]) < 0.1 * p_hi[2] if len(p_lo) else True
    return 0.5 * (lo + hi), float(0.5 * (p_hi[0] + p_hi[1])), dict(kw=kw, merged_first_pair=bool(merged_first), pts_hi=[float(v) for v in p_hi], pts_lo=[float(v) for v in p_lo])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # --- variant B: ascending inflection (triple root as an INNER horizon)
    print("B. Merger of the first maximum and minimum of H (ascending inflection): scan of the slope excess s1 at sigma_min=1")
    s1_star, u_s, info = find_merge_first(sigma_min=1.0)
    print(f"  s1* = {s1_star}, u_s = {u_s}, first pair merged: {info and info['merged_first_pair']}; points at s1 slightly above: {info and info['pts_hi']}; below: {info and info['pts_lo']}")
    summaryB = dict(s1_star=s1_star, u_s=u_s, info=info)
    if s1_star is not None:
        kw = info["kw"]
        d = s1_star - 1.0
        sig, s, mI, H, Hp = profile(d, **kw)
        for frac in (1.0, 0.999, 1.001):
            scale = frac / float(np.interp(u_s, U, H))
            fam = MonotoneFamily(d, scale, kw=kw)
            s_, x, T = am.characterize(fam)
            hs = s_["horizons"]
            kap = [round(h["kappa"], 6) for h in hs]
            Rs = [round(h["R"], 5) for h in hs]
            m_at_plus = float(np.interp(hs[-1]["R"], fam._Rg, fam._mg)) if hs else None
            print(f"    scale*H(u_s)={frac}: horizons R={Rs}, kappa={kap}; m(R+)/M={m_at_plus}; K(0)={s_['K_center']:.4g}, K_max={s_['K_max']:.4g} at R={s_['x_K_max']:.3g}; "
                  f"WEC viol={s_['WEC_violated']}, NEC viol={s_['NEC_violated']}, min(rho+p_perp)*8pi={s_['min_nec_t']:.3g}; ell={np.sqrt(3/(8*np.pi*fam.rho_c)):.4g}, R1={fam.R1:.4g}; "
                  f"T_H/S={s_.get('T_H_over_schw')}, b_c/S={s_.get('b_c_over_schw')}, min sigma={sig.min():.3f}")
            summaryB.setdefault("cases", []).append(dict(frac=frac, horizons_R=Rs, kappa=kap, m_at_plus_over_M=m_at_plus, K_center=s_["K_center"], K_max=s_["K_max"],
                                                         WEC_violated=s_["WEC_violated"], NEC_violated=s_["NEC_violated"], ell=float(np.sqrt(3 / (8 * np.pi * fam.rho_c))),
                                                         T_H_over_schw=s_.get("T_H_over_schw"), b_c_over_schw=s_.get("b_c_over_schw")))
    (OUT / "triple_root_monotone_B.json").write_text(json.dumps(summaryB, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print("\nA. Merger of the minimum and the outer maximum (descending inflection) -- for completeness:")
    kw = dict(u1=1.0, u2=4.0, u3=20.0, w1=0.25, w2=0.35, w3=0.25, s1=3.0)
    d_star, ns = find_merge(kw)
    print(f"Profile: sigma from 0 to {kw['s1']}, dip of depth d near u2={kw['u2']}, then to 6 near u3={kw['u3']}")
    print(f"  number of stationary points of H at d=0: {ns[0]}, at d=2.99: {ns[-1]}; merger at d* = {d_star}")
    summary = dict(kw=kw, d_star=d_star, n_stat_d0=ns[0], n_stat_dmax=ns[-1])
    if d_star is None:
        print("  No S-shape -- a triple root is unreachable in this family")
    else:
        for d in (d_star + 0.15, d_star, max(d_star - 0.15, 0.0)):
            sig, s, mI, H, Hp = profile(d, **kw)
            pts, mask = stationary_points(Hp)
            print(f"\n  d={d:.4f}: stationary points of H at u = {np.round(pts, 3)}, min sigma = {sig.min():.3f} (>=0 => rho monotone)")
            # choose the level: for d>d* there are three points (max, min, ...): set the horizon level 1 at the local minimum of H (two roots merge -> kappa=0 double)
            # for d=d*: stationary inflection -> triple root. Level: scale = 1/H(u_s).
            if len(pts) >= 3:
                u_s = pts[1]            # minimum: level at the minimum -- double tangency inside the trap
            elif len(pts) == 2:
                u_s = pts[1]            # merged pair (descending inflection)
            else:
                u_s = pts[0]
            H_s = float(np.interp(u_s, U, H))
            for frac in (1.0, 0.999, 1.001):
                scale = frac / H_s
                fam = MonotoneFamily(d, scale, kw=kw)
                s_, x, T = am.characterize(fam)
                hs = s_["horizons"]
                kap = [round(h["kappa"], 6) for h in hs]
                Rs = [round(h["R"], 5) for h in hs]
                print(f"    scale*H(u_s)={frac}: horizons R={Rs}, kappa={kap}; K(0)={s_['K_center']:.4g}, K_max={s_['K_max']:.4g} at R={s_['x_K_max']:.3g}; "
                      f"WEC viol={s_['WEC_violated']}, NEC viol={s_['NEC_violated']}, min(rho+p_perp)*8pi={s_['min_nec_t']:.3g}; "
                      f"ell={np.sqrt(3/(8*np.pi*fam.rho_c)):.4g}, R1={fam.R1:.4g}; T_H/S={s_.get('T_H_over_schw')}, b_c/S={s_.get('b_c_over_schw')}")
                summary.setdefault("cases", []).append(dict(d=d, frac=frac, horizons_R=Rs, kappa=kap, K_center=s_["K_center"], K_max=s_["K_max"],
                                                            WEC_violated=s_["WEC_violated"], NEC_violated=s_["NEC_violated"], ell=float(np.sqrt(3 / (8 * np.pi * fam.rho_c)))))
    (OUT / "triple_root_monotone.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n->", OUT)


if __name__ == "__main__":
    main()
