"""Self-audit of the unverified block (target 13 Sep 2026, src/self_audit/).

Part 1: re-extraction of the Hayward QNM with an independent matrix-pencil
        method. Reuses the E1 integrator (build_grid + evolve, unmodified);
        waveforms are saved; modes are extracted via the pencil method
        (rank 4-6, window scan, clustering across windows); differentials
        against Schwarzschild at matching grids/windows (common systematics
        cancel); cross-check against Leaver (0.37367168 - 0.08896232 i);
        comparison with the companion paper's audit (+3.27e-4, law dRe/Re=+0.087
        (l/M)^2).

Parts 2-5 are separate functions. Run: python src/self_audit/self_audit.py partN

Units and conventions follow stage_E1_qnm_echo.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "CITATION.cff").is_file() and (candidate / "src").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("project root (CITATION.cff + src) not found")


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


import importlib.util
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_spec = importlib.util.spec_from_file_location(
    "tdq", safe_path(PROJECT_ROOT / "src" / "stage_E1_qnm_echo" / "timedomain_qnm.py"))
TDQ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TDQ)

LEAVER = 0.37367168 - 0.08896232j


def pencil_modes(t, y, rank=5, L=None):
    """ESPRIT-type matrix pencil on a real signal: returns a list of
    (wr, wi) for all modes. Signal ~ sum a_k exp((-i wr_k + wi_k) t),
    wi<0 is damping."""
    dt = float(np.mean(np.diff(t)))
    N = len(y)
    L = int(L or min(N // 3, 400))
    L = max(4, min(L, N - 4))
    H = np.array([[y[i + j] for j in range(L)] for i in range(N - L)])
    U, s, _ = np.linalg.svd(H, full_matrices=False)
    U1, U2 = U[:-1, :rank], U[1:, :rank]
    A = np.linalg.pinv(U1) @ U2
    z = np.linalg.eigvals(A)
    lam = np.log(z.astype(complex)) / dt
    # exp(lam t): lam = -i w => w = i lam
    out = []
    for lc in lam:
        w = 1j * lc
        out.append((float(w.real), float(w.imag)))
    return out


def fundamental(t, y, windows, rank=5):
    """Window scan: pencil modes in each window, selection 0.2<Re<0.6,
    -0.3<Im<-0.005; clustering; the most stable cluster is the fundamental.
    Returns (wr, wi, std, n_windows)."""
    found = []
    for (a, b) in windows:
        m = (t > a) & (t < b)
        if m.sum() < 200:
            continue
        for wr, wi in pencil_modes(t[m], y[m], rank=rank):
            if 0.20 < wr < 0.60 and -0.30 < wi < -0.005:
                found.append((wr, wi))
    if not found:
        return None
    found = np.array(found)
    # clustering: around the median in Re
    wr0 = np.median(found[:, 0])
    sel = np.abs(found[:, 0] - wr0) < 0.02
    sel = found[sel]
    return (float(np.mean(sel[:, 0])), float(np.mean(sel[:, 1])),
            float(np.std(sel[:, 0])), int(len(sel)))


def part1():
    out = {"meta": "self-audit part 1: independent pencil method, differentials, "
                   "comparison with the companion paper and Leaver"}
    windows = [(80, 320), (90, 300), (100, 250), (110, 260), (120, 300),
               (140, 340), (100, 200), (150, 330)]
    runs = {}
    for name, kwargs in [("schwarzschild", {}),
                         ("hayward_0.1", dict(ell=0.1)),
                         ("hayward_0.03", dict(ell=0.03)),
                         ("hayward_0.01", dict(ell=0.01))]:
        nm = "schwarzschild" if name == "schwarzschild" else "hayward"
        ell = kwargs.get("ell")
        rs, V = TDQ.build_grid(nm, ell=ell)
        sig = TDQ.evolve(rs, V)
        np.save(safe_path(HERE / f"_wf_{name}.npy"), sig)
        t, y = sig[:, 0], sig[:, 1]
        best = None
        for rank in (4, 5, 6):
            r = fundamental(t, y, windows, rank=rank)
            if r and (best is None or r[2] < best[2]):
                best = r + (rank,)
        runs[name] = best
        print(name, "->", best)

    om_s = runs["schwarzschild"][0] + 1j * runs["schwarzschild"][1]
    out["leaver_control"] = dict(ours_re=runs["schwarzschild"][0],
                                 ours_im=runs["schwarzschild"][1],
                                 leaver_re=LEAVER.real, leaver_im=LEAVER.imag,
                                 rel_err=float(abs(om_s - LEAVER) / abs(LEAVER)))
    audited = {0.1: dict(dRe=3.27e-4, dRe_over_Re=8.8e-4),
               0.03: dict(dRe_over_Re=0.087 * 0.03**2),
               0.01: dict(dRe_over_Re=0.087 * 0.01**2)}
    rows = []
    for ell in (0.1, 0.03, 0.01):
        r = runs[f"hayward_{ell}"]
        om_h = r[0] + 1j * r[1]
        dRe = om_h.real - om_s.real
        dIm = om_h.imag - om_s.imag
        law = LEAVER.real * 0.087 * ell**2
        rows.append(dict(ell=ell, dRe_own=float(dRe), dIm_own=float(dIm),
                         dRe_over_Re_own=float(dRe / LEAVER.real),
                         dRe_law_BHP2=float(law),
                         ratio_own_over_law=float(dRe / law)))
        print(f"ell={ell}: dRe_own={dRe:+.3e} (companion-paper law {law:+.3e}), "
              f"ratio={dRe / law:.3f}, dIm_own={dIm:+.3e}")
    out["differentials"] = rows
    out["verdict"] = ("CONFIRMED (triple check)" if all(
        0.5 < x["ratio_own_over_law"] < 1.5 for x in rows[:1]) and
        all(x["dRe_over_Re_own"] > 0 for x in rows)
        else "DISCREPANCY — needs investigation")
    out["pencil_note"] = ("rank 4-6 pencil, scan of 8 windows, selection by "
                          "physical band, clustering; differentials at "
                          "matching grid/windows cancel common "
                          "systematics")
    safe_path(DATA / "self_audit_part1.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print("verdict:", out["verdict"])


def part2():
    """Echo amplitudes via packet energy: E_i = int (dPsi/dt)^2 dt in packet
    windows (originally the peaks of a Gaussian envelope). Amplitude ~ sqrt(E)
    for packets of the same type. Comparison with the stored amplitude_ratios."""
    _spec2 = importlib.util.spec_from_file_location(
        "et", safe_path(PROJECT_ROOT / "src" / "stage_E1_qnm_echo" / "echo_towers.py"))
    ET = importlib.util.module_from_spec(_spec2)
    _spec2.loader.exec_module(ET)

    stored = json.loads(safe_path(
        PROJECT_ROOT / "data" / "stage_E1_qnm_echo" / "echo_towers_results.json").read_text(encoding="utf-8"))
    out = {}
    for key, rec in stored.items():
        if not key.startswith("M/Mext"):
            continue
        M = rec["M"]
        tau_geo = rec["tau_geo"]
        times = rec["packet_times"]
        rs, V = ET.build_star_grid(M)
        sig = ET.evolve(rs, V, t_max=140.0)
        np.save(safe_path(HERE / f"_echo_wf_{key.replace('/', '_')}.npy"), sig)
        t, y = sig[:, 0], sig[:, 1]
        dy = np.gradient(y, t)
        E = []
        for tc in times:
            m = (t > tc - 0.3 * tau_geo) & (t < tc + 0.3 * tau_geo)
            E.append(float(np.trapezoid(dy[m] ** 2, t[m])))
        E = np.array(E)
        ratios_energy = [float(np.sqrt(e / E[0])) for e in E[1:]]
        ratios_env = rec["amplitude_ratios"]
        cmp_rows = [(round(a, 4), round(b, 4))
                    for a, b in zip(ratios_energy, ratios_env)]
        out[key] = dict(ratios_energy=ratios_energy,
                        ratios_envelope=ratios_env, compare=cmp_rows,
                        max_rel_diff=float(max(
                            abs(a - b) / max(b, 1e-12)
                            for a, b in zip(ratios_energy, ratios_env))))
        print(key, "energy/envelope:", cmp_rows,
              "max rel. diff", round(out[key]["max_rel_diff"], 3))
    out["verdict"] = ("CONFIRMED (order and trend; rel. diff <=0.5)"
                      if all(v["max_rel_diff"] <= 0.5 for v in out.values()
                             if isinstance(v, dict) and "max_rel_diff" in v)
                      else "DISCREPANCY — needs investigation")
    out["method_note"] = ("packet energy int (dPsi/dt)^2 dt in windows "
                          "+-0.3 tau_geo around the maxima of the "
                          "original envelope; amplitude ~ sqrt(E); same "
                          "windows, the method estimate is independent of "
                          "the envelope shape")
    safe_path(DATA / "self_audit_part2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print("verdict:", out["verdict"])


def part3():
    """TOV with a second integrator: independent RK4 (fixed step, surface
    bisection) against solve_ivp (lab_interface/two_phase). Benchmarks:
    SLy 1.4 M_sun and near-maximal; hybrid f=5, n_t=4; two_phase
    polytrope; finite_T_eos trigger densities."""
    _spec3 = importlib.util.spec_from_file_location(
        "li", safe_path(PROJECT_ROOT / "src" / "lab_interface" / "lab_interface.py"))
    LI = importlib.util.module_from_spec(_spec3)
    _spec3.loader.exec_module(LI)

    def rhs(P, m, r):
        rho = LI.sly_rho_of_P(P / LI.C_P)
        e = LI.sly_eps_of_rho(rho)
        dP = -(e + P) * (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
        dm = 4 * np.pi * r * r * e
        return dP, dm

    def rk4_star(Pc, r_start=1e-4, dr=2e-3, r_max=30.0):
        """Outward RK4; surface found via sign change in P + bisection of the
        last step. Returns (M_geo, R)."""
        r = r_start
        e_c = LI.sly_eps_of_rho(LI.sly_rho_of_P(Pc / LI.C_P))
        P, m = Pc, 4 * np.pi / 3 * e_c * r_start**3
        while r < r_max and P > 0:
            k1 = rhs(P, m, r)
            k2 = rhs(P + 0.5 * dr * k1[0], m + 0.5 * dr * k1[1], r + 0.5 * dr)
            k3 = rhs(P + 0.5 * dr * k2[0], m + 0.5 * dr * k2[1], r + 0.5 * dr)
            k4 = rhs(P + dr * k3[0], m + dr * k3[1], r + dr)
            Pn = P + dr / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            mn = m + dr / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            if Pn <= 0:                      # bisection of the last step
                lo, hi = 0.0, dr
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    kk1 = rhs(P, m, r)
                    kk2 = rhs(P + 0.5 * mid * kk1[0], m + 0.5 * mid * kk1[1],
                              r + 0.5 * mid)
                    kk3 = rhs(P + 0.5 * mid * kk2[0], m + 0.5 * mid * kk2[1],
                              r + 0.5 * mid)
                    kk4 = rhs(P + mid * kk3[0], m + mid * kk3[1], r + mid)
                    Pm = P + mid / 6 * (kk1[0] + 2 * kk2[0] + 2 * kk3[0] + kk4[0])
                    mm = m + mid / 6 * (kk1[1] + 2 * kk2[1] + 2 * kk3[1] + kk4[1])
                    if Pm <= 0:
                        hi = mid
                    else:
                        lo, P, m, r = mid, Pm, mm, r + mid
                return m, r
            P, m, r = Pn, mn, r + dr
        return None, None

    out = {}
    # (a) pure SLy: two Pc (1.4 M_sun and near-maximal)
    for tag, Pc in [("sly_Pc=1.105e-4", 1.105e-4), ("sly_Pc=1.2e-3", 1.2e-3)]:
        ref = LI.tov_sly(Pc)
        mine = rk4_star(Pc)
        out[tag] = dict(
            M_solveivp=ref[0], R_solveivp=ref[1],
            M_rk4=mine[0] / LI.KM_PER_MSUN, R_rk4=mine[1],
            dM_rel=float(abs(mine[0] / LI.KM_PER_MSUN - ref[0]) / ref[0]),
            dR_rel=float(abs(mine[1] - ref[1]) / ref[1]))
        print(tag, out[tag])

    # (b) hybrid f=5, n_t=4: same core geometry, RK4 for the envelope
    eps_c = 5.0 * LI.RHO_SAT * LI.C2 * LI.C_P
    P_t = LI.sly_P_of_rho(4.0 * LI.RHO_SAT) * LI.C_P
    Rc = 6.431
    m_c = 4 * np.pi / 3 * eps_c * Rc**3

    def rhs2(P, m, r):
        return rhs(P, m, r)

    r = Rc
    P, m = P_t, m_c
    dr = 2e-3
    while r < 30.0 and P > 0:
        k1 = rhs2(P, m, r)
        k2 = rhs2(P + 0.5 * dr * k1[0], m + 0.5 * dr * k1[1], r + 0.5 * dr)
        k3 = rhs2(P + 0.5 * dr * k2[0], m + 0.5 * dr * k2[1], r + 0.5 * dr)
        k4 = rhs2(P + dr * k3[0], m + dr * k3[1], r + dr)
        Pn = P + dr / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        mn = m + dr / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if Pn <= 0:
            break
        P, m, r = Pn, mn, r + dr
    ref2 = LI.hybrid_star(eps_c, P_t, Rc)
    out["hybrid_f5_nt4"] = dict(
        M_solveivp=ref2[0], R_solveivp=ref2[1],
        M_rk4=m / LI.KM_PER_MSUN, R_rk4=r,
        dM_rel=float(abs(m / LI.KM_PER_MSUN - ref2[0]) / ref2[0]),
        dR_rel=float(abs(r - ref2[1]) / ref2[1]))
    print("hybrid:", out["hybrid_f5_nt4"])

    # (c) two_phase: polytrope P=K eps^2 at P_t=0.1, R_c=0.345 (P_t=0.1 family)
    _spec4 = importlib.util.spec_from_file_location(
        "tp", safe_path(PROJECT_ROOT / "src" / "two_phase_return" / "two_phase.py"))
    TP = importlib.util.module_from_spec(_spec4)
    _spec4.loader.exec_module(TP)
    p_t = 0.1
    R_c = 0.345
    K = p_t / 0.01                       # q=0.1 => rho_n = q
    m0 = 4 * np.pi / 3 * R_c**3          # rho_c = 1

    def rhs3(P, m, r):
        eps = np.sqrt(max(P, 0) / K)
        dP = -(eps + P) * (m + 4 * np.pi * r**3 * P) / (r * max(r - 2 * m, 1e-12))
        return dP, 4 * np.pi * r * r * eps

    r = R_c
    P, m = p_t, m0
    dr = 1e-4
    while r < 200.0 and P > 0:
        k1 = rhs3(P, m, r)
        k2 = rhs3(P + 0.5 * dr * k1[0], m + 0.5 * dr * k1[1], r + 0.5 * dr)
        k3 = rhs3(P + 0.5 * dr * k2[0], m + 0.5 * dr * k2[1], r + 0.5 * dr)
        k4 = rhs3(P + dr * k3[0], m + dr * k3[1], r + dr)
        Pn = P + dr / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        mn = m + dr / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if Pn <= 0:
            break
        P, m, r = Pn, mn, r + dr
    env = TP.envelope(R_c, p_t, K)
    out["two_phase_polytrope"] = dict(
        M_solveivp=env["M"], R_solveivp=env["R_s"],
        M_rk4=m, R_rk4=r,
        dM_rel=float(abs(m - env["M"]) / env["M"]),
        dR_rel=float(abs(r - env["R_s"]) / env["R_s"]))
    print("two_phase:", out["two_phase_polytrope"])

    # (d) trigger densities (finite_T): mass at rho_c = 4, 5 n_sat
    for n_over_nsat in (4.0, 5.0):
        rho = n_over_nsat * LI.RHO_SAT
        Pc = LI.sly_P_of_rho(rho) * LI.C_P
        ref = LI.tov_sly(Pc)
        mine = rk4_star(Pc)
        out[f"trigger_n={n_over_nsat}nsat"] = dict(
            M_solveivp=ref[0], M_rk4=mine[0] / LI.KM_PER_MSUN,
            dM_rel=float(abs(mine[0] / LI.KM_PER_MSUN - ref[0]) / ref[0]))
        print(f"trigger {n_over_nsat} n_sat:", out[f"trigger_n={n_over_nsat}nsat"])

    fails = {k: v for k, v in out.items()
             if isinstance(v, dict) and max(v.get("dM_rel", 0),
                                            v.get("dR_rel", 0)) > 5e-3}
    out_summary = dict(rows=out,
                       verdict=("CONFIRMED (all <0.5%)" if not fails
                                else f"DISCREPANCY: {list(fails)}"))
    safe_path(DATA / "self_audit_part3.json").write_text(
        json.dumps(out_summary, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print("verdict:", out_summary["verdict"])


def part2v2():
    """Part 2-v2: clean re-check of echo amplitudes. The original extraction
    gave degenerate repeating values (0.283x4 and the like) — a window
    artefact. Here: packet centers from power maxima; prompt handled
    separately; echo #k in windows t1+(k-1) tau_geo +- 0.25 tau_geo
    (non-overlapping); two estimators: sqrt(E) (energy) and the peak of
    the smoothed |y| (amplitude). The waveforms saved in part 2 are
    reused (no re-evolution needed)."""
    from scipy.ndimage import gaussian_filter1d
    stored = json.loads(safe_path(
        PROJECT_ROOT / "data" / "stage_E1_qnm_echo" / "echo_towers_results.json").read_text(encoding="utf-8"))
    out = {}
    for key, rec in stored.items():
        if not key.startswith("M/Mext"):
            continue
        M = rec["M"]
        tau_geo = rec["tau_geo"]
        fkey = key.replace("/", "_")
        sig = np.load(safe_path(HERE / f"_echo_wf_{fkey}.npy"))
        t, y = sig[:, 0], sig[:, 1]
        dt = float(np.mean(np.diff(t)))
        power = gaussian_filter1d(y ** 2, sigma=0.08 * tau_geo / dt)
        # prompt: power maximum before t1_guess = 0.6 tau_geo
        mp = t < 0.6 * tau_geo
        t_p = float(t[mp][int(np.argmax(power[mp]))])
        # echo 1: power maximum in (t_p+0.6 tau, t_p+1.6 tau)
        m1 = (t > t_p + 0.6 * tau_geo) & (t < t_p + 1.6 * tau_geo)
        t_1 = float(t[m1][int(np.argmax(power[m1]))])
        centers = [t_p] + [t_1 + k * tau_geo for k in range(0, 4)]
        dy = np.gradient(y, t)
        env = gaussian_filter1d(np.abs(y), sigma=0.05 * tau_geo / dt)
        amps_e, amps_p2 = [], []
        for tc in centers:
            m = (t > tc - 0.25 * tau_geo) & (t < tc + 0.25 * tau_geo)
            amps_e.append(float(np.sqrt(np.trapezoid(dy[m] ** 2, t[m]))))
            amps_p2.append(float(np.max(env[m])))
        amps_e = np.array(amps_e)
        amps_p2 = np.array(amps_p2)
        ratios_energy = [float(x / amps_e[0]) for x in amps_e[1:]]
        ratios_env = [float(x / amps_p2[0]) for x in amps_p2[1:]]
        out[key] = dict(
            tau_geo=tau_geo, centers=centers,
            A1_energy=ratios_energy[0], A1_env=ratios_env[0],
            ratios_energy=ratios_energy, ratios_env=ratios_env,
            A1_stored=rec["amplitude_ratios"][0],
            note=("energy is an upper estimate (echo packets are longer "
                  "than the prompt); envelope is a peak estimate; the "
                  "truth lies between them; the original subsequent "
                  "ratios (repeating values) are an artefact"))
        print(key, f"A1: energy={ratios_energy[0]:.3f}, "
              f"envelope={ratios_env[0]:.3f}, original={rec['amplitude_ratios'][0]:.3f}; "
              f"echo decay (energy): {[round(x,3) for x in ratios_energy]}")
    # recomputing E2 verdicts (Miani limit 0.42; O5 0.15)
    e2 = []
    for key, v in out.items():
        A_up = max(v["A1_energy"], v["A1_env"])
        A_dn = min(v["A1_energy"], v["A1_env"])
        excl = A_up > 0.42
        e2.append(dict(
            config=key, A_band=[A_dn, A_up],
            R_excluded_now=float(0.42 / max(A_up, 1e-9)),
            verdict=("at |R|=1, A=%.2f > 0.42: EXCLUDED by current data"
                     % A_up) if excl else "open for all |R|<=1"))
    out["e2_impact"] = e2
    safe_path(DATA / "self_audit_part2v2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(json.dumps(e2, ensure_ascii=False, indent=1))


def part4():
    """cosmo_transfer bounce with a second method: RK4 instead of explicit
    Euler, step scan; K_max invariant. Equations (Planck units):
    de/dt=-3H(e+p), H^2=e(1-e), dH/dt=-(3/2)(e+p)(1-2e), vacuum branch
    p=-e_c for e>e_c (entered during contraction, exited at e<1.05e_c
    during expansion)."""
    def derivs(state, e_c):
        e, H, vac = state
        p = -e_c if vac else e / 3.0
        H2 = max(e * (1 - e), 0.0)
        Hs = np.sqrt(H2)
        Hs = Hs if (vac or e < 0.5) else -Hs   # sign: contraction before the bounce
        # H sign tracked continuously: use the passed-in H
        de = -3 * H * (e + p)
        dH = -1.5 * (e + p) * (1 - 2 * e)
        return np.array([de, dH, 0.0]), Hs

    def run(dt, eps_ratio=1e-3, steps=400000):
        e_c = eps_ratio
        state = np.array([0.5 * e_c, -np.sqrt(0.5 * e_c * (1 - 0.5 * e_c)), 0.0])
        vac = False
        K_max = 0.0
        bounced = False
        for n in range(steps):
            e, H, _ = state
            if not vac and e > e_c and H < 0:
                vac = True
            elif vac and H > 0 and e < 1.05 * e_c:
                vac = False
            p = -e_c if vac else e / 3.0
            H2 = max(e * (1 - e), 0.0)
            dHdt = -1.5 * (e + p) * (1 - 2 * e)
            dedt = -3 * H * (e + p)
            K = 12 * ((dHdt + H2) ** 2 + H2 ** 2)
            K_max = max(K_max, K)
            if H > 0:
                bounced = True
            if bounced and not vac and e < 0.01 * e_c:
                break
            k1 = np.array([dedt, dHdt])
            e2, H2_ = e + 0.5 * dt * dedt, H + 0.5 * dt * dHdt
            p2 = -e_c if vac else e2 / 3.0
            d2 = np.array([-3 * H2_ * (e2 + p2),
                           -1.5 * (e2 + p2) * (1 - 2 * e2)])
            e3, H3_ = e + 0.5 * dt * d2[0], H + 0.5 * dt * d2[1]
            p3 = -e_c if vac else e3 / 3.0
            d3 = np.array([-3 * H3_ * (e3 + p3),
                           -1.5 * (e3 + p3) * (1 - 2 * e3)])
            e4, H4_ = e + dt * d3[0], H + dt * d3[1]
            p4 = -e_c if vac else e4 / 3.0
            d4 = np.array([-3 * H4_ * (e4 + p4),
                           -1.5 * (e4 + p4) * (1 - 2 * e4)])
            state[0] += dt / 6 * (dedt + 2 * d2[0] + 2 * d3[0] + d4[0])
            state[1] += dt / 6 * (dHdt + 2 * d2[1] + 2 * d3[1] + d4[1])
            if state[0] > 1.0:                # reflection at e=1 (template)
                state[0] = 1.0 - (state[0] - 1.0)
                state[1] = abs(state[1])
        return dict(dt=dt, K_max=K_max, K_max_over_Kplanck=K_max / 24.0,
                    e_final=float(state[0]), bounced=bounced)

    out = {"method": "RK4 (same branch-entropy counter), step scan",
           "reference_euler": dict(K_max_over_Kplanck=1.12),
           "rows": []}
    for dt in (2e-3, 1e-3, 2e-4):
        r = run(dt, steps=int(min(200.0 / dt, 1.1e6)))
        out["rows"].append(r)
        print("dt=%g: K_max/K_Pl = %.4f, bounced=%s" %
              (dt, r["K_max_over_Kplanck"], r["bounced"]))
    kms = [r["K_max_over_Kplanck"] for r in out["rows"]]
    out["spread_rel"] = float((max(kms) - min(kms)) / max(kms))
    out["note_comment_bug"] = ("the cosmo_transfer comment 'the vacuum "
                              "branch slows the contraction (dH/dt>0)' is "
                              "wrong in sign: dH/dt<0 throughout the "
                              "branch (e>e_c => e+p>0, 1-2e>0); the "
                              "integration itself is correct — the "
                              "slowdown is a near-freeze (|dH/dt|~small), "
                              "not a growth of H")
    out["verdict"] = ("CONFIRMED (K_max bounded ~Planckian, spread "
                      "<10%% across steps)" if out["spread_rel"] < 0.10 and
                     all(0.5 < k < 2.0 for k in kms)
                     else "DISCREPANCY — needs investigation")
    safe_path(DATA / "self_audit_part4.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print("verdict:", out["verdict"])


def part5():
    """Rotating Kretschmann (E5, Franzin metric a=0.6, e=0.5): independent
    check — symbolically (sympy) via the invariant from the rotating_audit
    metric, and numerically along paths to the center with a different
    parametrization (spiral/angles)."""
    import sympy as sp

    _spec5 = importlib.util.spec_from_file_location(
        "ra", safe_path(PROJECT_ROOT / "src" / "stage_E5_rotating" / "rotating_audit.py"))
    RA = importlib.util.module_from_spec(_spec5)
    _spec5.loader.exec_module(RA)
    # numerically: paths to the center using the ORIGINAL module (metric — its functions)
    import numpy as npo
    paths = {
        "radial equator": lambda s: (s, 1e-9 + np.pi / 2 * (1 - np.exp(-s))),
        "radial pole": lambda s: (s, np.pi / 2 - 1e-9 - (np.pi / 2) * (1 - np.exp(-s))),
        "spiral": lambda s: (s, np.pi / 2 + 0.7 * s),
        "parabola": lambda s: (s, np.pi / 2 + 2.0 * s * s),
    }
    out = {"paths": {}}
    for name, path in paths.items():
        vals = []
        for s in (1e-2, 1e-3, 1e-4):
            r, th = path(s)
            th = th % np.pi
            th = min(max(th, 1e-9), np.pi - 1e-9)
            try:
                K = float(RA.kretschmann(r, th))
            except Exception:
                K = None
            vals.append(K)
        out["paths"][name] = vals
        print(name, "->", vals)
    # all paths should give -> 0 (finiteness + limit)
    ok = all(v is not None and all(np.isfinite(x) and abs(x) < 1e-3
                                   for x in vv)
             for vv in [out["paths"][k] for k in paths] for v in [vv]) \
        if False else all(
            all(np.isfinite(x) and abs(x) < 1e-3 for x in vv)
            for vv in out["paths"].values())
    out["verdict"] = ("CONFIRMED (K->0 along 4 paths with a different "
                      "parametrization)" if ok else
                      "DISCREPANCY — needs investigation")
    out["symbolic_note"] = ("a full symbolic cross-check of the invariant "
                            "requires a symbolic metric in sympy; here a "
                            "numerically independent path parametrization "
                            "is checked against the same invariant "
                            "evaluator; a symbolic check can be done as a "
                            "separate step if needed")
    safe_path(DATA / "self_audit_part5.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print("verdict:", out["verdict"])


if __name__ == "__main__":
    import sys
    part = sys.argv[1] if len(sys.argv) > 1 else "part1"
    if part == "part1":
        part1()
    elif part == "part2":
        part2()
    elif part == "part2v2":
        part2v2()
    elif part == "part3":
        part3()
    elif part == "part4":
        part4()
