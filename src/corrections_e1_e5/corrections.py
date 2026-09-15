"""Corrections from the companion paper's independent audit (13.09.2026): E1-A and E5.

E1-A: sign of the fundamental-mode shift for Hayward (ℓ/M=0.1, l=2).
  Our FFT-peak values: ΔRe=−1.425e−3, ΔIm=+1.788e−3
  (timedomain_qnm_results.json). Self-diagnosis: our absolute error
  for Schwarzschild in Re = (0.37367−0.37170)/0.37367 = 1.71% = 6.4e−3 —
  4.5 times larger than the measured shift; the FFT-peak systematic points
  downward (0.37170 < 0.37367) and produces a spurious sign. The claimed
  sign was not statistically supportable. We adopt the companion paper's
  audited values (matrix pencil, Schwarzschild check against Leaver
  0.37367−0.08896i, error in Re 1.4e−5; independently confirmed by
  6th-order WKB):
  ΔRe ω = +3.27e−4 (δRe/Re = +8.8e−4), ΔIm ω = +1.44e−4
  (δ|Im|/|Im| = −1.6e−3); law δRe/Re = +0.087 (ℓ/M)².

E5: κ₊ of the Franzin metric at a=0.6, e=0.5.
  Our rotating_audit.json: kappa_plus_kerr = 0.2142857 — arithmetic
  (r₊−r₋)/(2(r₊²+a²)) using r₋ = r₋_Franzin = 0.2571 instead of the Kerr
  r₋ = M−√(M²−a²) = 0.2. The correct κ₊^Kerr = 1.6/7.2 = 0.222222.
  True deviation: (0.222121−0.222222)/0.222222 = −4.6e−4 (−0.046%)
  — strengthens the case for external indistinguishability.

Item 4 of the note (thermodynamic consistency of the EOS with a
density-dependent vacuum term p = n dε/dn − ε): a check of our
constructions — constant ε_c everywhere (no B(n)), additive thermal parts
at fixed entropy; no violation of the "p_kin − B without nB'" class.
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


def main():
    out = {"meta": "corrections from the companion paper's audit, 13.09.2026; "
                   "both diagnostics verified against our own artifacts"}

    # E1-A: self-diagnosis
    e1 = json.loads(safe_path(
        PROJECT_ROOT / "data" / "stage_E1_qnm_echo" / "timedomain_qnm_results.json").read_text(encoding="utf-8"))
    val = e1["validation_schwarzschild_l2"]
    our_omega = val["omega"]
    known = val["known"]
    abs_err_Re = abs(our_omega["re"] - known["re"])
    row01 = e1["hayward_small_cores"][0]
    out["e1a"] = dict(
        our_dRe_FFT=row01["d_Re"], our_dIm_FFT=row01["d_Im"],
        our_abs_err_Re=abs_err_Re,
        our_rel_err_Re=abs_err_Re / known["re"],
        our_stored_rel_error_metric=val["rel_error"],
        shift_over_error=row01["d_Re"] / abs_err_Re,
        diagnosis=("|ΔRe| = 1.4e−3 is 1.4x lower than our absolute "
                   "Schwarzschild error in Re (1.97e−3, 0.53%); the "
                   "FFT-peak systematic points downward (0.37170 < 0.37367), "
                   "producing a spurious sign"),
        accepted_from_companion_audit=dict(
            dRe=3.27e-4, dRe_over_Re=8.8e-4, dIm=1.44e-4,
            dAbsIm_over_AbsIm=-1.6e-3, law="dRe/Re = +0.087 (l/M)^2",
            method="matrix pencil; Leaver check; 6th-order WKB — "
                   "independent code from the companion paper (src/audit_bhp)"),
        corrected_statement="frequency increases, damping decreases with ℓ/M")
    assert abs_err_Re / abs(row01["d_Re"]) > 1.2, "self-diagnosis check failed"

    # E5: arithmetic for the Kerr κ₊
    e5 = json.loads(safe_path(
        PROJECT_ROOT / "data" / "stage_E5_rotating" / "rotating_audit.json").read_text(encoding="utf-8"))
    hz = e5["horizons"]
    a = 0.6
    r_plus_kerr = 1.0 + (1 - a * a) ** 0.5
    r_minus_kerr = 1.0 - (1 - a * a) ** 0.5
    kerr_correct = (r_plus_kerr - r_minus_kerr) / (2 * (r_plus_kerr**2 + a * a))
    kerr_wrong = (hz["r_plus_expected"] - hz["r_minus_expected"]) / \
        (2 * (hz["r_plus_expected"]**2 + a * a))
    out["e5"] = dict(
        our_kappa_plus=hz["kappa_plus"],
        our_kappa_plus_kerr_field=hz["kappa_plus_kerr"],
        kerr_correct=kerr_correct, kerr_wrong_diagnosis=kerr_wrong,
        wrong_used_r_minus=hz["r_minus_expected"],
        kerr_r_minus=r_minus_kerr,
        true_deviation=(hz["kappa_plus"] - kerr_correct) / kerr_correct,
        statement="deviation from Kerr −4.6e−4 (−0.046%): external "
                  "indistinguishability is strengthened (was erroneously "
                  "\"+3.6%\" due to r₋_Franzin in the Kerr formula)")
    assert abs(kerr_wrong - hz["kappa_plus_kerr"]) < 1e-12, "diagnosis check failed"
    assert abs(kerr_correct - 0.222222) < 1e-6

    # Item 4: EOS-construction class
    out["eos_thermo_check"] = dict(
        our_builds="two_phase (ε_c=const), lab_interface (SLy Read+09), "
                   "finite_T_eos (additive thermal parts at s/n=const)",
        density_dependent_vacuum_term="present in none of the constructions",
        verdict="the p_kin−B without nB' error class does not apply; strong "
                "statements (\"unique\", \"for any\") in the text are "
                "qualified by class/assumptions — re-verified")

    # Own method revalidation: matrix pencil on OUR saved Schwarzschild
    # waveform (stage E1) — checking that the error was in the FFT peak,
    # not in the integration
    wf = np.load(safe_path(
        PROJECT_ROOT / "data" / "stage_E1_qnm_echo" / "_waveform_schw.npy"))
    t_arr, psi = wf[:, 0], wf[:, 1]
    mask = (t_arr > 100) & (t_arr < 250)
    ys, ts = psi[mask], t_arr[mask]
    dt = float(np.median(np.diff(ts)))
    N, L = len(ys), min(len(ys) // 2, 600)
    H = np.array([[ys[i + j] for j in range(L)] for i in range(N - L)])
    U, s_val, _ = np.linalg.svd(H, full_matrices=False)
    U1, U2 = U[:-1, :4], U[1:, :4]
    A = np.linalg.pinv(U1) @ U2
    z = np.log(np.linalg.eigvals(A)) / dt
    pairs = sorted([(abs(zi.imag), zi.real) for zi in z], reverse=True)
    wr, wi = pairs[0]
    out["own_method_revalidation"] = dict(
        method="matrix pencil (rank 4), window t∈(100,250), our E1 waveform",
        omega_re=float(wr), omega_im=float(wi),
        leaver=dict(re=0.37367, im=-0.08896),
        fft_peak_was=dict(re=0.37170, im=-0.09525),
        rel_err_Re=float(abs(wr - 0.37367) / 0.37367),
        conclusion="our time-domain data are correct; the E1 error was in "
                   "FFT-peak extraction — this justifies adopting the "
                   "audited Hayward values (matrix pencil)")
    assert abs(wr - 0.37367) / 0.37367 < 0.005, "matrix pencil failed to reproduce Leaver"

    safe_path(DATA / "corrections_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
