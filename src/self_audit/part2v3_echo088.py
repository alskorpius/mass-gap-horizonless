"""Part 5 of the v9 target: repeat measurement of echo amplitudes for
M/M_ext=0.88.

Method: heavy envelope smoothing (sigma=0.15 tau_geo), local maxima above
2% of the maximum, grouping of nearby peaks, amplitudes (energy and
envelope) in local windows +-0.25 of the gap. Cross-check of intervals
against tau_geo and tau_packets from the original extraction."""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
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
    stored = json.loads(safe_path(
        PROJECT_ROOT / "data" / "stage_E1_qnm_echo" / "echo_towers_results.json").read_text(encoding="utf-8"))["M/Mext=0.88"]
    tau_geo = stored["tau_geo"]
    sig = np.load(safe_path(DATA / "_echo_wf_M_Mext=0.88.npy"))
    t, y = sig[:, 0], sig[:, 1]
    dt = float(np.mean(np.diff(t)))

    env = gaussian_filter1d(np.abs(y), sigma=0.15 * tau_geo / dt)
    mask = t < 60
    te, ee = t[mask], env[mask]
    thr = 0.02 * float(np.max(ee))
    loc = [(te[i], ee[i]) for i in range(2, len(te) - 2)
           if ee[i] >= ee[i - 1] and ee[i] >= ee[i + 1] and ee[i] > thr]
    groups = []
    for tt, vv in loc:
        if groups and tt - groups[-1][-1][0] < 0.5:
            groups[-1].append((tt, vv))
        else:
            groups.append([(tt, vv)])
    peaks = []
    for g in groups:
        peaks.append((float(np.mean([x[0] for x in g])),
                      float(max(x[1] for x in g))))
    print("peaks (t, env):", [(round(a, 2), round(b, 4)) for a, b in peaks])
    gaps = np.diff([p[0] for p in peaks])
    print("intervals:", [round(g, 2) for g in gaps])
    print("tau_geo:", round(tau_geo, 3), " tau_packets (original):",
          round(stored["tau_packets"], 3))

    dy = np.gradient(y, t)
    amps_e, amps_p, centers = [], [], []
    for i, (tc, _) in enumerate(peaks):
        if i + 1 < len(peaks):
            gap = peaks[i + 1][0] - tc
        elif i >= 1:
            gap = tc - peaks[i - 1][0]
        else:
            gap = tau_geo
        w = 0.25 * max(gap, 0.6 * tau_geo)
        m = (t > tc - w) & (t < tc + w)
        amps_e.append(float(np.sqrt(np.trapezoid(dy[m] ** 2, t[m]))))
        amps_p.append(float(np.max(env[m])))
        centers.append(tc)
    ratios_e = [a / amps_e[0] for a in amps_e[1:]]
    ratios_p = [a / amps_p[0] for a in amps_p[1:]]
    print("A1: energy=%.3f envelope=%.3f (original=%.3f)" %
          (ratios_e[0], ratios_p[0], stored["amplitude_ratios"][0]))
    print("energy decay:", [round(x, 3) for x in ratios_e])
    print("envelope decay:", [round(x, 3) for x in ratios_p])

    med_gap = float(np.median(gaps)) if len(gaps) else None
    out = dict(
        method="peaks of the smoothed envelope (0.15 tau_geo), 2% "
               "threshold, grouping 0.5, local windows +-0.25 of the gap",
        peaks=peaks, gaps=[float(g) for g in gaps],
        median_gap=med_gap, tau_geo=tau_geo,
        tau_packets_stored=stored["tau_packets"],
        A1_energy=ratios_e[0], A1_envelope=ratios_p[0],
        A1_stored=stored["amplitude_ratios"][0],
        ratios_energy=[float(x) for x in ratios_e],
        ratios_envelope=[float(x) for x in ratios_p],
        verdict="")
    good_gaps = [g for g in gaps if abs(g - tau_geo) < 0.4 * tau_geo]
    regular = len(good_gaps) >= max(2, len(gaps) // 2)
    A1_band = [min(ratios_e[0], ratios_p[0]), max(ratios_e[0], ratios_p[0])]
    out["A1_band"] = A1_band
    out["verdict"] = (
        "0.88 closed: train of %d packets, %d intervals near tau_geo; "
        "A1 in band [%.2f, %.2f] — the 0.88 configuration is NOT excluded "
        "(A<0.42 even at |R|=1)"
        % (len(peaks), len(good_gaps), A1_band[0], A1_band[1])
        if regular and A1_band[1] < 0.42 else
        "0.88: train irregular (%d/%d intervals near tau_geo); "
        "A1 band [%.2f, %.2f]"
        % (len(good_gaps), len(gaps), A1_band[0], A1_band[1]))
    safe_path(DATA / "self_audit_part2v3.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print("verdict:", out["verdict"])


if __name__ == "__main__":
    main()
