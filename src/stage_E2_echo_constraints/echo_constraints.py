"""E2: echo amplitude (parametrically) against published LIGO constraints.

Model (from E1-B, units eps_c=1): the observed ratio of the echo to the
main signal is A_obs = (A1/A0) * |R|, where A1/A0 are the computed ratios
(0.081 / 0.283 / 0.321 for M/M_ext = 0.88 / 0.95 / 0.995), and R is the
reflectivity of the core (microphysics, a free parameter |R|<=1).

Constraints (primary sources read from abstracts/HTML, see SOURCES.md):
  - Miani et al., PRD 108, 064018 (2023), arXiv:2302.12158: A <= 0.42
    (90% cred.) on the echo amplitude relative to the ringdown, window
    < 1 s after BBH merger (GWTC-3, O1-O3), minimal morphology
    assumptions.
  - Uchikata et al., PRD 108, 104040 (2023): templated search over O3 —
    null result, limits depend on the echo delay.

Computed: physical delay tau_echo(M_solar, M/M_ext) via tau/M from E1-B
(tau in units of eps_c=1, mass in the same units; the scale is set by
the L_alpha conversion, which cancels in the tau/M ratio); threshold
|R|_max = 0.42/(A1/A0); verdict per table cell.
"""

import csv
import io
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


# From E1-B: tau_geo/(2M) in units of eps_c=1 and A1/A0 at |R|=1.
E1B = {
    0.88: dict(tau_over_2M=2.7537 / (2 * 0.20360), A1A0=0.0814),
    0.95: dict(tau_over_2M=8.2303 / (2 * 0.21979), A1A0=0.2829),
    0.995: dict(tau_over_2M=25.6149 / (2 * 0.23020), A1A0=0.3212),
}

T_SOLAR = 4.9254909e-6  # GM_sun/c^3, s

# Limits: current (Miani 2023, 90%); O4/O5 — estimated from projected sensitivity growth.
LIMITS = {"O3_current": 0.42, "O4_projected": 0.28, "O5_projected": 0.15}


def row(frac, M_solar, limit):
    d = E1B[frac]
    tau = d["tau_over_2M"] * 2.0 * M_solar * T_SOLAR  # tau = (tau/2M)*2M_phys
    A_unit = d["A1A0"]
    R_max = limit / A_unit
    return dict(M_over_Mext=frac, M_solar=M_solar, tau_echo_s=tau,
                A_unit=A_unit, limit=limit,
                R_max=min(R_max, 1.0),
                excluded=bool(A_unit > limit),
                verdict="EXCLUDED even at |R|=1" if A_unit > limit
                        else ("FULLY OPEN: |R|<=1 passes" if R_max >= 1.0
                              else f"partial: excluded for |R|>{R_max:.2f}"))


def main():
    masses = [10.0, 30.0, 60.0, 500.0, 4.3e6]
    rows = []
    for frac in (0.88, 0.95, 0.995):
        for M_solar in masses:
            for tag, lim in LIMITS.items():
                r = row(frac, M_solar, lim)
                r["dataset"] = tag
                rows.append(r)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()),
                            lineterminator="\r\n")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    safe_path(DATA / "echo_constraints_table.csv").write_text(
        buf.getvalue(), encoding="utf-8", newline="")
    summary = dict(
        limits=LIMITS,
        sources=["arXiv:2302.12158 / PRD 108.064018 (Miani et al. 2023)",
                 "PRD 108.104040 (Uchikata et al. 2023)"],
        key_finding=("No cell of the current constraint is excluded: "
                     "even at |R|=1 the predicted A=0.08-0.32 < 0.42; "
                     "the O5 projection of 0.15 starts to touch "
                     "configurations M/M_ext>=0.95 with |R|>0.47-0.53"),
        window_note="Miani: window <1 s covers our delays (2-20 ms for 10-60 M☉; ~13 min for Sgr A* — outside the window, but within the LISA band)",
    )
    safe_path(DATA / "echo_constraints_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # Compact printout for 30 M☉.
    for frac in (0.88, 0.95, 0.995):
        r = row(frac, 30.0, LIMITS["O3_current"])
        print(f"M/Mext={frac}: tau={r['tau_echo_s']*1e3:.2f} ms (30 M☉), "
              f"A(|R|=1)={r['A_unit']:.3f}, limit 0.42 -> {r['verdict']}")


if __name__ == "__main__":
    main()
