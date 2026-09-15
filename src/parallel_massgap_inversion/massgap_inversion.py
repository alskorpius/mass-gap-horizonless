"""Parallel mini-stage P3: inversion of the branch criterion onto the mass gap.

Our result (stage 4): the horizonless branch exists for
M sqrt(eps_c) < c_ext(n), c_ext in [0.18, 0.39] (profile shape; n=6:
0.2314). Inversion: M_crit = c_ext / sqrt(eps_c) -- the upper mass of
horizonless objects as a function of the vacuum-transition threshold eps_c.

Observational context (borrowed, references in SOURCES.md):
- maximum NS mass ~2.2-2.5 M_sun (pulsars);
- the "mass gap" ~2.5-5 M_sun is nearly empty; candidates: GW190814 (2.6),
  GW230529 (2.5-4.5), PSR J0514-4002E (2.1-2.7);
- Schwarzschild BHs are observed mostly at M > 5 M_sun.

Computed:
 1) M_crit(eps_c) for eps_c = 1, 4.7, 10, 22 x nuclear saturation density;
 2) inverse problem: which eps_c corresponds to M_crit = 2.3, 2.6, 4.5, 5 M_sun;
 3) conversion to g/cm^3.
Units: G=c=1; geometric eps [1/km^2] = G rho / c^2.
Nuclear saturation density: 2.8e14 g/cm^3 (literature value, Lattimer-Prakash
already in SOURCES.md). M_sun = 1.4767 km.

Caveats: the c_ext form factor depends on the profile (range x2); the
scarcity of gap objects may be astrophysical (formation-related) rather
than an EOS effect; the tidal deformability of our branch was not computed
(needs a separate step); the stage-4 criterion was derived for the family
of n-profiles -- carrying it over to other families is assumed, not proven.
"""

import csv
import io
import json
from pathlib import Path
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


# Constants (SI) and conversion.
G_SI = 6.67430e-11
C_SI = 299792458.0
MSUN_KM = 1.47670e0  # GM_sun/c^2, km
RHO_NUC_G_CM3 = 2.8e14  # nuclear saturation density, g/cm^3

# Geometric eps [km^-2] from density [g/cm^3].
RHO_SI_PER_GCM3 = 1.0e3  # 1 g/cm^3 = 1e3 kg/m^3
KM2_PER_M2 = 1.0e6  # 1 m^-2 = 1e6 km^-2


def rho_to_eps_km(rho_g_cm3):
    rho_si = rho_g_cm3 * RHO_SI_PER_GCM3  # kg/m^3
    eps_m2 = G_SI * rho_si / C_SI**2  # 1/m^2  (= G rho / c^2)
    return eps_m2 * KM2_PER_M2


def eps_to_rho_g_cm3(eps_km2):
    eps_m2 = eps_km2 / KM2_PER_M2
    rho_si = eps_m2 * C_SI**2 / G_SI
    return rho_si / RHO_SI_PER_GCM3


EPS_NUC_KM2 = rho_to_eps_km(RHO_NUC_G_CM3)
C_EXT = dict(lo=0.18, n6=0.2314, hi=0.39)


def mcrit_msun(eps_km2, c_ext):
    return (c_ext / eps_km2**0.5) / MSUN_KM


def eps_for_mcrit(m_msun, c_ext):
    return (c_ext / (m_msun * MSUN_KM))**2


def main():
    rows = []
    # 1) Forward problem.
    for mult in (1.0, 2.0, 4.7, 10.0, 22.0):
        eps = EPS_NUC_KM2 * mult
        rows.append(dict(direction="forward", label=f"eps_c = {mult:g} x nuclear",
                         eps_km2=eps, rho_g_cm3=eps_to_rho_g_cm3(eps),
                         M_crit_lo=mcrit_msun(eps, C_EXT["lo"]),
                         M_crit_n6=mcrit_msun(eps, C_EXT["n6"]),
                         M_crit_hi=mcrit_msun(eps, C_EXT["hi"])))
    # 2) Inversion using anchor masses of the observational gap.
    anchors = [("NS max ~2.3", 2.3), ("GW190814 2.6", 2.6),
               ("GW230529 ~3.6 (center)", 3.6), ("gap top ~5.0", 5.0)]
    for label, m in anchors:
        eps_n6 = eps_for_mcrit(m, C_EXT["n6"])
        rows.append(dict(direction="inverse", label=label,
                         eps_km2=eps_n6, rho_g_cm3=eps_to_rho_g_cm3(eps_n6),
                         rho_over_nuclear=eps_to_rho_g_cm3(eps_n6) / RHO_NUC_G_CM3,
                         M_crit_lo=mcrit_msun(eps_for_mcrit(m, C_EXT["lo"]), C_EXT["lo"]),
                         M_crit_n6=m,
                         M_crit_hi=mcrit_msun(eps_for_mcrit(m, C_EXT["hi"]), C_EXT["hi"])))

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()),
                            lineterminator="\r\n", extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    safe_path(DATA / "massgap_inversion.csv").write_text(
        buf.getvalue(), encoding="utf-8", newline="")
    summary = dict(
        eps_nuclear_km2=EPS_NUC_KM2,
        rho_nuclear_g_cm3=RHO_NUC_G_CM3,
        c_ext=C_EXT,
        msun_km=MSUN_KM,
        rows=rows,
        note="the c_ext form factor gives a factor-2 range; the scarcity of gap "
             "objects may result from formation rather than the EOS; Lambda was not computed")
    safe_path(DATA / "massgap_inversion.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"eps_nuclear = {EPS_NUC_KM2:.4e} km^-2 "
          f"({RHO_NUC_G_CM3:.2e} g/cm^3)")
    for r in rows:
        if r["direction"] == "forward":
            print(f"{r['label']:24s}: M_crit = {r['M_crit_lo']:6.2f} – "
                  f"{r['M_crit_hi']:6.2f} M_sun (n=6: {r['M_crit_n6']:6.2f})")
        else:
            print(f"inverse {r['label']:22s}: eps_c = "
                  f"{r.get('rho_over_nuclear', float('nan')):6.1f} x nuclear "
                  f"({r['rho_g_cm3']:.2e} g/cm^3)")


if __name__ == "__main__":
    main()
