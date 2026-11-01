#!/bin/bash
# NAMD MDFF on the Amber topology
# Project mixeddl, step 02 - written by MDATK 0.0.11 on 2026-10-31.
# MDATK only writes this script; review it, then run it with: bash mixeddl_02_namd-mdff_20261029-1200.sh
# NAMD reads the Amber parm7 natively (amber on): CP / L-RNA types and dihedrals, NBfix19 / 0BPh /
# Mg2+ pair terms and HMR masses are used exactly as built. Restraints are MDATK's L-RNA-safe ones;
# never add VMD's chirality / ssrestraints / cispeptide restraints (they assume D-nucleotides).
# Please cite (NAMD, Molecular dynamics flexible fitting (MDFF), CPPTRAJ, AmberTools):
#   - Phillips J. C., Hardy D. J., Maia J. D. C., Stone J. E., Ribeiro J. V., Bernardi R. C., Buch
#     R., Fiorin G., Hénin J., Jiang W., et al. J. Chem. Phys. 2020, 153, 044130.
#     doi:10.1063/5.0014475
#   - Trabuco L. G., Villa E., Mitra K., Frank J., Schulten K. Structure 2008, 16, 673-683.
#     doi:10.1016/j.str.2008.03.005
#   - Roe D. R., Cheatham T. E. J. Chem. Theory Comput. 2013, 9, 3084-3095. doi:10.1021/ct400341p
#   - Case D. A., Aktulga H. M., Belfon K., Cerutti D. S., Cisneros G. A., Cruzeiro V. W. D.,
#     Forouzesh N., Giese T. J., Götz A. W., Gohlke H., et al. J. Chem. Inf. Model. 2023, 63,
#     6183-6191. doi:10.1021/acs.jcim.3c01153

set -eo pipefail
cd "$(dirname "$0")"

NAMD="namd3 +p8"

# Coordinates as ASCII Amber restart (pmemd writes NetCDF restarts: converted with cpptraj)
if python3 mixeddl_02_mdff_20261029-1200.py is-netcdf ../01_amber-build/mixeddl_01_tleap_20261029-1200.rst7; then
    cpptraj -p ../01_amber-build/mixeddl_01_parmed_20261029-1200.parm7 -y ../01_amber-build/mixeddl_01_tleap_20261029-1200.rst7 -x mixeddl_02_start_20261029-1200.rst7 > mixeddl_02_start_20261029-1200.log
else
    python3 mixeddl_02_mdff_20261029-1200.py coordinates ../01_amber-build/mixeddl_01_tleap_20261029-1200.rst7
fi
[ -s "mixeddl_02_start_20261029-1200.rst7" ] || { echo "ERROR: the coordinate conversion did not write mixeddl_02_start_20261029-1200.rst7" >&2; exit 1; }

# Checks, MDFF potential, coupling and restraint files, periodic cell
python3 mixeddl_02_mdff_20261029-1200.py prep | tee mixeddl_02_prep_20261029-1200.log
[ -s "mixeddl_02_potential_20261029-1200.dx" ] || { echo "ERROR: the MDFF preparation did not write mixeddl_02_potential_20261029-1200.dx" >&2; exit 1; }

# Same energies in sander and NAMD? (else NAMD does not reproduce the topology's force field)
sander -O -i mixeddl_02_energy_20261029-1200.mdin -o mixeddl_02_energy_20261029-1200.mdout -p ../01_amber-build/mixeddl_01_parmed_20261029-1200.parm7 -c mixeddl_02_start_20261029-1200.rst7 -r mixeddl_02_energy_20261029-1200.rst7 -inf mixeddl_02_energy_20261029-1200.mdinfo || { echo "ERROR: sander failed; see mixeddl_02_energy_20261029-1200.mdout" >&2; exit 1; }
$NAMD mixeddl_02_energy_20261029-1200.namd > mixeddl_02_energy_20261029-1200.namd.log 2>&1 || { echo "ERROR: NAMD failed on the Amber topology; see mixeddl_02_energy_20261029-1200.namd.log" >&2; exit 1; }
python3 mixeddl_02_energy_20261029-1200.py mixeddl_02_energy_20261029-1200.mdout mixeddl_02_energy_20261029-1200.namd.log | tee mixeddl_02_energy-check_20261029-1200.log

# Finished runs are skipped only if they ran from these inputs and the same stage files
KEY=$( { sed '/^%VERSION/d' ../01_amber-build/mixeddl_01_parmed_20261029-1200.parm7; cat mixeddl_02_start_20261029-1200.rst7 mixeddl_02_prep_20261029-1200.json mixeddl_02_potential_20261029-1200.dx mixeddl_02_coupling_20261029-1200.pdb mixeddl_02_chirality_20261029-1200.txt mixeddl_02_ssdihedrals_20261029-1200.txt mixeddl_02_basepairs_20261029-1200.txt mixeddl_02_cell_20261029-1200.xsc; } | cksum )
FRESH=0

# min: min 2000 steps, map coupling 0.3
KEY=$( { echo "$KEY"; cat "mixeddl_02_min_20261029-1200.namd"; } | cksum )
if [ "$FRESH" = 0 ] && [ -s "mixeddl_02_min_20261029-1200.coor" ] && grep -q "WallClock:" "mixeddl_02_min_20261029-1200.log" 2>/dev/null && [ "$(cat "mixeddl_02_min_20261029-1200.key" 2>/dev/null)" = "$KEY" ]; then
    echo "min already finished - skipped"
else
    FRESH=1
    rm -f "mixeddl_02_min_20261029-1200.key"
    $NAMD mixeddl_02_min_20261029-1200.namd > mixeddl_02_min_20261029-1200.log 2>&1 || { echo "ERROR: NAMD failed on mixeddl_02_min_20261029-1200.namd; see mixeddl_02_min_20261029-1200.log" >&2; exit 1; }
    grep -q "WallClock:" "mixeddl_02_min_20261029-1200.log" || { echo "ERROR: NAMD did not finish mixeddl_02_min_20261029-1200.namd; see mixeddl_02_min_20261029-1200.log" >&2; exit 1; }
    echo "$KEY" > "mixeddl_02_min_20261029-1200.key"
fi

# ramp-001: md 10000 steps, map coupling 0.05
KEY=$( { echo "$KEY"; cat "mixeddl_02_ramp-001_20261029-1200.namd"; } | cksum )
if [ "$FRESH" = 0 ] && [ -s "mixeddl_02_ramp-001_20261029-1200.coor" ] && grep -q "WallClock:" "mixeddl_02_ramp-001_20261029-1200.log" 2>/dev/null && [ "$(cat "mixeddl_02_ramp-001_20261029-1200.key" 2>/dev/null)" = "$KEY" ]; then
    echo "ramp-001 already finished - skipped"
else
    FRESH=1
    rm -f "mixeddl_02_ramp-001_20261029-1200.key"
    $NAMD mixeddl_02_ramp-001_20261029-1200.namd > mixeddl_02_ramp-001_20261029-1200.log 2>&1 || { echo "ERROR: NAMD failed on mixeddl_02_ramp-001_20261029-1200.namd; see mixeddl_02_ramp-001_20261029-1200.log" >&2; exit 1; }
    grep -q "WallClock:" "mixeddl_02_ramp-001_20261029-1200.log" || { echo "ERROR: NAMD did not finish mixeddl_02_ramp-001_20261029-1200.namd; see mixeddl_02_ramp-001_20261029-1200.log" >&2; exit 1; }
    echo "$KEY" > "mixeddl_02_ramp-001_20261029-1200.key"
fi

# ramp-002: md 10000 steps, map coupling 0.1125
KEY=$( { echo "$KEY"; cat "mixeddl_02_ramp-002_20261029-1200.namd"; } | cksum )
if [ "$FRESH" = 0 ] && [ -s "mixeddl_02_ramp-002_20261029-1200.coor" ] && grep -q "WallClock:" "mixeddl_02_ramp-002_20261029-1200.log" 2>/dev/null && [ "$(cat "mixeddl_02_ramp-002_20261029-1200.key" 2>/dev/null)" = "$KEY" ]; then
    echo "ramp-002 already finished - skipped"
else
    FRESH=1
    rm -f "mixeddl_02_ramp-002_20261029-1200.key"
    $NAMD mixeddl_02_ramp-002_20261029-1200.namd > mixeddl_02_ramp-002_20261029-1200.log 2>&1 || { echo "ERROR: NAMD failed on mixeddl_02_ramp-002_20261029-1200.namd; see mixeddl_02_ramp-002_20261029-1200.log" >&2; exit 1; }
    grep -q "WallClock:" "mixeddl_02_ramp-002_20261029-1200.log" || { echo "ERROR: NAMD did not finish mixeddl_02_ramp-002_20261029-1200.namd; see mixeddl_02_ramp-002_20261029-1200.log" >&2; exit 1; }
    echo "$KEY" > "mixeddl_02_ramp-002_20261029-1200.key"
fi

# ramp-003: md 10000 steps, map coupling 0.175
KEY=$( { echo "$KEY"; cat "mixeddl_02_ramp-003_20261029-1200.namd"; } | cksum )
if [ "$FRESH" = 0 ] && [ -s "mixeddl_02_ramp-003_20261029-1200.coor" ] && grep -q "WallClock:" "mixeddl_02_ramp-003_20261029-1200.log" 2>/dev/null && [ "$(cat "mixeddl_02_ramp-003_20261029-1200.key" 2>/dev/null)" = "$KEY" ]; then
    echo "ramp-003 already finished - skipped"
else
    FRESH=1
    rm -f "mixeddl_02_ramp-003_20261029-1200.key"
    $NAMD mixeddl_02_ramp-003_20261029-1200.namd > mixeddl_02_ramp-003_20261029-1200.log 2>&1 || { echo "ERROR: NAMD failed on mixeddl_02_ramp-003_20261029-1200.namd; see mixeddl_02_ramp-003_20261029-1200.log" >&2; exit 1; }
    grep -q "WallClock:" "mixeddl_02_ramp-003_20261029-1200.log" || { echo "ERROR: NAMD did not finish mixeddl_02_ramp-003_20261029-1200.namd; see mixeddl_02_ramp-003_20261029-1200.log" >&2; exit 1; }
    echo "$KEY" > "mixeddl_02_ramp-003_20261029-1200.key"
fi

# ramp-004: md 10000 steps, map coupling 0.2375
KEY=$( { echo "$KEY"; cat "mixeddl_02_ramp-004_20261029-1200.namd"; } | cksum )
if [ "$FRESH" = 0 ] && [ -s "mixeddl_02_ramp-004_20261029-1200.coor" ] && grep -q "WallClock:" "mixeddl_02_ramp-004_20261029-1200.log" 2>/dev/null && [ "$(cat "mixeddl_02_ramp-004_20261029-1200.key" 2>/dev/null)" = "$KEY" ]; then
    echo "ramp-004 already finished - skipped"
else
    FRESH=1
    rm -f "mixeddl_02_ramp-004_20261029-1200.key"
    $NAMD mixeddl_02_ramp-004_20261029-1200.namd > mixeddl_02_ramp-004_20261029-1200.log 2>&1 || { echo "ERROR: NAMD failed on mixeddl_02_ramp-004_20261029-1200.namd; see mixeddl_02_ramp-004_20261029-1200.log" >&2; exit 1; }
    grep -q "WallClock:" "mixeddl_02_ramp-004_20261029-1200.log" || { echo "ERROR: NAMD did not finish mixeddl_02_ramp-004_20261029-1200.namd; see mixeddl_02_ramp-004_20261029-1200.log" >&2; exit 1; }
    echo "$KEY" > "mixeddl_02_ramp-004_20261029-1200.key"
fi

# ramp-005: md 10000 steps, map coupling 0.3
KEY=$( { echo "$KEY"; cat "mixeddl_02_ramp-005_20261029-1200.namd"; } | cksum )
if [ "$FRESH" = 0 ] && [ -s "mixeddl_02_ramp-005_20261029-1200.coor" ] && grep -q "WallClock:" "mixeddl_02_ramp-005_20261029-1200.log" 2>/dev/null && [ "$(cat "mixeddl_02_ramp-005_20261029-1200.key" 2>/dev/null)" = "$KEY" ]; then
    echo "ramp-005 already finished - skipped"
else
    FRESH=1
    rm -f "mixeddl_02_ramp-005_20261029-1200.key"
    $NAMD mixeddl_02_ramp-005_20261029-1200.namd > mixeddl_02_ramp-005_20261029-1200.log 2>&1 || { echo "ERROR: NAMD failed on mixeddl_02_ramp-005_20261029-1200.namd; see mixeddl_02_ramp-005_20261029-1200.log" >&2; exit 1; }
    grep -q "WallClock:" "mixeddl_02_ramp-005_20261029-1200.log" || { echo "ERROR: NAMD did not finish mixeddl_02_ramp-005_20261029-1200.namd; see mixeddl_02_ramp-005_20261029-1200.log" >&2; exit 1; }
    echo "$KEY" > "mixeddl_02_ramp-005_20261029-1200.key"
fi

# fit: md 250000 steps, map coupling 0.3
KEY=$( { echo "$KEY"; cat "mixeddl_02_fit_20261029-1200.namd"; } | cksum )
if [ "$FRESH" = 0 ] && [ -s "mixeddl_02_fit_20261029-1200.coor" ] && grep -q "WallClock:" "mixeddl_02_fit_20261029-1200.log" 2>/dev/null && [ "$(cat "mixeddl_02_fit_20261029-1200.key" 2>/dev/null)" = "$KEY" ]; then
    echo "fit already finished - skipped"
else
    FRESH=1
    rm -f "mixeddl_02_fit_20261029-1200.key"
    $NAMD mixeddl_02_fit_20261029-1200.namd > mixeddl_02_fit_20261029-1200.log 2>&1 || { echo "ERROR: NAMD failed on mixeddl_02_fit_20261029-1200.namd; see mixeddl_02_fit_20261029-1200.log" >&2; exit 1; }
    grep -q "WallClock:" "mixeddl_02_fit_20261029-1200.log" || { echo "ERROR: NAMD did not finish mixeddl_02_fit_20261029-1200.namd; see mixeddl_02_fit_20261029-1200.log" >&2; exit 1; }
    echo "$KEY" > "mixeddl_02_fit_20261029-1200.key"
fi

# final-min: min 2000 steps, map coupling 10
KEY=$( { echo "$KEY"; cat "mixeddl_02_final-min_20261029-1200.namd"; } | cksum )
if [ "$FRESH" = 0 ] && [ -s "mixeddl_02_final-min_20261029-1200.coor" ] && grep -q "WallClock:" "mixeddl_02_final-min_20261029-1200.log" 2>/dev/null && [ "$(cat "mixeddl_02_final-min_20261029-1200.key" 2>/dev/null)" = "$KEY" ]; then
    echo "final-min already finished - skipped"
else
    FRESH=1
    rm -f "mixeddl_02_final-min_20261029-1200.key"
    $NAMD mixeddl_02_final-min_20261029-1200.namd > mixeddl_02_final-min_20261029-1200.log 2>&1 || { echo "ERROR: NAMD failed on mixeddl_02_final-min_20261029-1200.namd; see mixeddl_02_final-min_20261029-1200.log" >&2; exit 1; }
    grep -q "WallClock:" "mixeddl_02_final-min_20261029-1200.log" || { echo "ERROR: NAMD did not finish mixeddl_02_final-min_20261029-1200.namd; see mixeddl_02_final-min_20261029-1200.log" >&2; exit 1; }
    echo "$KEY" > "mixeddl_02_final-min_20261029-1200.key"
fi

# The fitted model (solute only) in the map's frame
python3 mixeddl_02_mdff_20261029-1200.py final
[ -s "mixeddl_02_fitted_20261029-1200.pdb" ] || { echo "ERROR: the final conversion did not write mixeddl_02_fitted_20261029-1200.pdb" >&2; exit 1; }
echo "Fitted model: mixeddl_02_fitted_20261029-1200.pdb"
