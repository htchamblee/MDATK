"""Optimised 12-6 Mg²⁺ parameters with Mg²⁺-RNA pair terms (microMg / nanoMg).

Grotz, Cruz-León and Schwierz optimised plain 12-6 Lennard-Jones parameters for
Mg²⁺ that reproduce solvation free energies, ion pairing and water exchange
without the 12-6-4 C4 term, so they also work in GROMACS:

* microMg: water exchange on the experimental (microsecond) time scale;
* nanoMg: accelerated (nanosecond) water exchange for better sampling of
  ion binding.

Mg²⁺-water uses the Lorentz-Berthelot rule.  Mg²⁺-RNA and Mg²⁺-Cl⁻ use explicit
pair terms (scaled combination rules, the same scaling for every RNA atom
type of parmBSC0χOL3) tuned to phosphate binding affinities.

Every number below is copied verbatim from the authors' GROMACS parameter files
(sigma in nm, epsilon in kJ/mol; ``[ atomtypes ]`` for the ion, ``[ nonbond_params ]``
for the pairs) at the commits named in ``source``:

* TIP3P: Grotz, Cruz-León & Schwierz, J. Chem. Theory Comput. 17, 2530 (2021),
  doi:10.1021/acs.jctc.0c01281
* SPC/E, TIP3P-FB, TIP4P/2005, TIP4P-Ew, TIP4P-D: Grotz & Schwierz, J. Chem. Theory
  Comput. 18, 526 (2022), doi:10.1021/acs.jctc.1c00791.  For TIP4P/2005 the paper's
  Table 1 repeats the microMg Mg-OP pair (0.2217 nm / 6.6266 kJ/mol) in the nanoMg
  row; its own lambda factors and the parameter file give 0.2196970595 / 7.62567188774,
  which is what is used here
* OPC (the SPC/E values, shown to transfer): Grotz & Schwierz, J. Chem. Phys.
  156, 114501 (2022), doi:10.1063/5.0087292

Where the papers' tables and the parameter files differ, the files are right: N. Schwierz
confirmed (personal communication to the MDATK author, 2026) that the files match the published
scaling factors and the tables carry transcription errors - TIP4P-D microMg sigma(Mg-OP) is
0.2598 nm (lambda_sigma = 1.3255; Table 1 prints 0.2197 nm, similarly for nanoMg), TIP3P nanoMg
sigma(Mg-OP) 0.2279 nm (not 0.2277) and OPC microMg sigma(Mg-Cl) 0.4318 nm (not 0.4316).

The authors publish no Amber files; MDATK converts the values (Rmin/2 = 2^(1/6) sigma / 2, Rmin_ij = 2^(1/6)
sigma_ij, 1 kcal = 4.184 kJ) into a frcmod for the ion and ParmEd changeLJPair
lines for the pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

KJ_PER_KCAL = 4.184
TWO_SIXTH = 2.0 ** (1.0 / 6.0)

MODELS = {"micro": "microMg", "nano": "nanoMg"}
CITATIONS = {
    "tip3p": "Grotz, Cruz-Leon & Schwierz, JCTC 17, 2530 (2021), doi:10.1021/acs.jctc.0c01281",
    "spce": "Grotz & Schwierz, JCTC 18, 526 (2022), doi:10.1021/acs.jctc.1c00791",
    "opc": "Grotz & Schwierz, J. Chem. Phys. 156, 114501 (2022), doi:10.1063/5.0087292",
    "fb3": "Grotz & Schwierz, JCTC 18, 526 (2022), doi:10.1021/acs.jctc.1c00791",
    "tip4pew": "Grotz & Schwierz, JCTC 18, 526 (2022), doi:10.1021/acs.jctc.1c00791",
    "tip4pd": "Grotz & Schwierz, JCTC 18, 526 (2022), doi:10.1021/acs.jctc.1c00791",
    "tip4p2005": "Grotz & Schwierz, JCTC 18, 526 (2022), doi:10.1021/acs.jctc.1c00791",
}
# citation resource (mdatk.citations) of each published set
RESOURCES = {(m, w): ("mg_micro_tip3p" if m == "micro" else "mg_nano_tip3p") if w == "tip3p"
             else "mg_opc" if w == "opc" else "mg_tip4p_family"
             for m in MODELS for w in ("tip3p", "spce", "opc", "fb3", "tip4pew", "tip4pd", "tip4p2005")}
# (model, water) combinations without a published set, with the reason shown to the user
UNAVAILABLE: Dict[Tuple[str, str], str] = {}


@dataclass(frozen=True)
class MgSet:
    ion: Tuple[float, float]  # Mg2+ sigma (nm), epsilon (kJ/mol)
    chloride: str  # the Cl- model the set was optimised with
    chloride_lj: Tuple[float, float]  # its sigma, epsilon
    mg_cl: Tuple[float, float]  # Mg2+-Cl- pair sigma, epsilon
    rna: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # Amber type -> pair
    source: str = ""

    @property
    def rmin_half(self) -> float:
        """Amber Rmin/2 of the ion (Å)."""
        return self.ion[0] * 10.0 * TWO_SIXTH / 2.0

    @property
    def eps_kcal(self) -> float:
        return self.ion[1] / KJ_PER_KCAL


def amber_pair(sigma_nm: float, eps_kj: float) -> Tuple[float, float]:
    """(Rmin_ij in Å, epsilon_ij in kcal/mol) of a GROMACS pair."""
    return sigma_nm * 10.0 * TWO_SIXTH, eps_kj / KJ_PER_KCAL


MG_SETS: Dict[Tuple[str, str], MgSet] = {
    ("micro", "tip3p"): MgSet(
        ion=(0.101934, 235.804213166223),
        chloride="Mamatkulov-Schwierz (CXY)", chloride_lj=(0.440939, 0.283829),
        mg_cl=(0.48782511, 0.8180958),
        rna={
            "C": (0.25133119375, 2.94761543562), "C4": (0.25133119375, 2.94761543562),
            "C5": (0.25133119375, 2.94761543562), "CA": (0.25133119375, 2.94761543562),
            "CB": (0.25133119375, 2.94761543562), "CI": (0.25133119375, 3.32453146161),
            "CP": (0.25133119375, 2.94761543562), "CQ": (0.25133119375, 2.94761543562),
            "CS": (0.25133119375, 2.94761543562), "CT": (0.25133119375, 3.32453146161),
            "H": (0.1187788875, 1.25942242703), "H1": (0.19853299375, 1.25942242703),
            "H2": (0.18839900625, 1.25942242703), "H4": (0.20076249375, 1.2310260012),
            "H5": (0.1956955, 1.2310260012), "HA": (0.2058294875, 1.2310260012),
            "HO": (0.0579749625, 0.0), "N*": (0.2428187125, 4.14425105946),
            "N2": (0.2428187125, 4.14425105946), "NA": (0.2428187125, 4.14425105946),
            "NB": (0.2428187125, 4.14425105946), "NC": (0.2428187125, 4.14425105946),
            "O": (0.2263204125, 4.60607753071), "O2": (0.2263204125, 4.60607753071),
            "OH": (0.23238044375, 4.61046323173), "OS": (0.22860053125, 4.14425105946),
            "P": (0.27078813125, 4.49507139822),
        },
        source="https://github.com/bio-phys/Magnesium-FFs/blob/e71379d1a88b54221018af53ca54d3c18066ba05/ff_Mg.itp"),
    ("micro", "spce"): MgSet(
        ion=(0.103643, 290.584188),
        chloride="Smith-Dang (Cl)", chloride_lj=(0.439443, 0.4155978),
        mg_cl=(0.431753, 1.0989365),
        rna={
            "C": (0.2444069295, 4.96546278106), "C4": (0.2444069295, 4.96546278106),
            "C5": (0.2444069295, 4.96546278106), "CA": (0.2444069295, 4.96546278106),
            "CB": (0.2444069295, 4.96546278106), "CI": (0.2444069295, 5.60040398677),
            "CP": (0.2444069295, 4.96546278106), "CQ": (0.2444069295, 4.96546278106),
            "CS": (0.2444069295, 4.96546278106), "CT": (0.2444069295, 5.60040398677),
            "H": (0.11600307345, 2.12158448876), "H1": (0.1932611391, 2.12158448876),
            "H2": (0.183444312, 2.12158448876), "H4": (0.1954208631, 2.0737487386),
            "H5": (0.19051244955, 2.0737487386), "HA": (0.20032927665, 2.0737487386),
            "HO": (0.05710211085, 0.0), "N*": (0.23616086085, 6.98127854213),
            "N2": (0.23616086085, 6.98127854213), "NA": (0.23616086085, 6.98127854213),
            "NB": (0.23616086085, 6.98127854213), "NC": (0.23616086085, 6.98127854213),
            "O": (0.22017890325, 7.75925728609), "O2": (0.22017890325, 7.75925728609),
            "OH": (0.2260492755, 7.76664530386), "OS": (0.2223876618, 6.98127854213),
            "P": (0.263254929, 7.57225975152),
        },
        source="https://github.com/bio-phys/optimizedMgFFs/blob/1df53e9c94fc14a97864ac35c6951085bb60d1c0/ff_Mg_spce.itp"),
    ("micro", "opc"): MgSet(
        ion=(0.103643, 290.584188),
        chloride="Smith-Dang (Cl)", chloride_lj=(0.439443, 0.4155978),
        mg_cl=(0.431753, 1.0989365),
        rna={
            "C": (0.2444069295, 4.96546278106), "C4": (0.2444069295, 4.96546278106),
            "C5": (0.2444069295, 4.96546278106), "CA": (0.2444069295, 4.96546278106),
            "CB": (0.2444069295, 4.96546278106), "CI": (0.2444069295, 5.60040398677),
            "CP": (0.2444069295, 4.96546278106), "CQ": (0.2444069295, 4.96546278106),
            "CS": (0.2444069295, 4.96546278106), "CT": (0.2444069295, 5.60040398677),
            "H": (0.11600307345, 2.12158448876), "H1": (0.1932611391, 2.12158448876),
            "H2": (0.183444312, 2.12158448876), "H4": (0.1954208631, 2.0737487386),
            "H5": (0.19051244955, 2.0737487386), "HA": (0.20032927665, 2.0737487386),
            "HO": (0.05710211085, 0.0), "N*": (0.23616086085, 6.98127854213),
            "N2": (0.23616086085, 6.98127854213), "NA": (0.23616086085, 6.98127854213),
            "NB": (0.23616086085, 6.98127854213), "NC": (0.23616086085, 6.98127854213),
            "O": (0.22017890325, 7.75925728609), "O2": (0.22017890325, 7.75925728609),
            "OH": (0.2260492755, 7.76664530386), "OS": (0.2223876618, 6.98127854213),
            "P": (0.263254929, 7.57225975152),
        },
        source="https://github.com/bio-phys/MgFF_OPC/blob/cf8148a3131c198b19f3a2896b68bad6f6f62f5b/ff_Mg_opc.itp"),
    ("micro", "fb3"): MgSet(
        ion=(0.103203543645, 311.377007),
        chloride="co-optimised with Mg2+ (CXY)", chloride_lj=(0.493358, 0.050960),
        mg_cl=(0.474266, 0.398343),
        rna={
            "C": (0.242790982336, 5.20038021817), "C4": (0.242790982336, 5.20038021817),
            "C5": (0.242790982336, 5.20038021817), "CA": (0.242790982336, 5.20038021817),
            "CB": (0.242790982336, 5.20038021817), "CI": (0.242790982336, 5.86536067043),
            "CP": (0.242790982336, 5.20038021817), "CQ": (0.242790982336, 5.20038021817),
            "CS": (0.242790982336, 5.20038021817), "CT": (0.242790982336, 5.86536067043),
            "H": (0.115109609186, 2.22195724608), "H1": (0.191932971136, 2.22195724608),
            "H2": (0.182171379836, 2.22195724608), "H4": (0.194080543136, 2.17185837316),
            "H5": (0.189199747486, 2.17185837316), "HA": (0.198961338786, 2.17185837316),
            "HO": (0.0565400613859, 0.0), "N*": (0.234591311386, 7.31156478837),
            "N2": (0.234591311386, 7.31156478837), "NA": (0.234591311386, 7.31156478837),
            "NB": (0.234591311386, 7.31156478837), "NC": (0.234591311386, 7.31156478837),
            "O": (0.218699278586, 8.12634992495), "O2": (0.218699278586, 8.12634992495),
            "OH": (0.224536620336, 8.13408747191), "OS": (0.220895609236, 7.31156478837),
            "P": (0.261532930836, 7.93050548456),
        },
        source="https://github.com/bio-phys/optimizedMgFFs/blob/1df53e9c94fc14a97864ac35c6951085bb60d1c0/ff_Mg_tip3pfb.itp"),
    ("micro", "tip4pew"): MgSet(
        ion=(0.090965, 647.628446),
        chloride="co-optimised with Mg2+ (CXY)", chloride_lj=(0.498225, 0.048805),
        mg_cl=(0.468406, 0.562205),
        rna={
            "C": (0.2468378496, 4.2407264894), "C4": (0.2468378496, 4.2407264894),
            "C5": (0.2468378496, 4.2407264894), "CA": (0.2468378496, 4.2407264894),
            "CB": (0.2468378496, 4.2407264894), "CI": (0.2468378496, 4.78299457376),
            "CP": (0.2468378496, 4.2407264894), "CQ": (0.2468378496, 4.2407264894),
            "CS": (0.2468378496, 4.2407264894), "CT": (0.2468378496, 4.78299457376),
            "H": (0.1133416544, 1.81192769691), "H1": (0.19366368, 1.81192769691),
            "H2": (0.1834575296, 1.81192769691), "H4": (0.195909056, 1.77107383458),
            "H5": (0.1908059808, 1.77107383458), "HA": (0.2010121312, 1.77107383458),
            "HO": (0.052104752, 0.0), "N*": (0.238264752, 5.96232297952),
            "N2": (0.238264752, 5.96232297952), "NA": (0.238264752, 5.96232297952),
            "NB": (0.238264752, 5.96232297952), "NC": (0.238264752, 5.96232297952),
            "O": (0.2216489696, 6.62675149568), "O2": (0.2216489696, 6.62675149568),
            "OH": (0.2277521536, 6.63306119209), "OS": (0.2239453248, 5.96232297952),
            "P": (0.2664333376, 6.46704726805),
        },
        source="https://github.com/bio-phys/optimizedMgFFs/blob/1df53e9c94fc14a97864ac35c6951085bb60d1c0/ff_Mg_tip4pew.itp"),
    ("nano", "tip3p"): MgSet(
        ion=(0.102534, 389.798801356),
        chloride="Mamatkulov-Schwierz (CXY)", chloride_lj=(0.440939, 0.283829),
        mg_cl=(0.4883651, 1.0518375),
        rna={
            "C": (0.25299994675, 2.9607744331), "C4": (0.25299994675, 2.9607744331),
            "C5": (0.25299994675, 2.9607744331), "CA": (0.25299994675, 2.9607744331),
            "CB": (0.25299994675, 2.9607744331), "CI": (0.25299994675, 3.33937311992),
            "CP": (0.25299994675, 2.9607744331), "CQ": (0.25299994675, 2.9607744331),
            "CS": (0.25299994675, 2.9607744331), "CT": (0.25299994675, 3.33937311992),
            "H": (0.1197484635, 1.26504484858), "H1": (0.19992325075, 1.26504484858),
            "H2": (0.18973580925, 1.26504484858), "H4": (0.20216451075, 1.23652165299),
            "H5": (0.19707079, 1.23652165299), "HA": (0.2072582315, 1.23652165299),
            "HO": (0.0586238145, 0.0), "N*": (0.2444425645, 4.16275218026),
            "N2": (0.2444425645, 4.16275218026), "NA": (0.2444425645, 4.16275218026),
            "NB": (0.2444425645, 4.16275218026), "NC": (0.2444425645, 4.16275218026),
            "O": (0.2278572405, 4.62664037682), "O2": (0.2278572405, 4.62664037682),
            "OH": (0.23394923675, 4.63104565687), "OS": (0.23014938625, 4.16275218026),
            "P": (0.27255951425, 4.51513868125),
        },
        source="https://github.com/bio-phys/Magnesium-FFs/blob/e71379d1a88b54221018af53ca54d3c18066ba05/ff_Mg.itp"),
    ("nano", "spce"): MgSet(
        ion=(0.104643, 470.698354),
        chloride="Smith-Dang (Cl)", chloride_lj=(0.439443, 0.4155978),
        mg_cl=(0.432548, 1.3986464),
        rna={
            "C": (0.2469141635, 4.29467538672), "C4": (0.2469141635, 4.29467538672),
            "C5": (0.2469141635, 4.29467538672), "CA": (0.2469141635, 4.29467538672),
            "CB": (0.2469141635, 4.29467538672), "CI": (0.2469141635, 4.84384199784),
            "CP": (0.2469141635, 4.29467538672), "CQ": (0.2469141635, 4.29467538672),
            "CS": (0.2469141635, 4.29467538672), "CT": (0.2469141635, 4.84384199784),
            "H": (0.11748484785, 1.83497834672), "H1": (0.1953599123, 1.83497834672),
            "H2": (0.185464686, 1.83497834672), "H4": (0.1975368843, 1.79360475722),
            "H5": (0.19258927115, 1.79360475722), "HA": (0.20248449745, 1.79360475722),
            "HO": (0.05811349005, 0.0), "N*": (0.23860224005, 6.03817336766),
            "N2": (0.23860224005, 6.03817336766), "NA": (0.23860224005, 6.03817336766),
            "NB": (0.23860224005, 6.03817336766), "NC": (0.23860224005, 6.03817336766),
            "O": (0.22249264725, 6.71105448879), "O2": (0.22249264725, 6.71105448879),
            "OH": (0.2284099015, 6.71744445474), "OS": (0.2247190454, 6.03817336766),
            "P": (0.265912687, 6.54931856517),
        },
        source="https://github.com/bio-phys/optimizedMgFFs/blob/1df53e9c94fc14a97864ac35c6951085bb60d1c0/ff_Mg_spce.itp"),
    ("nano", "opc"): MgSet(
        ion=(0.104643, 470.698354),
        chloride="Smith-Dang (Cl)", chloride_lj=(0.439443, 0.4155978),
        mg_cl=(0.432548, 1.3986464),
        rna={
            "C": (0.2469141635, 4.29467538672), "C4": (0.2469141635, 4.29467538672),
            "C5": (0.2469141635, 4.29467538672), "CA": (0.2469141635, 4.29467538672),
            "CB": (0.2469141635, 4.29467538672), "CI": (0.2469141635, 4.84384199784),
            "CP": (0.2469141635, 4.29467538672), "CQ": (0.2469141635, 4.29467538672),
            "CS": (0.2469141635, 4.29467538672), "CT": (0.2469141635, 4.84384199784),
            "H": (0.11748484785, 1.83497834672), "H1": (0.1953599123, 1.83497834672),
            "H2": (0.185464686, 1.83497834672), "H4": (0.1975368843, 1.79360475722),
            "H5": (0.19258927115, 1.79360475722), "HA": (0.20248449745, 1.79360475722),
            "HO": (0.05811349005, 0.0), "N*": (0.23860224005, 6.03817336766),
            "N2": (0.23860224005, 6.03817336766), "NA": (0.23860224005, 6.03817336766),
            "NB": (0.23860224005, 6.03817336766), "NC": (0.23860224005, 6.03817336766),
            "O": (0.22249264725, 6.71105448879), "O2": (0.22249264725, 6.71105448879),
            "OH": (0.2284099015, 6.71744445474), "OS": (0.2247190454, 6.03817336766),
            "P": (0.265912687, 6.54931856517),
        },
        source="https://github.com/bio-phys/MgFF_OPC/blob/cf8148a3131c198b19f3a2896b68bad6f6f62f5b/ff_Mg_opc.itp"),
    ("nano", "fb3"): MgSet(
        ion=(0.103403543645, 380.380222),
        chloride="co-optimised with Mg2+ (CXY)", chloride_lj=(0.493358, 0.050960),
        mg_cl=(0.474425, 0.440274),
        rna={
            "C": (0.243898136059, 4.88088221116), "C4": (0.243898136059, 4.88088221116),
            "C5": (0.243898136059, 4.88088221116), "CA": (0.243898136059, 4.88088221116),
            "CB": (0.243898136059, 4.88088221116), "CI": (0.243898136059, 5.50500797198),
            "CP": (0.243898136059, 4.88088221116), "CQ": (0.243898136059, 4.88088221116),
            "CS": (0.243898136059, 4.88088221116), "CT": (0.243898136059, 5.50500797198),
            "H": (0.115692380159, 2.08544589845), "H1": (0.192831252859, 2.08544589845),
            "H2": (0.183029571059, 2.08544589845), "H4": (0.194987644859, 2.03842497163),
            "H5": (0.190086803959, 2.03842497163), "HA": (0.199888485759, 2.03842497163),
            "HO": (0.0568822893591, 0.0), "N*": (0.235664789359, 6.8623610225),
            "N2": (0.235664789359, 6.8623610225), "NA": (0.235664789359, 6.8623610225),
            "NB": (0.235664789359, 6.8623610225), "NC": (0.235664789359, 6.8623610225),
            "O": (0.219707488559, 7.6270878525), "O2": (0.219707488559, 7.6270878525),
            "OH": (0.225568804059, 7.63435002445), "OS": (0.221912839459, 6.8623610225),
            "P": (0.262717057059, 7.44327559164),
        },
        source="https://github.com/bio-phys/optimizedMgFFs/blob/1df53e9c94fc14a97864ac35c6951085bb60d1c0/ff_Mg_tip3pfb.itp"),
    ("nano", "tip4pew"): MgSet(
        ion=(0.092565, 760.063940),
        chloride="co-optimised with Mg2+ (CXY)", chloride_lj=(0.498225, 0.048805),
        mg_cl=(0.469678, 0.609055),
        rna={
            "C": (0.2484680074, 3.92104369374), "C4": (0.2484680074, 3.92104369374),
            "C5": (0.2484680074, 3.92104369374), "CA": (0.2484680074, 3.92104369374),
            "CB": (0.2484680074, 3.92104369374), "CI": (0.2484680074, 4.42243345745),
            "CP": (0.2484680074, 3.92104369374), "CQ": (0.2484680074, 3.92104369374),
            "CS": (0.2484680074, 3.92104369374), "CT": (0.2484680074, 4.42243345745),
            "H": (0.11458726485, 1.67533739496), "H1": (0.195140665, 1.67533739496),
            "H2": (0.1849051149, 1.67533739496), "H4": (0.197392509, 1.63756325893),
            "H5": (0.19227473395, 1.63756325893), "HA": (0.20251028405, 1.63756325893),
            "HO": (0.05317396425, 0.0), "N*": (0.23987021425, 5.51285940683),
            "N2": (0.23987021425, 5.51285940683), "NA": (0.23987021425, 5.51285940683),
            "NB": (0.23987021425, 5.51285940683), "NC": (0.23987021425, 5.51285940683),
            "O": (0.22320656865, 6.1272006641), "O2": (0.22320656865, 6.1272006641),
            "OH": (0.2293273334, 6.13303471054), "OS": (0.2255095387, 5.51285940683),
            "P": (0.2681199419, 5.97953557506),
        },
        source="https://github.com/bio-phys/optimizedMgFFs/blob/1df53e9c94fc14a97864ac35c6951085bb60d1c0/ff_Mg_tip4pew.itp"),
    ("micro", "tip4pd"): MgSet(
        ion=(0.096035, 621.504072),
        chloride="co-optimised with Mg2+ (CXY)", chloride_lj=(0.509112, 0.035496),
        mg_cl=(0.481091, 0.469690),
        rna={
            "C": (0.2889603255, 4.88109435372), "C4": (0.2889603255, 4.88109435372),
            "C5": (0.2889603255, 4.88109435372), "CA": (0.2889603255, 4.88109435372),
            "CB": (0.2889603255, 4.88109435372), "CI": (0.2889603255, 5.50524724152),
            "CP": (0.2889603255, 4.88109435372), "CQ": (0.2889603255, 4.88109435372),
            "CS": (0.2889603255, 4.88109435372), "CT": (0.2889603255, 5.50524724152),
            "H": (0.13450047325, 2.08553654022), "H1": (0.2274359175, 2.08553654022),
            "H2": (0.215627038, 2.08553654022), "H4": (0.2300338975, 2.03851356969),
            "H5": (0.22412945775, 2.03851356969), "HA": (0.23593833725, 2.03851356969),
            "HO": (0.06364719625, 0.0), "N*": (0.27904094625, 6.86265928801),
            "N2": (0.27904094625, 6.86265928801), "NA": (0.27904094625, 6.86265928801),
            "NB": (0.27904094625, 6.86265928801), "NC": (0.27904094625, 6.86265928801),
            "O": (0.25981589425, 7.62741935607), "O2": (0.25981589425, 7.62741935607),
            "OH": (0.2668774955, 7.63468184366), "OS": (0.262472859, 6.86265928801),
            "P": (0.311633003, 7.443599106),
        },
        source="https://github.com/bio-phys/optimizedMgFFs/blob/1df53e9c94fc14a97864ac35c6951085bb60d1c0/ff_Mg_tip4pd.itp"),
    ("nano", "tip4pd"): MgSet(
        ion=(0.097035, 680.009868),
        chloride="co-optimised with Mg2+ (CXY)", chloride_lj=(0.509112, 0.035496),
        mg_cl=(0.481886, 0.491300),
        rna={
            "C": (0.2453984731, 4.43304884946), "C4": (0.2453984731, 4.43304884946),
            "C5": (0.2453984731, 4.43304884946), "CA": (0.2453984731, 4.43304884946),
            "CB": (0.2453984731, 4.43304884946), "CI": (0.2453984731, 4.99990948371),
            "CP": (0.2453984731, 4.43304884946), "CQ": (0.2453984731, 4.43304884946),
            "CS": (0.2453984731, 4.43304884946), "CT": (0.2453984731, 4.99990948371),
            "H": (0.11452419165, 1.89410093109), "H1": (0.1932686635, 1.89410093109),
            "H2": (0.1832629656, 1.89410093109), "H4": (0.1954699395, 1.85139429396),
            "H5": (0.19046709055, 1.85139429396), "HA": (0.20047278845, 1.85139429396),
            "HO": (0.05449000425, 0.0), "N*": (0.23699375425, 6.23272193822),
            "N2": (0.23699375425, 6.23272193822), "NA": (0.23699375425, 6.23272193822),
            "NB": (0.23699375425, 6.23272193822), "NC": (0.23699375425, 6.23272193822),
            "O": (0.22070431185, 6.92728313581), "O2": (0.22070431185, 6.92728313581),
            "OH": (0.2266876271, 6.93387898501), "OS": (0.2229555658, 6.23272193822),
            "P": (0.2646090986, 6.76033611757),
        },
        source="https://github.com/bio-phys/optimizedMgFFs/blob/1df53e9c94fc14a97864ac35c6951085bb60d1c0/ff_Mg_tip4pd.itp"),
    ("micro", "tip4p2005"): MgSet(
        ion=(0.09011, 712.674442),
        chloride="co-optimised with Mg2+ (CXY)", chloride_lj=(0.503464, 0.042887),
        mg_cl=(0.471891, 0.552851),
        rna={
            "C": (0.2469502134, 4.24041767915), "C4": (0.2469502134, 4.24041767915),
            "C5": (0.2469502134, 4.24041767915), "CA": (0.2469502134, 4.24041767915),
            "CB": (0.2469502134, 4.24041767915), "CI": (0.2469502134, 4.78264627547),
            "CP": (0.2469502134, 4.24041767915), "CQ": (0.2469502134, 4.24041767915),
            "CS": (0.2469502134, 4.24041767915), "CT": (0.2469502134, 4.78264627547),
            "H": (0.1131277356, 1.8117957521), "H1": (0.193646079, 1.8117957521),
            "H2": (0.1834149834, 1.8117957521), "H4": (0.195896943, 1.77094486475),
            "H5": (0.1907813952, 1.77094486475), "HA": (0.2010124908, 1.77094486475),
            "HO": (0.051741162, 0.0), "N*": (0.238356162, 5.96188880239),
            "N2": (0.238356162, 5.96188880239), "NA": (0.238356162, 5.96188880239),
            "NB": (0.238356162, 5.96188880239), "NC": (0.238356162, 5.96188880239),
            "O": (0.2216997684, 6.62626893478), "O2": (0.2216997684, 6.62626893478),
            "OH": (0.2278178694, 6.63257817171), "OS": (0.2240017362, 5.96188880239),
            "P": (0.2665935954, 6.46657633683),
        },
        source="https://github.com/bio-phys/optimizedMgFFs/blob/1df53e9c94fc14a97864ac35c6951085bb60d1c0/ff_Mg_tip4p2005.itp"),
    ("nano", "tip4p2005"): MgSet(
        ion=(0.09131, 774.618078),
        chloride="co-optimised with Mg2+ (CXY)", chloride_lj=(0.503464, 0.042887),
        mg_cl=(0.472845, 0.576377),
        rna={
            "C": (0.24464187825, 4.879976078), "C4": (0.24464187825, 4.879976078),
            "C5": (0.24464187825, 4.879976078), "CA": (0.24464187825, 4.879976078),
            "CB": (0.24464187825, 4.879976078), "CI": (0.24464187825, 5.50398597021),
            "CP": (0.24464187825, 4.879976078), "CQ": (0.24464187825, 4.879976078),
            "CS": (0.24464187825, 4.879976078), "CT": (0.24464187825, 5.50398597021),
            "H": (0.1124391605, 2.08505873654), "H1": (0.19198292625, 2.08505873654),
            "H2": (0.18187566575, 2.08505873654), "H4": (0.19420654625, 2.03804653914),
            "H5": (0.189152916, 2.03804653914), "HA": (0.1992601765, 2.03804653914),
            "HO": (0.0517955975, 0.0), "N*": (0.2361518475, 6.86108702887),
            "N2": (0.2361518475, 6.86108702887), "NA": (0.2361518475, 6.86108702887),
            "NB": (0.2361518475, 6.86108702887), "NC": (0.2361518475, 6.86108702887),
            "O": (0.2196970595, 7.62567188774), "O2": (0.2196970595, 7.62567188774),
            "OH": (0.22574110825, 7.63293271147), "OS": (0.22197116475, 6.86108702887),
            "P": (0.26404750075, 7.44189375153),
        },
        source="https://github.com/bio-phys/optimizedMgFFs/blob/1df53e9c94fc14a97864ac35c6951085bb60d1c0/ff_Mg_tip4p2005.itp"),
}


def get_set(model: str, water: str) -> MgSet:
    return MG_SETS[(model, water)]


# Atom type names the ion unit may carry: AmberTools' atomic_ions.lib types its MG unit Mg2+, while
# parm10.dat and leaprc.RNA.OL3 know the older type MG. The frcmod defines both, so the published
# ion terms apply whichever the unit carries (the topology check confirms it after the build).
ION_TYPES = ("Mg2+", "MG")
# reference types a non-standard atom type may inherit Mg2+ pair terms from: the parmBSC0chiOL3 types
# of every set, plus the CP phosphate oxygens (OP = O2, OR = OS)
CP_ALIASES = {"OP": "O2", "OR": "OS"}
# Mg2+ pair references of the L-RNA library types (Dudek & Trylska style): bridging phosphate
# oxygens O3'/O5', sugar ring O4', hydroxyl oxygens O2' / terminal OH
LRNA_DEFAULT_REFS = {"LD": "OS", "LE": "OS", "LH": "OH"}


def reference_types(model: str = "micro", water: str = "tip3p") -> List[str]:
    return sorted(set(get_set(model, water).rna) | set(CP_ALIASES))


def frcmod_text(model: str, water: str, water_label: str, chloride: bool = False) -> str:
    """frcmod replacing the Li-Merz 12-6 Mg2+ parameters (atom types Mg2+ and MG); ``chloride`` also
    replaces Cl- by the Cl- model the set was optimised with (used where Amber has no Cl- for the water)."""
    p = get_set(model, water)
    cl_rmin, cl_eps = p.chloride_lj[0] * 10.0 * TWO_SIXTH / 2.0, p.chloride_lj[1] / KJ_PER_KCAL
    return (f"{MODELS[model]} Mg2+ for {water_label} water ({CITATIONS[water]}); "
            f"sigma {p.ion[0]} nm, epsilon {p.ion[1]} kJ/mol"
            + (f"; Cl- {p.chloride}: sigma {p.chloride_lj[0]} nm, epsilon {p.chloride_lj[1]} kJ/mol"
               if chloride else "") + "\n"
            "MASS\n"
            + "".join(f"{t:<4} 24.305\n" for t in ION_TYPES)
            + ("Cl-  35.45\n" if chloride else "") +
            "\n"
            "NONB\n"
            + "".join(f"  {t:<8} {p.rmin_half:.8f}  {p.eps_kcal:.8f}\n" for t in ION_TYPES)
            + (f"  Cl-      {cl_rmin:.8f}  {cl_eps:.8f}\n" if chloride else "") +
            "\n")


def parmed_lines(model: str, water: str, water_label: str, frcmod: str, chloride: bool,
                 extra_types: Optional[Dict[str, str]] = None) -> List[str]:
    """ParmEd changeLJPair lines for the Mg2+-RNA (and Mg2+-Cl-) pair terms; ``extra_types`` maps
    non-standard atom types (e.g. of an L-RNA library) onto the reference type whose terms they get."""
    p = get_set(model, water)
    lines = ["#",
             f"# {MODELS[model]} Mg2+ for {water_label} water: {CITATIONS[water]}",
             f"# The ion's own Lennard-Jones terms come from {frcmod} (Mg2+-water follows the",
             "# Lorentz-Berthelot rule); below are the authors' explicit Mg2+-RNA pair terms for every",
             "# parmBSC0chiOL3 atom type, converted from",
             f"# {p.source}",
             "# (Rmin_ij = 2^(1/6) sigma_ij, epsilon in kcal/mol).  They come after every addLJType,",
             "# which would reset the pair terms of the atoms it moves to a new type."]
    for ty, (sigma, eps) in p.rna.items():
        rmin, e = amber_pair(sigma, eps)
        mask = ty.replace("*", "\\*")  # N* is a type name, not a wildcard
        lines.append(f"changeLJPair :MG @%{mask} {rmin:.8f} {e:.8f}")
    lines.append("# phosphate oxygens retyped by the CP correction (O2 -> OP, OS -> OR) get the O2 / OS terms;")
    lines.append("# ParmEd skips these two lines when no atom has those types")
    for ty, src in CP_ALIASES.items():
        rmin, e = amber_pair(*p.rna[src])
        lines.append(f"changeLJPair :MG @%{ty} {rmin:.8f} {e:.8f}")
    if extra_types:
        lines.append("# non-standard atom types (e.g. an L-RNA library) with the terms of their chemical twin")
        for ty, ref in extra_types.items():
            rmin, e = amber_pair(*p.rna[CP_ALIASES.get(ref, ref)])
            lines += [f"# {ty} as {ref}", f"changeLJPair :MG @%{ty.replace('*', chr(92) + '*')} {rmin:.8f} {e:.8f}"]
    if chloride:
        rmin, e = amber_pair(*p.mg_cl)
        lines += [f"# Mg2+-Cl- pair term (optimised with the {p.chloride} Cl- model)",
                  f"changeLJPair :MG :Cl- {rmin:.8f} {e:.8f}"]
    return lines
