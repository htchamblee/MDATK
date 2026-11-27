# Generic Lennard-Jones parameters of the energy analysis

The *Intermolecular energy* analysis needs Lennard-Jones parameters for every atom. With an Amber **parm7**
topology it uses that topology's own table, exactly (`lj_source: parm7` in the result's metadata). Any other
topology (PDB, GRO, PSF …) only tells an atom's element, so each atom gets the parameters of **one Amber atom type
of its element**, from the tables below (`lj_source: generic`). This file is the record of those tables: every
number, the file it was read from, and the publication behind it. The code is `mdatk/analysis/generic_lj.py`;
`tests/test_generic_lj.py` keeps this file, the code and the source lines quoted in the test identical.

The values are approximations by construction: one atom type per element cannot reproduce a force field. Use a
parm7 topology for energies that are meant to be the force field's.

## Units and combination

Every row is **R\*** = Rmin/2 in Å and the well depth **ε** in kcal/mol, as Amber's parameter files give them.
A pair *i*, *j* at distance *r* has

    Rmin = R*_i + R*_j        ε_ij = sqrt(ε_i ε_j)        (Lorentz-Berthelot, as in Amber)
    E = ε_ij [ (Rmin / r)^12 - 2 (Rmin / r)^6 ]  =  A / r^12 - B / r^6,   A = ε_ij Rmin^12,   B = 2 ε_ij Rmin^6

Distances below 0.5 Å are taken as 0.5 Å. The electrostatic term is 332.0522 q_i q_j / (ε_r r) kcal/mol with the
topology's charges.

## How an atom gets its row

1. **A virtual site** (the extra point of a four- or five-site water model such as TIP4P, a lone pair) has **no
   Lennard-Jones term**: R\* 0 and ε 0, Amber's atom type `EP` (the line `EP 0.00 0.0000 lone pair` of
   `parm10.dat`, section `MOD4 RE`). It is a massless charge site, so its charge still counts in the electrostatic
   term. An atom is taken for one when the topology gives it no element (an Amber extra point has atomic number 0)
   and it is named `EP…` (`EPW`, `EP1`, `EP2`), `LP…`, `M` or `MW`; an atom with an element is never one.
2. **A hydrogen** takes the row of the element it is bonded to (table 2). The bonded atom comes from the
   topology's bonds; where the topology has none for that hydrogen (a PDB without CONECT records, or with them for
   a ligand only), it is the nearest non-hydrogen atom of the hydrogen's residue within 1.45 Å at the
   first analysed frame (in a real structure that atom is 0.96-1.34 Å away and the next one about 2 Å; a virtual
   site is not counted: TIP4P's extra point is nearer to the hydrogens than their oxygen). A hydrogen
   with no such atom takes the row *H on N*: the small radius, which cannot turn a contact repulsive.
3. **An atom that is alone in its residue** and whose element is in table 3 is an ion and takes that row.
4. Otherwise the atom takes the row of its element in table 1.
5. An element that is only in table 3 (a metal bound in a larger residue) takes its ion's row.
6. An element in no table takes **carbon's row**, and the analysis says so in its log
   (`lj_default_elements` in the result's metadata).

The element is the topology's, or is guessed from the atom name where the topology has none.

## Table 1: atoms of molecules

Read from `parm10.dat` (the parameter file of ff10, ff14SB, OL3 and OL15; section `MOD4 RE`). The last column is
the comment of that line in the file, as written there.

| Element | R\* (Å) | ε (kcal/mol) | Amber type | File | Comment in the file |
|---|---|---|---|---|---|
| C | 1.908 | 0.086 | `C` | `parm10.dat` | OPLS |
| N | 1.824 | 0.17 | `N` | `parm10.dat` | OPLS |
| O | 1.6612 | 0.21 | `O` | `parm10.dat` | OPLS |
| P | 2.1 | 0.2 | `P` | `parm10.dat` | JCC,7,(1986),230; |
| S | 2.0 | 0.25 | `S` | `parm10.dat` | W. Cornell CH3SH and CH3SCH3 FEP's |
| F | 1.75 | 0.061 | `F` | `parm10.dat` | Gough et al. JCC 13,(1992),963. |
| CL | 1.948 | 0.265 | `Cl` | `parm10.dat` | Fox, JPCB,102,8070,(98),flex.mdl CHCl3 |
| BR | 2.22 | 0.32 | `Br` | `parm10.dat` | Junmei(?) |
| I | 2.35 | 0.4 | `I` | `parm10.dat` | JCC,7,(1986),230; |

What one type per element leaves out (all from the same file, R\* / ε):

- **Carbon:** every carbon type has R\* 1.908; the sp3 types (`CT` 1.908 / 0.1094) have a deeper well than the sp2 types
  used here (`C` 1.908 / 0.086).
- **Oxygen:** hydroxyl `OH` 1.721 / 0.2104, ether and ester `OS` 1.6837 / 0.17, and the water oxygen of TIP3P `OW` 1.7683 / 0.152 differ from the
  carbonyl / phosphate oxygen used here (`O` 1.6612 / 0.21).
- **Nitrogen, phosphorus, sulfur:** every type of these elements in the file has the values used here.

## Table 2: hydrogens, by the atom they are bonded to

Also from `parm10.dat`. Amber's hydrogens differ far more than its heavy atoms: a hydrogen on nitrogen is small,
one on oxygen has no Lennard-Jones term at all, one on carbon is large. One value for all of them (R\* 1.1 Å,
MDATK up to 0.1.0) made every hydrogen bond strongly repulsive.

| Hydrogen | R\* (Å) | ε (kcal/mol) | Amber type | File | Comment in the file |
|---|---|---|---|---|---|
| H on N | 0.6 | 0.0157 | `H` | `parm10.dat` | Ferguson base pair geom. |
| H on O | 0.0 | 0.0 | `HO` | `parm10.dat` | OPLS Jorgensen, JACS,110,(1988),1657 |
| H on S | 0.6 | 0.0157 | `HS` | `parm10.dat` | W. Cornell CH3SH --> CH3OH FEP |
| H on C / other | 1.387 | 0.0157 | `H1` | `parm10.dat` | Veenstra et al JCC,8,(1992),963 |

Hydrogens on carbon range from `HP` 1.1 / 0.0157 (next to a charged group) through `H2` 1.287 / 0.0157, `H1` 1.387 / 0.0157, `H5` 1.359 / 0.015,
`H4` 1.409 / 0.015 and `HA` 1.459 / 0.015 to `HC` 1.487 / 0.0157 (aliphatic, no electron-withdrawing neighbour); `H1` ("H aliph. bond. to C with 1 electrwd. group" in the file), in the middle of that range, is used for
all of them.

## Table 3: monatomic ions

What AmberTools' `leaprc.water.tip3p` loads: `frcmod.ionsjc_tip3p` for the monovalent ions and
`frcmod.ions234lm_126_tip3p` for the divalent ones (its "12-6 normal usage set", the CM set of the paper: a
compromise between hydration free energy and ion-oxygen distance).

| Element | R\* (Å) | ε (kcal/mol) | Ion | File | Set |
|---|---|---|---|---|---|
| LI | 1.025 | 0.0279896 | `Li+` | `frcmod.ionsjc_tip3p` | Joung & Cheatham 2008, TIP3P set |
| NA | 1.369 | 0.0874393 | `Na+` | `frcmod.ionsjc_tip3p` | Joung & Cheatham 2008, TIP3P set |
| K | 1.705 | 0.1936829 | `K+` | `frcmod.ionsjc_tip3p` | Joung & Cheatham 2008, TIP3P set |
| RB | 1.813 | 0.3278219 | `Rb+` | `frcmod.ionsjc_tip3p` | Joung & Cheatham 2008, TIP3P set |
| CS | 1.976 | 0.4065394 | `Cs+` | `frcmod.ionsjc_tip3p` | Joung & Cheatham 2008, TIP3P set |
| F | 2.303 | 0.003364 | `F-` | `frcmod.ionsjc_tip3p` | Joung & Cheatham 2008, TIP3P set |
| CL | 2.513 | 0.035591 | `Cl-` | `frcmod.ionsjc_tip3p` | Joung & Cheatham 2008, TIP3P set |
| BR | 2.608 | 0.0586554 | `Br-` | `frcmod.ionsjc_tip3p` | Joung & Cheatham 2008, TIP3P set |
| I | 2.86 | 0.0536816 | `I-` | `frcmod.ionsjc_tip3p` | Joung & Cheatham 2008, TIP3P set |
| MG | 1.36 | 0.01020237 | `Mg2+` | `frcmod.ions234lm_126_tip3p` | Li et al. 2013, CM set for TIP3P |
| CA | 1.649 | 0.1059287 | `Ca2+` | `frcmod.ions234lm_126_tip3p` | Li et al. 2013, CM set for TIP3P |
| SR | 1.81 | 0.22132374 | `Sr2+` | `frcmod.ions234lm_126_tip3p` | Li et al. 2013, CM set for TIP3P |
| BA | 2.019 | 0.40664608 | `Ba2+` | `frcmod.ions234lm_126_tip3p` | Li et al. 2013, CM set for TIP3P |
| MN | 1.407 | 0.0168671 | `Mn2+` | `frcmod.ions234lm_126_tip3p` | Li et al. 2013, CM set for TIP3P |
| FE | 1.353 | 0.00941798 | `Fe2+` | `frcmod.ions234lm_126_tip3p` | Li et al. 2013, CM set for TIP3P |
| CO | 1.299 | 0.00483892 | `Co2+` | `frcmod.ions234lm_126_tip3p` | Li et al. 2013, CM set for TIP3P |
| NI | 1.255 | 0.0026232 | `Ni2+` | `frcmod.ions234lm_126_tip3p` | Li et al. 2013, CM set for TIP3P |
| CU | 1.218 | 0.00148497 | `Cu2+` | `frcmod.ions234lm_126_tip3p` | Li et al. 2013, CM set for TIP3P |
| ZN | 1.271 | 0.00330286 | `Zn2+` | `frcmod.ions234lm_126_tip3p` | Li et al. 2013, CM set for TIP3P |
| CD | 1.412 | 0.01773416 | `Cd2+` | `frcmod.ions234lm_126_tip3p` | Li et al. 2013, CM set for TIP3P |

- **The water model is not known** without a parm7 topology, and ion parameters are fitted per water model. The
  TIP3P sets are used throughout. For comparison, the same publication's Na⁺ is 1.212 / 0.3526418 for SPC/E and
  1.226 / 0.1684375 for TIP4P-Ew, its Cl⁻ 2.711 / 0.0127850 and 2.760 / 0.0116615.
- **The charge is not known either:** iron is taken as Fe²⁺ and copper as Cu²⁺.
- A halogen that is part of a molecule takes its row of table 1, not its ion's.

## Where the numbers were read

The three files were read on 2026-11-25 from public copies of the AmberTools parameter files (AmberTools itself is
not needed, and was not installed where this table was made):

| File | Copy read | SHA-256 (first 16 hex digits) |
|---|---|---|
| `parm10.dat` | github.com/ParmEd/ParmEd, `test/files/parm/parm10.dat` | `1ca292d5679e3545` |
| `frcmod.ionsjc_tip3p` | github.com/ParmEd/ParmEd, `test/files/parm/frcmod.ionsjc_tip3p` | `eb7a1a59bf356581` |
| `frcmod.ions234lm_126_tip3p` | github.com/choderalab/ambermini, `share/amber/dat/leap/parm/` | `f8bdb3df35ebdd90` |

Cross-checks made then:

- The rows of tables 1 and 2 that exist in `parm94.dat` and `parm99.dat` (same repository) have the same values
  there; chlorine and bromine first appear in `parm99.dat`.
- The Mg²⁺, Ca²⁺, Mn²⁺, Zn²⁺, Na⁺ and Cl⁻ rows agree to all printed digits with OpenMM's conversion of the same
  two ion files (`amber14/tip3p.xml`), converted back from σ in nm and ε in kJ/mol.
- In an AmberTools installation the files are in `$AMBERHOME/dat/leap/parm/`; the lines can be compared with
  `grep` there.

## The table up to MDATK 0.1.0

For results made with MDATK 0.1.0 or older (analysis version 1): one row per element, whatever the atom, written
without a record of its sources. Its values, R\* / ε:

| Element | Value | What it was |
|---|---|---|
| H | 1.1 / 0.0157 | Amber type `HP` (a hydrogen on a carbon next to a charged group), for every hydrogen |
| C, N, O, P, S, F, Br | as table 1 | the same Amber types |
| Mg | 0.7926 / 0.8947 | Amber type `MG` of parm99 (Åqvist's Mg²⁺) |
| Zn | 1.10 / 0.0125 | Amber type `Zn` of parm99 |
| Na, K, Cl | 1.369 / 0.0874, 1.705 / 0.1937, 2.513 / 0.0355 | Joung & Cheatham 2008 (TIP3P), cut to four decimals |
| Ca | 1.71 / 0.12 | no published set |
| Mn | 1.2 / 0.05 | no published set |
| any other element | 1.9 / 0.1 | no published set |

Chlorine in a molecule took the chloride ion's row. With hydrogens in the topology, the van der Waals energy of a
hydrogen-bonded interface came out strongly positive. Measured on MDATK's demo system (a 12-base-pair RNA duplex
with hydrogens, strand against strand, first frame, 12 Å cutoff): -34.7 kcal/mol with the topology's own parm7
parameters, +145.4 kcal/mol with that table, -34.1 kcal/mol with the tables above.

## References

- **parm94** (the types `C`, `N`, `O`, `P`, `S`, `F`, `I`, `H`, `HO`, `HS`, `H1` and their values): Cornell W. D., Cieplak
  P., Bayly C. I., Gould I. R., Merz K. M., Ferguson D. M., Spellmeyer D. C., Fox T., Caldwell J. W., Kollman P. A.
  *J. Am. Chem. Soc.* 1995, 117, 5179-5197. doi:10.1021/ja00124a002
- **parm99** (adds `Cl` and `Br`; `F` and `I` are in parm94 already): Wang J., Cieplak P., Kollman P. A. *J. Comput. Chem.* 2000, 21,
  1049-1074. doi:10.1002/1096-987X(200009)21:12<1049::AID-JCC3>3.0.CO;2-F
- **Monovalent ions:** Joung I. S., Cheatham T. E. *J. Phys. Chem. B* 2008, 112, 9020-9041.
  doi:10.1021/jp8001614
- **Divalent ions:** Li P., Roberts B. P., Chakravorty D. K., Merz K. M. *J. Chem. Theory Comput.* 2013, 9,
  2733-2748. doi:10.1021/ct400146w

The comments quoted in tables 1 and 2 name the origin of single rows as the Amber developers recorded it (OPLS,
Veenstra et al., Gough et al., Fox, …); they are reproduced as they stand in the file and were not traced further.
MDATK lists the four references above under *Citations* whenever a run used this table (resource `generic_lj`).
