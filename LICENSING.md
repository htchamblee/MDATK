# Licensing plan (not yet in effect)

The released Python package (up to v0.0.3c) is licensed under the GNU GPL v3 (see LICENSE). That
does not change: copies already distributed stay GPLv3.

The stand-alone Python builds (AppImage, Windows) are GPLv3 as well and may therefore bundle GPL
components (Barnaba, a GPL FFmpeg build); their `licenses/` folder lists every bundled component. The
dependency rules below apply to the planned C++ version only.

## Intended licensing of the C++ version

Not adopted yet; it takes effect only after legal review and a formal change of LICENSE.

- **Non-commercial use:** free, under a standard source-available non-commercial license
  (candidate: PolyForm Noncommercial 1.0.0) with a notice prohibiting biological-weapons use.
- **Commercial use:** a paid MDATK Commercial License (draft held by the author; to be reviewed
  by a lawyer). Licensor to be the author's LLC once formed, with the copyright assigned to it.

## Rules until then

1. **Copyright:** all code must be owned by the author. Outside contributions require a
   contributor license agreement that permits relicensing.
2. **Resources:** no third-party or institutional resources (equipment, networks, funded time)
   go into the C++ version.
3. **Dependencies bundled in binaries must allow proprietary use:**

   | Dependency | License | Rule |
   |---|---|---|
   | Qt 6 | LGPL v3 | link dynamically |
   | chemfiles | BSD-3 | OK |
   | Eigen | MPL 2.0 | OK |
   | FFmpeg | LGPL (only) | no GPL components (no x264); use OpenH264 / AV1 |
   | Lua | MIT | OK |
   | NumPy, SciPy, Matplotlib (interim) | BSD-style | OK |
   | MDAnalysis (interim) | check the bundled version's license | verify before bundling |
   | Barnaba | GPL v3 | never bundled; optional user-installed engine |

   External programs MDATK writes scripts for (Amber, GROMACS, Rosetta, Phenix, Coot, CCP4 ...)
   are not distributed with MDATK and keep their own licenses.
