# MDATK C++ architecture plan

Status: agreed plan, no C++ code yet. The Python package (v0.0.3c) is frozen as the reference
implementation and receives bug fixes only.

## Why

MDATK must never ask a user to manage a language runtime. A chemist who only uses email should
download one file, start it, and work. Python cannot promise that (virtual environments, conda,
minor releases that break compatibility), and the roadmap (real-time Vulkan rendering of systems
with millions of atoms) needs native performance.

## Delivery

| Platform | Binary | Build from source |
|---|---|---|
| Linux | AppImage (one file, no install, no admin rights) | CMake, dependencies fetched at configure time |
| Windows | installer (.exe) | CMake, same |

macOS is not supported. No software-rendering fallback until someone asks for one (Vulkan is
assumed). Standard: C++20. Build: CMake; dependencies fetched (FetchContent / vcpkg manifest) so
a source build does not pollute the system.

## Toolkit and libraries

- GUI: Qt 6 (Widgets for the sidebar, wizard and planner; a Vulkan window for the 3D view;
  charts drawn in Qt, e.g. QCustomPlot, replacing Matplotlib early)
- Trajectories and topologies: chemfiles (+ own parm7 / PDB readers where chemfiles falls short)
- Linear algebra: Eigen; neighbour search: own cell list / nanoflann
- Video: FFmpeg libraries, LGPL-only build (no x264; OpenH264 or AV1 encoders)
- Scripting: embedded Lua for script/wizard generators
- Interim only: an embedded, private Python interpreter inside the bundle for pieces not yet
  ported (never visible to the user, never installed on the system)

## Modules

Each module is a library with a C++ API; only `app` depends on Qt widgets.

| Module | Contents | Depends on |
|---|---|---|
| `core` | data model (topology, coordinates, selections, residue groups), `.mdatk.toml` project file, step log, citations, settings | - |
| `io` | chemfiles wrapper, NetCDF/parm7, PDB read/write, MRC/CCP4 maps, mmCIF dictionaries | core |
| `chem` | PDB hygiene (L-RNA detection, naming schemes, chains, renumbering), base pairs, dot-bracket, annotation, helix builder, mutations, dictionary renaming | core, io |
| `analysis` | RMSD, contacts, H-bonds, Mg/O-Mg-O, stacking, energy, trace/proximity; parallel over frames; result cache | core, io, chem |
| `render` | Vulkan scene (trace, cartoon, licorice, surfaces, density shadow), panels and charts, video encoding; usable headless for batch videos | core, analysis |
| `workflow` | planner model, step types and checks; generators written in Lua and loaded from `.lua` files (users can add steps) | core, chem |
| `app` | Qt GUI and the CLI: two thin front ends over the same libraries | all |

PDB hygiene stays in C++ (it shares chirality and base logic with the builder and analyses);
Lua generators call it rather than reimplement it.

External programs (cpptraj, Amber, GROMACS, Rosetta, SimRNA, RNAMake, Phenix, Coot, acedrg,
eLBOW, VMD) stay external: they are validated, published tools that keep users' work
publishable. MDATK writes and (on request) runs their scripts.

## Compatibility

- `.mdatk.toml` project files and the CLI commands stay compatible.
- The analysis cache is not compatible (recomputed).
- Generated scripts must match the Python generators' output for the same settings.

## Migration order (speed first)

1. Skeleton: CMake, Qt 6 + Vulkan window, AppImage and Windows installer in CI; opens an existing
   `.mdatk.toml`. Proves the delivery promise before any science moves.
2. `core` + `io`: project file, chemfiles reading, PDB/maps.
3. `analysis`: all analyses, compared against Python on the test trajectories.
4. `render`: Vulkan structure view and representations; Qt charts (retires Matplotlib);
   video encoding.
5. `app`: GUI pages, canvas layout editor, CLI.
6. `chem`: hygiene, base pairs, annotation, builder, dictionaries.
7. `workflow`: Lua generators for every step type, planner, wizard, runner.
8. Remove the embedded Python.

## Definition of done (per module)

A module is done when it reproduces Python v0.0.3c on the existing test trajectories, PDB files
and workflows: the same numbers within floating-point tolerance, the same generated files
byte for byte (except time stamps), and the same project files. The Python test suite is the
specification; each Python test gets a C++ (or CLI-level) counterpart.

## Repository

Same repository. The C++ code lives under `cpp/` next to the frozen Python package so both can
be run on the same inputs for comparison.
