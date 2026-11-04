# Changelog

All releases of MDATK, newest first. The dates are those of the version commits. There was no 0.0.9:
0.0.10 followed 0.0.8. Since 0.0.8 a regression test (`tests/test_golden.py`) compares the files that
the step types of 0.0.7 write with the 0.0.7 output; the intended changes are listed in the test and in
the notes below.

- [0.0.12](#0012-2026-11-02) · [0.0.11](#0011-2026-10-31) · [0.0.10](#0010-2026-10-30) ·
  [0.0.8](#008-2026-10-29) · [0.0.7](#007-2026-10-25) · [0.0.6](#006-2026-10-25) ·
  [0.0.5.1](#0051-2026-10-25) · [0.0.5](#005-2026-10-25) · [0.0.4.1](#0041-2026-10-22) ·
  [0.0.4](#004-2026-10-21) · [0.0.3d](#003d-2026-10-20) · [0.0.3c](#003c-2026-10-18) ·
  [0.0.3b](#003b-2026-10-18) · [0.0.3a](#003a-2026-10-18) · [0.0.2c](#002c-2026-10-16) ·
  [0.0.2b](#002b-2026-10-15) · [0.0.2a](#002a-2026-10-15) · [0.0.1](#001-2026-10-12)

## Unreleased

### Custom scripts

- **Placeholders with real names in the help.** The help of a custom script's *Arguments*, input and output
  *Name* and *Script text* wrote `{in:<name>}` / `$MDATK_IN_<NAME>`, which read as text to type; it now
  shows them with example names (`{in:structure}`, `$MDATK_IN_STRUCTURE`, `{out:result}`).
- **A placeholder that names nothing declared says what to write.** `{in:<name>}` as it stood in the help
  gives "replace '<name>' in {in:<name>} with the input's name: write {in:structure}" (every declared input
  offered), a name in other case "names are case-sensitive: write {in:structure}", and with nothing declared
  yet "declare the input under Input files first"; the README's *Custom scripts* gives an example.

## 0.0.12 (2026-11-02)

### Model checks

- **`mdatk pdb qc MODEL.pdb …`** (`mdatk/modelcheck.py`, standard library only): clashes as overlap inside
  Bondi contact (0.4 Å), N/O pairs with a donor counted as hydrogen bonds from 2.2 to 3.2 Å (donors by
  base, terminal hydroxyls, protonated A⁺ / C⁺ on request; two acceptors never), carbon contacts beside
  an H-bond excused only when no closer than the H-bond, exclusions along bonds (up to three) instead of
  residue numbers, every chain and neighbouring residue checked, waters and ions skipped or (ions)
  checked against first-shell distances, heavy atoms only; O3′–P links within every TER segment
  (1.4–2.0 Å a bond, 2.0–3.0 Å strained, beyond a break) and P–P spacing; loop compaction for typed
  ranges or the residues unpaired in every given dot-bracket; sugar hands from the geometry against
  expected ranges, epimers named; kinks; a decoy table (`--scores`: run, tag, pdb, score terms) for
  report columns, limits and ranking; gates with `--on-fail report|filter|stop`, text, `--tsv` or
  `--json` reports, copies of the passed and best models.
- **Hydrogens stay out of the clash check** (no `--hydrogens`; the step's *Include hydrogens* is gone):
  their positions differ between programs, and the builder's ideal A-form helix has H2′(i)…H6/H8(i+1) at
  1.57–1.70 Å. Checked: the builder places H2′ (and H1′, H3′, H4′) from the sugar's own geometry, along
  the negative sum of the three bond directions, 1.09 Å from C2′ and 4.6° from the exact tetrahedral
  (equal-angle) direction, which would move the contact by under 0.03 Å; the short contact is the Arnott
  fiber's C2′(i)…C8/C6(i+1) of 3.5 Å, so the builder is unchanged and no radius is tuned.
- **Hydrogen contacts on MolProbity's conventions** (`--hydrogen-check`, step option *Report hydrogen
  contacts*; off by default; never a gate or a filter): explicit hydrogens only, Probe's radii (1.17 Å on
  carbon, 1.00 Å polar or aromatic at nuclear positions; 1.22 / 1.05 Å at electron-cloud positions,
  chosen from the median C–H or `--hydrogen-radii`; C 1.70, N 1.55, O 1.40, P 1.80 Å), a clash at an
  overlap of 0.4 Å or more, 0.6 Å allowed to a hydrogen bond, pairs up to four bonds apart left out,
  clashscore per 1000 atoms; its own report section and columns (`h_atoms h_clashes h_worst_overlap
  h_clashscore h_radii`).
- **O3′–P links: warn with a number.** Up to `--link-max` (2.0 Å) a bond; up to `--link-fail` (3.0 Å) a
  strained bond, reported with the strain tleap's bond puts into the first minimisation, E = k(r − r0)²
  (Amber's k: 230 kcal/mol/Å², 1.61 Å, OS–P of parm10.dat: 35 kcal/mol at 2.0 Å, 182 at 2.5, 444 at 3.0)
  in the text, the new columns `warnings`, `strained_links` and `link_strain` and the JSON; beyond, a
  break that fails the links gate. k and r0 from the force field with `--link-params` (tleap input,
  leaprc, parm*.dat, frcmod; `--link-types`, `--amberhome`) or `--link-k` / `--link-r0`
  (`modelcheck.link_bond_from_files`). Step option *Longest O3′–P link before a break*.
- **Kinks** (`--kink-min 60`, step section *Kinks*): the virtual C1′(i−1)–C1′(i)–C1′(i+1) angle per
  residue within each TER segment, flagged below 60° except at allowed positions (`--kink-allow
  RANGES`, or a two-column reactivity file of in-line probing with `--kink-reactivity FILE
  --kink-reactivity-min X [--kink-reactivity-chain C]`); report columns `kinks`, `sharpest_kink`,
  `sharpest_kink_at`; a gate only with `--gate-kinks` (off by default).
- **Filter: up to N, fail only below a minimum.** `--pass N` means up to N slots, `--min-pass M`
  (default 1) the fewest passing models (status 1 only below it; 0: never); slots no passing model fills
  are written as empty-slot files (`REMARK 951 empty slot k of N: only M models passed`, `END`, no
  atoms; `pdbtools.empty_slot_text`, `empty_slot`, `is_empty_slot`), and the report states `M of K models
  passed the gates`. Step options *Models passed on (up to)* and *Fewest passing models*; the step's
  passed models carry `empty_slots` in their port data. Empty slots are skipped with a note by the model
  check, `mdatk decoys` / `pdb interface`, `mdatk ensemble census|compare`, `mdatk frame` and `mdatk map
  score`; `read_pdb` refuses one by name; the PDB preparation of the Amber build says what it is.
- **Planner step *Model check (clashes · backbone · chirality)*** (Structure prediction): models (several),
  optional secondary structures and decoy table in; a report table, the passed models and the best one
  out. Expected hands are measured on the first model when the plan is written (or typed, or none); range
  options take `group:NAME`. The script runs a copy of the PDB tools and the model checks (`python3`
  only) and removes the outputs of an earlier run before it starts.
- **`mdatk pdb check`**: `--expect-l` / `--expect-d RANGES`, `--strict` (status 1 for a residue of the
  other hand, an unmeasurable expected one, or an epimerised, contradicting or flat centre) and `--table`
  (per residue: the three-centre call, the C4′ hand alone, the volumes, the expectation, a note).
- **PDB tools**: `infer_base` (the base from the atom names, for residue names that are not
  recognised) and `nucleotide(..., infer=True)`; `--strip-h`, `--drop-virtual` (Rosetta's VRT / XXX) and
  the scheme `standard` (plain A C G U for both hands, every record ATOM, with a warning that the names no
  longer record the hand), so `mdatk pdb clean --scheme amber --strip-5p --strip-h --drop-virtual`
  turns a Rosetta decoy into a tleap input and `--scheme standard` makes the copy for chirality-blind
  programs; the record policy `atom`; ranges accept `*` for every chain where hands are expected.
- **Amber build**: *Remove Rosetta virtual residues (VRT/XXX)* in the PDB preparation, on by default
  (tleap cannot type them, so only builds that failed change), and *Check O3′–P links before tleap*
  (off, warn or stop; off by default), also measured on an existing input PDB when the plan is written.
  The check warns with the strain from 2.0 Å and stops (stop) only beyond *Longest O3′–P link before a
  break* (3.0 Å), short or unmeasurable links; it reads k and r0 from the force field the first tleap
  input loads (`--link-params` of that input, `$AMBERHOME`; O3′ typed OR for OL3 + CP, else OS), and,
  when on, the PDB is prepared, named and checked first: before the CP, L-RNA library and TIP4P/2005
  helpers and every tleap run, the dry one of a two-pass build included (tested for each). A structure
  that is an empty slot of a model check (its port data say so, or the existing file is one) gets a guard
  as the script's first lines that stops it before any helper or tleap; the PDB preparation names an
  empty slot too. Generated files are unchanged apart from the two helpers that copy library code: the
  PDB tools and the PDB preparation (listed as intended in `tests/test_golden.py`).

### Rosetta: decoy table, validity filters, run audit, frames

- **FARFAR2 decoy table and score filter** (step *Rosetta · FARFAR2 models*, `mdatk rosetta scores`): SCORE
  terms are looked up by column name in the header that applies (`mdatk/workflows/scripts/silent_scores.py`,
  standard library, embedded in the step's folder). New output *scores* (file kind *table*):
  `<project>_<step>_scores_<time>.tsv` with `run` (`rNN` from the silent file's name, `NN-rNN` when two
  FARFAR2 steps feed the step), `tag`, `pdb` (plan-relative, empty when not extracted) and every SCORE term,
  one row per decoy, written with *Write the decoy table*, a filter or *Extract every model*;
  `mdatk.rosetta.read_decoy_table` reads it (TSV or CSV, extra columns kept). Filters: validity before
  clustering, quality after it (see below). *Extract every model of every run*: `models/<run>/<tag>.pdb`,
  each run in a `mktemp` folder of its own, earlier models of the run removed first, and a check that every
  run has one PDB per SCORE line. All new options are off by default; the step's files are unchanged
  without them.
- **Run audit** (`mdatk rosetta audit`, step *Rosetta · run audit*): what each rna_denovo run was given,
  from the command lines in its logs, with `@flags` files expanded recursively, `::` read as `:`, short and
  long flag spellings folded, every command line of a log read (restarts), input files fingerprinted by
  their whole content (same content under MDATK's step-numbered names is no difference), conditions
  compared and checked in pairs (`--pair`, `--pair-suffix`, `--allow`, `--ignore`), and, with
  `--controlled RANGES`, how each condition gives the pairing of those residues (secondary structure,
  obligate pairs, `-s` / `-input_res`, chain by chain). Report (text) and a flags-by-condition table.
- **FARFAR2 models: validity before clustering, quality after it** (the single score filter is replaced; a
  score cut before clustering reshapes the cluster populations a register census counts):
  - *Validity filters (before clustering)*, judged on every decoy and counted in order (decoys -> after
    hands -> after bases -> after closure -> after pairs): sugar hands against expected counts
    (`44 D / 29 L`) and / or ranges (`C:*`, `group:NAME`); base identity against the FARFAR2 step's sequence
    or a typed one; closure by a score term (`linear_chainbreak <= 5`) and / or every O3'-P <= 2.0 Å
    (`mdatk.modelcheck.check_backbone`); the defining pairs of a register (`mdatk.ensemble.parse_register`
    syntax, formed or approached with `mdatk.ensemble`'s distances). They imply *Extract every model*; a
    helper copied into the step folder (`…_validity_….py`: PDB tools, model checks, silent-file scores and
    `mdatk/workflows/scripts/decoy_validity.py`, standard library only) writes `…_validity_….tsv` and the
    report `…_validity_….txt` (new output *validity*), the valid decoys of each run go into
    `…_filtered-rNN_….out` and only those are clustered. The decoy table gets `valid` (1 / 0), `invalid_by`,
    `invalid_reason` and each criterion's columns for every decoy.
  - *Quality filters (after clustering, choosing representatives)*: score limits (`score <= -300,
    elec_dens_fast <= -40`, read with `mdatk.modelcheck.parse_score_limits`) on the ranked cluster centres;
    none by default. *Before clustering (changes cluster populations)* stays as an explicit choice with a
    planner warning; the default is after. Plans with the earlier `filter_term` / `filter_max` get the limit
    `TERM <= MAX`. With no filter (the default) the step's files are unchanged (`tests/test_golden.py`).
  - `silent_scores`: `keep` (the decoys to cluster: valid and within the limits), `filter-ranking --limit`,
    `table --validity`.
- **Run audit: which Rosetta, and the seed of every part**. Provenance comes from the lines before
  `core.init: command:` (`Rosetta version: <package> r<rev> <version> <commit> <repository> <date>`,
  `Rosetta extras`, the `-version` block, `core.init` commit / checkout / branch lines, older
  `… from <repository>` lines), kept raw and parsed tolerantly into `[rosetta version]`, `[rosetta extras]`
  and `[rosetta provenance]`, compared like flags (a pair built with different versions is confounded);
  the report lists every build with its runs and warns when there is more than one. `-constant_seed`,
  `-jran` and `-seed_offset` are a part's identity: the seed Rosetta logs (`real_seed`), else the flags
  (default 1111111), is reported for every run and restart (report, a `[seeds]` row in the flags table,
  `--runs-tsv` / step output *runs*), and two runs or starts with the same seed under `-constant_seed` and
  the same inputs are duplicates: an error in the report and on standard error (`--strict` fails). The same
  seed with other inputs is noted as a matched-seed design. Fully expanded command lines are the normal
  case; `@flags` files are still expanded.
- **Frames of multi-body models** (`mdatk/frames.py`, `mdatk frame check|reframe`, step *MDATK · frame
  check*): fit on one body, measure one or more others (P, C1′, C4′ by default; chain-qualified ranges or
  residue groups; atoms keyed by chain, residue number and insertion code); one frame when the 95th
  percentile of every body stays below 1 Å, models without the fit body listed but not counted; a re-fitted
  reference applied to the models when one rigid transform suffices (carried RMSD ≤ 1 Å), otherwise the
  step fails; moved decoys written as `<run>/<tag>.pdb` with ANISOU rotated, a manifest and a decoy table
  of the moved files; a warning for models sampled with the map in the score. Models come from PDB files
  or a decoy table's `pdb` column.
- Bugs of the user scripts these replace, fixed: runs of MDATK-named silent files collided on the stamp;
  a missing score term crashed the audit; duplicate tags overwrote each other silently; problems went to
  standard error only; extracted decoys of earlier runs stayed beside new ones and failed extractions went
  unnoticed; `::` flags and `@flags` files hid confounds; `-r` could not be switched off; relative paths
  were resolved for the first run only; flag aliases and restarts were missed; directory keys depended on
  the working folder; inputs were sampled, not hashed; atoms and output files were keyed without the chain
  or the run (same-named decoys of different runs overwrote each other or shared a label); a set-wide
  median let models without the fit body into the statistics; unusable models vanished from the table;
  ANISOU records were not rotated.

### Decoy ensembles, map-model scores and statistics

- **Four planner steps** with their commands, written from the user's analysis scripts (register census,
  register comparison, threshold sweep, pilot comparison, Q-score preparation and regions, body hybrids)
  and generalised: every residue range, chain, register, reference sequence and path is an option
  (chain-qualified ranges or project residue groups `group:NAME`), with the scripts' thresholds as defaults.
  - **Register census** (`register-census`, Analysis; `mdatk ensemble census`): which register (a named set
    of defining base pairs, `pres=B:22-B:37, ...`) each decoy formed (Watson-Crick N1/N3 < 3.5 Å) or
    approached (N1-N3 < 6 Å at C1'-C1' 8-13 Å), per folder and per build condition (FORCED / UNFORCED /
    PARTIAL / UNKNOWN per register) read from the Rosetta logs' command lines (`@flags` expanded:
    `-secstruct_file`, `-secstruct`, `-obligate_pair(_explicit)`), a `*.secstruct` beside the decoys or a
    connected secondary structure; never pooled across conditions; matched groups by folder suffix; score
    deciles within one weight set; Mann-Whitney between single-register decoys; L-ribose counts.
  - **Ensemble comparison** (`ensemble-compare`, Analysis; `mdatk ensemble compare`, modes compare / sweep
    / pilot): closure (a score term by name, default `linear_chainbreak < 5`, or every O3'-P < 2.0 Å),
    defining pairs, interface atoms above a contour calibrated on fixed regions (per decoy, or common on
    unmoved regions), Clopper-Pearson / Wilson intervals, Fisher tests, the closure-threshold sweep and the
    rates the second arm would need to be told apart with a given power. Numbers only, no verdict.
  - **Map-model score** (`map-model-score`, Refinement; `mdatk map score`): per-residue Q-score with
    ChimeraX's `qscore` command (below), or MDATK's own per-residue CC; models ranked by named regions
    against a reference region with the regions removed, chain termini left out, NaN counted
    (`mdatk ensemble regions` ranks per-residue tables from elsewhere).
  - **Body hybrids** (`body-hybrids`, Refinement; `mdatk pdb hybrids`): all 2^k old/new combinations of up
    to six bodies over the atoms both models have, coordinates verbatim; with a map, each body's main
    effect.
- **Numbering**: register pairs and ranges are in a reference construct's numbering; each chain of a decoy
  is located in the reference sequence (typed, FASTA, or the plan's FARFAR2 sequence) and the bases of the
  defining residues are checked. Bases from MDATK's residue-name parser, else from a diagnostic atom.
- **Decoy table**: steps take the FARFAR2 models step's `scores` table (`run`, `tag`, `pdb`, SCORE terms
  by name) as an optional input; the commands take `--scores TSV`.
- `mdatk/ensemble_stats.py` (standard library only): two-sided Fisher exact test, Clopper-Pearson in log
  space, Wilson, exact Fisher power and the detectable rates at a power, Mann-Whitney U (exact / normal);
  each checked against SciPy in the tests.
- Bugs of the scripts fixed in the rewrite (regression tests in `tests/test_ensemble.py`): residues keyed by
  number alone (chains merged in compare.py, register_census.py, body_hybrids.py and q_regions.py);
  a register counted only at exactly 3 pairs; a secondary structure called FORCED when its brackets stood at
  the register's positions but paired other residues; `-secstruct_file=` and `@flags` not read; hydrogens
  recognised by the atom name; the Clopper-Pearson fallback overflowing for large n; division by zero for
  an empty arm; decoys without calibration or region atoms dropped silently; score columns read by position;
  the pilot's "POWER" band (a Wilson interval, not a power); scaffold and interface overlapping in the
  region means, NaN averaged and 'whole' keeping the termini; Rosetta's virtual residues scored as atoms;
  overlapping bodies resolved silently.
- **Q-score with ChimeraX** (the user's decision: Phenix has no official per-residue Q-score; its
  cctbx program is `mmtbx.development.qscore`): method `qscore` of `mdatk map score`, `mdatk pdb hybrids`
  and the steps *Map-model score* and *Body hybrids* runs ChimeraX's `qscore` command (the QScore bundle of
  the ChimeraX Toolshed, ChimeraX-QScore by T. I. Croll) as `chimerax --nogui --cmd "open MODEL id 1
  maxModels 1; open MAP id 2; qscore #1 toVolume #2 useGui false logDetails true [options]; exit"`, on the
  model as it is (L names such as LG5 and HETATM records are scored; the Phenix command template and the
  standard-name copy are gone). Options: the ChimeraX executable (`--chimerax`, default `chimerax`),
  further qscore options as keyword value pairs (`--qscore-options 'pointsPerShell 16'`; MDATK's own
  keywords and `;` refused, undeclared keywords warned about), models per ChimeraX run (`--batch`, default
  1: the map opened once, each model opened as #2, scored and closed after a `log text MDATK-QSCORE k`
  marker; a run that stops at a model resumes after it), the column the regions use (`--q-column qavg`,
  `qbb` or `qsc`), a time limit per model. Rosetta's virtual residues are left out of the selection
  (`#1 & ~:VRT`) and of the table. The `logDetails` table is read tolerantly (blank lines, the dashed rule,
  tabs or spaces, `INFO:` lines and `<pre>` tags, insertion codes, `N/A`, warnings such as `Ignored bad PDB
  record`, the `Overall mean Q-Score` line, reported per model). The per-residue table has `qavg`, `qworst`,
  `qbb`, `qsc` and `score`; ChimeraX's log of every run stays in `qscore/`. A model ChimeraX cannot score is
  listed in the report with the reason (e.g. atoms outside the map box, where ChimeraX-QScore 1.2 stops its
  table) and not ranked, and the command ends with status 1 after writing the others; a missing ChimeraX or
  qscore command stops at once. `mdatk ensemble regions` also reads ChimeraX logs (one model per table) and
  qscore's CSV. Citations: UCSF ChimeraX (Meng et al. 2023, Pettersen et al. 2021) and the Q-score paper
  (Pintilie et al. 2020); the pages and authors of the latter are from memory (stated in the data). Tested
  with a stand-in `chimerax` printing the user's log format, not with ChimeraX itself.
- **Ranking on dQ**: `mdatk map score`, `mdatk ensemble regions` and the *Map-model score* step rank by
  default on dQ = Q(first region) − Q(reference), the interface against the scaffold, instead of the first
  region's mean (`--sort NAME` still ranks on the mean); the report heads its columns `dQ:NAME` (`dCC:NAME`
  for the CC), says what it ranked on and gives the spread of every dQ over the models.
- **Planner template *Decoy validation (FARFAR2)*** (`decoy-validation`): FARFAR2 → FARFAR2 models (the
  closure validity filter `linear_chainbreak ≤ 5` before clustering) → model check (clashes) → decoy
  interfaces (segment support) → map-model score (Q-score with an *interface* region, ranked on dQ) and a
  second model check gating on kinks, a map input, and notes on the validity filters left to fill in (hands
  and defining pairs need the construct), the rule "filter on validity before clustering; filter on quality
  after, when choosing representatives" and the kink test (positions that in-line probing shows flexible
  are allowed).

### Decoy interfaces

- **`mdatk/interfaces.py`**: the interface between two sides of every model of an ensemble, without a
  register, construct or numbering built in. Sides are chain-qualified ranges (`A:1-16,B:17-44`), whole
  chains, ranges for every chain or residue groups (`group:NAME`); residues are keyed by chain, number and
  insertion code, read with `pdbtools.read_pdb`; only nucleotides count (L and unknown names get their base
  from the atoms, the hand from the sugar's chiral volumes), never ions, water, virtual residues or
  hydrogens. Rewritten from the user's `interface_census.py`, `interface_clusters.py`, `support_check.py`
  and `dssr_interface.py`:
  - **census**: contacts, stacking, Watson-Crick / G·U / non-canonical pairs (builtin, barnaba or DSSR),
    base-phosphate and 2′-OH-phosphate hydrogen bonds, Mg-bridgeable P–P pairs, residues and the
    chain-qualified contact fingerprint; score filters on a decoy table (`linear_chainbreak <= 5` …);
  - **clusters**: deterministic leader clustering of the fingerprints by Jaccard distance (residues,
    blocks or pairs), exact medoids as representatives (copied as PDB files), reference interfaces typed
    or measured on reference models, scored on both sides and normalised per reference; neutral pattern
    and reference-share summary;
  - **segment support**: what holds each rebuilt residue (partner side, own scaffold, optional base-pair
    credit as its own class, or only its own rebuilt segments), longest unsupported runs, fixed or
    automatic run threshold, per-residue profile.
- **Decoy table reader** for the `scores` output of the FARFAR2 models step (`run`, `tag`, `pdb`, every
  score term by name; extra columns ignored; TSV, CSV or whitespace), matching models by path, file name or
  a tag unique across runs.
- **`mdatk decoys census|clusters|support`** and **`mdatk pdb interface`** (TSV of the pairs, stacks and
  base / 2′-OH to phosphate hydrogen bonds between two sides, with the hand of each partner;
  `--engine builtin|barnaba|dssr`). `-p PROJECT` resolves residue groups and logs the step.
- **Planner step *Decoy interfaces*** (`decoy-interface`, Analysis): inputs models (PDB, several), scores
  (table, optional), reference models (PDB, optional); outputs census, clusters and support (tables) and the
  cluster representatives (PDB, several). Its script calls `python3 -m mdatk decoys …` (or the AppImage);
  groups are written into it as ranges. The rebuilt residues of the support check are the complement of
  FARFAR2's fixed residues when the connected models carry them (`PortValue.data["input_res"]`), or typed.
- **DSSR engine** (`mdatk/dssr.py`; `mdatk pdb annotate --engine dssr`, `annotate(engine="dssr")`):
  x3dna-dssr (licensed separately, never bundled) runs on a copy with canonical names and MDATK's own
  residue numbering in a temporary folder; its JSON (DSSR 1.x layout: `nts`, `pairs`, `stacks`, `hbonds`)
  maps back by residue index, so it no longer depends on DSSR's id grammar; ids are also read tolerantly
  (`1:A.G1`, `AB.G1`, `A.0A/45`, `A.G1^B`, atom ids in either order). Checked against the JSON of DSSR
  2.9.4 (four files of the user's, with and without chain ids): a blank chain gives chain-less ids (`G1`,
  `O6@G1`), now read as the first chain of the copy instead of being dropped.
- **Fixed on the way** (problems of the scripts, each with a regression test):
  - an A·C mismatch and trans Watson-Crick pairs counted as Watson-Crick (one N1/N3 distance);
  - Mg-bridgeable P–P pairs tested only for residue pairs that already had a contact;
  - residues keyed by number, so chains numbered alike merged and sequence separation ignored chains (a
    base-paired residue of a duplex was "unsupported"); ions inside a range counted as residues and as
    partner support; hydrogens named `1H5'` counted as heavy atoms; base letters read from the first
    A/G/C/U character of the residue name;
  - one run's chainbreak applied to another run's decoys (silent-file scores keyed by a part every RNAdo
    run shares); the census score column read from file names;
  - the fingerprint dropped chains and broke on negative residue numbers;
  - reference shares biased toward the reference with more exclusive residues, target side only, scored
    from the representative alone, tallied over the printed clusters only; medoids from a 60-member sample;
    seeding dependent on the input order; a no-reference run printed reference messages for every cluster;
  - pair credit added to the scaffold class (and a bifurcated hydrogen bond counted as a pair);
  - DSSR: interface pairs lost with per-chain numbering, several-letter or blank chains, names with digits
    and insertion codes; every residue renamed (Mg and virtual residues became U); multi-model files
    written without ENDMDL; a flat sugar called L; DSSR's auxiliary files left in the working folder.

### MDATK and RNAdo in the applications menu

- **An RNAdo menu entry from the AppImage, too.** Desktop integration takes one desktop entry from an
  AppImage (MDATK's, with an RNAdo action), so RNAdo had no launcher of its own there. *Add MDATK and
  RNAdo to the applications menu…* (main menu of the analysis window and of RNAdo), `mdatk desktop
  install` or `rnado --install-launchers` (`rnado-gui --install-launchers` too) writes `mdatk.desktop` and
  `rnado.desktop` into `$XDG_DATA_HOME/applications` (`~/.local/share`) and their icons into
  `…/icons/hicolor/<size>/apps` (`mdatk.png`, `rnado.png` in every size, `scalable/apps/rnado.svg`).
  The window lists the files and the commands first, then what it wrote; in the AppImage (and a bundle
  folder) it warns that the entries hold the current path, so moving the file breaks them.
- **The entries start what is running** (`mdatk/launchers.py`, from `mdatk.gui.launch.launch_mode`): the
  AppImage (`"$APPIMAGE" gui %f` / `rnado %f`; without FUSE with `env APPIMAGE_EXTRACT_AND_RUN=1`), the
  programs of a bundle folder (`mdatk gui`, `mdatk-gui` where there is one, `rnado-gui`; an extracted
  AppImage's `AppRun`), the `mdatk-gui` / `rnado-gui` scripts that run this Python (venv `bin`, user
  `bin`, `PATH`; recognised by their `#!` line, pip's `/bin/sh` form included) or `<python> -m mdatk gui`
  / `-m mdatk.gui.rnado`, and a source checkout that is not installed with `env PYTHONPATH=<checkout>`.
  Arguments are quoted as the Desktop Entry Specification asks (tested through GLib, and by launching an
  entry whose program path has a space and a `$`).
- **Only MDATK's own files are touched.** Each file carries `X-MDATK-Installed=<version>` (a desktop-entry
  key, a PNG `tEXt` chunk, an SVG comment); files of the same name without it are left alone and listed
  (`--replace` moves them to `*.bak` first); files are written through a temporary file. `mdatk desktop
  uninstall` (or *Remove them* in the window) removes the marked files of any version and nothing else;
  `mdatk desktop status` shows each file (`installed`, `outdated`, `missing`, `not MDATK's`) and the
  commands. `update-desktop-database` and `gtk-update-icon-cache` (the latter only for a theme folder
  that has a cache) run when installed and never make it fail. On Windows the menu item and the command
  say that the installer made the Start-menu entries, and write nothing.
- **Package data:** the desktop-entry templates moved from `data/` to `mdatk/assets/` (installed and
  bundled with the icons; the AppImage build reads them there), and the MDATK icon is committed as
  `mdatk/assets/mdatk-icon-<size>.png` (16–256 px), drawn by `packaging/make_icon.py --assets`.

### RNAdo: completed runs move aside

- **A step that is written again first moves its completed run aside** (the user's decision: as soon as its
  scripts are written again, even if the step is never run again, scripts included, so the run stays
  repeatable when the step is written with changed settings). When a step is about to be written (*Write
  all files*, *Write step*, the wizard's *Write files*, every *Run* action, `workflow write [--run]`,
  `workflow generate`) and its folder holds a complete run of the plan's current stamp (runstate's
  *done*), that run moves into `<step folder>/superseded/<stamp>_<YYYYMMDD-HHMMSS>/` (`no-stamp_<time>`
  without time stamps; `superseded/<NN>_<tool>/<stamp>_<time>/` in the plan folder without step folders):
  the step's outputs, every file it writes for the stamp (scripts, inputs, helpers), its logs (files with
  closing lines and the other files named `<project>_<NN>_*_<stamp>.*`), stamp-free outputs with their
  completion marker (custom scripts' fixed names), and the stamp-free folders a run fills (`runs/` of
  FARFAR2 and SimRNA, `models/` of the FARFAR2 models step; new `Tool.run_folders`), relative paths kept.
  A `manifest.toml` records what moved, from which step and stamp, when and why; a saved project logs each
  move in its step log (`workflow supersede`). Partial runs, steps that use an earlier run, files of other
  stamps, other steps' and the user's files stay; nothing in `superseded/` is replaced (`…-2`); a step's
  move is all or nothing (on an error the files go back and the step is not written, reported as skipped).
  `runstate` never looks inside `superseded/`. New module `mdatk/workflows/supersede.py`;
  `Plan.write(..., project_path=, supersede=)` and `WriteReport.superseded`.
- RNAdo asks before writing (listing the steps, where their runs go and their files; *Cancel* or *Move
  aside and write*; refused while a step that reads them runs), the output log lists what moved, and the
  naming bar's *Move completed runs aside* is the project option `workflow.supersede_runs` (on by default;
  off: writing replaces as before). `mdatk workflow write` prints the moves on standard error and under
  `superseded` in its JSON, and `--dry-run` lists the files it would write and the runs it would move
  without touching anything. Generated files are unchanged.
- Steps that skip finished stages when run again (Amber MD *Skip finished stages*, NAMD MDFF *Skip
  finished runs*) reuse them from a run left in place: a partial run, or any run with the option off (a
  complete run is moved aside, so the step runs again from the start). `tests/test_namd_mdff_tool.py`
  writes with the option off to test that reuse.

### Running steps: CPU and GPU budgets, stop buttons

- **Two budgets instead of *Parallel*** (the user's decision): the scripts RNAdo and `mdatk workflow run`
  run at the same time share *CPU cores* (default: half of this computer's cores) and *GPUs* (default: the
  GPUs found, `nvidia-smi -L`, or the ids in MDATK's own `CUDA_VISIBLE_DEVICES`, which then limit what
  the scripts may use). A ready step starts when the cores and GPUs it uses are free, in plan order (a
  later, smaller step may start while an earlier one waits for room, but not behind a step that needs
  every core: that one starts once nothing runs, before the steps after it); the scheduling is a pure
  function of the plan order, the needs and the finished steps (`mdatk/workflows/budget.py`: `Slots`,
  `schedule`). Project file: `workflow.cpu_cores` (0: auto), `workflow.gpus` (-1: auto, 0: none); command
  line: `mdatk workflow run / write --run --cores N --gpus N`; RNAdo: *CPU cores* (shown as *auto (N)*),
  *GPUs* (*auto (N)*, *none*, 1, 2, ...) and *At once* in the naming bar.
- **What each step uses** (`Tool.run_resources(settings)` -> `Resources(cores, gpus, pinned)`, one core
  by default): Amber MD with pmemd.cuda 1 core + 1 GPU, pmemd.MPI its MPI processes, pmemd 1 core; NAMD
  MDFF its `+p` cores and one GPU; GROMACS its OpenMP threads (`-ntomp`, times `-ntmpi`, or `-nt`, from the
  extra mdrun flags; without `-ntomp` every core, since mdrun takes them all) + a GPU when offloading;
  FARFAR2 and SimRNA their parallel runs; FARFAR2 models its parallel extractions when it extracts every
  model; Phenix RSR `nproc`; register census and frame check their worker processes; a custom script its
  new section *Cores and GPUs* (*CPU cores*, 0: all; *GPUs*; its run script is unchanged). Planner boxes
  show it as a badge (`8c+1g`), the details pane in words, the run confirmation per script.
- **GPUs handed out:** a step that uses a GPU gets free ones in `CUDA_VISIBLE_DEVICES`, unless it names
  its own (the Amber MD step's *GPU*, NAMD's `+devices`, which counts the GPUs its *Environment* leaves
  visible, or `CUDA_VISIBLE_DEVICES` in the step's *Environment* setup commands): those ids are used as
  named and kept from other steps. Each start line says what the step got (`==== step 04: … - bash
  x.sh (1 core, GPU 1)`), and the step log's command shows the `CUDA_VISIBLE_DEVICES` given.
- **No deadlock, clear warnings:** a step larger than the whole budget runs alone once nothing else runs
  (and holds back the steps after it meanwhile); with no GPU found and none set, the steps that use a
  GPU run one at a time; a named GPU outside the budget is reported. The warnings open the run's output
  (`WARNING: …`, amber in RNAdo's log).
- **`parallel` stays, as a cap** on the scripts at once (*At once*; new default 0: no limit, the budgets
  decide; `--parallel 0` allowed). Project files keep their value, so a project saved with `parallel = 1`
  still runs one script at a time until *At once* is changed; a project file without the key (before
  0.0.7) gets 1, as it did. Without a budget (`run_jobs(budget=None)`, e.g. the wizard's *Write and
  run…*) only the cap counts and the environment is left as it is.
- **Stop scripts** in the planner toolbar (red, sensitive while scripts run) stops every running script,
  as *Cancel* in the status bar does.
- **Stopping one step:** a square stop button in the title bar of every running box, and *Stop this step*
  in a running or queued box's right-click menu, stop that step's script only (`runner.StepStops`; the
  script's process group is terminated, its children too). The step ends as *stopped* (a square marker,
  an amber `==== step 03: … - stopped` line, status `stopped` in the step log), the steps that depend on
  it are not started (`not started (step 03 was stopped)`), the other scripts go on, and the run is not
  reported as failed. The stopped run leaves `<project>_<step>_stopped_<time>.txt` beside the step's
  files, so *Check previous runs* / `mdatk workflow status` call it partial (`partial (stopped)`) until
  the step runs again (the next run removes the note) or a file of the run is written after it.
- Generated files are unchanged (`tests/test_golden.py` passes without new intended changes).

### The full set of decoys on the FARFAR2 box

- **Sets of files in the planner** (`Port(..., collection=True)`): the full set of N files of one kind,
  known only once its step has run, drawn as a large, stacked dot (radius 9 px against 5, a white ring
  inside; hit within 5 px of its edge; its connections thicker). Its output passes a list of the files
  (`PortValue.lists`: plan-relative paths, one per line, and the most it can name), and the scripts of the
  steps that take it read the list when they run (`base.listed_files`; a missing or empty list stops them).
  A set connects only to inputs drawn as large dots, which take files connected one by one as well; onto an
  input that takes a table it connects its step's table (`Port.table`); an output that needs an option of its
  step switches it on when connected (`Port.turns_on`). Hand-written edges of a set into another input are
  an error of the plan.
- **FARFAR2: *Every decoy (PDB)* and *Decoy table*** (section *Decoys*, *Extract every decoy after the runs*,
  off by default and switched on by connecting either output): after the runs the step's script runs
  `<project>_<step>_decoys_<time>.sh`, which extracts every model of every run into `models/<run>/<tag>.pdb`
  (the *Extract every model* machinery of the FARFAR2 models step, shared: a `mktemp` folder per run,
  earlier models removed, one PDB per SCORE line checked), then writes the decoy table
  `<project>_<step>_scores_<time>.tsv` and the list `<project>_<step>_decoys_<time>.txt` (`silent_scores table
  --list`; none listed is an error). It can be run alone to extract again without rna_denovo. The set
  carries FARFAR2's `-input_res` for the segment support of *Decoy interfaces*. Without the option the step's
  files are unchanged (`tests/test_golden.py`); the FARFAR2 models step's copies of `silent_scores` gain
  `--list`.
- **Steps that take the set**: the models of *Model check*, *Decoy interfaces*, *Register census*,
  *Map-model score* and *MDATK · frame check* and both arms of *Ensemble comparison*; their files are
  unchanged without a set. Counts planned from the inputs (passed models, moved models) take at most runs ×
  models per run; the model check's automatic hands are measured on the first listed decoy once it exists.

### Script wizard: the commands a step runs

- **Commands tab** (the script wizard's second tab, stand-alone and in a box's *Edit in wizard…*): what the
  step's run scripts will run, generated from the settings as they are (as the preview is: nothing written,
  no network) and updated with the checks after each change. By default the commands alone, under each
  script's name: the program and helper calls, continued lines joined, here-documents with their command,
  a variable the script sets once to fixed text shown with its value (`$PMEMD` as `pmemd.cuda`). Left out:
  the header and comments, `set`, `cd`, the environment set-up (conda, modules, `source`, `export`),
  assignments, `if` / `for` / `while` structure, checks (`[ … ]`, `command -v`, `grep -q`, the "did not
  write" lines), `rm` / `mkdir` / `mv` / `cp`, `echo` and the error handling after a command
  (`|| { echo "ERROR: …" >&2; exit 1; }`). A script run separately (production) is headed as such.
  *Whole scripts* shows the scripts as written, *Copy* copies what is shown; a step with errors lists its
  errors instead of a preview.
- **`mdatk.workflows.commands.command_lines(script)`**: the commands of a bash script as the tab shows
  them, tested on the generated scripts of the Amber build, Amber MD, FARFAR2, NAMD MDFF and custom script
  steps. Generated files are unchanged.

### Both programs end with their windows

- **A closed RNAdo no longer stays behind as a defunct process.** RNAdo started with *Open in RNAdo…* is
  a child of the analysis window, which looked at it only for its first seconds (to tell whether it
  started): closed while the analysis window stayed open, RNAdo stayed a defunct `mdatk` process (as the
  AppImage names it) until the analysis window ended. The analysis window now reaps it when it ends
  (`mdatk.gui.launch.watch_start` checks it every second).
- **Reported for the AppImage started from the applications menu:** after *Open in RNAdo…*, closing the
  analysis window and then RNAdo left an MDATK process running. Its cause is not confirmed. Reproduced
  with the 0.0.11 code built as an AppImage-like bundle (the PyInstaller bundle and AppRun, a stand-in for
  the AppImage runtime with its FUSE daemon, keepalive pipe and fd 1023, started from a desktop entry),
  that order left no process: both programs and both FUSE daemons ended. The released AppImage itself was
  not run in that test. Two guards are added, neither of which changes what happened in that test:
  - the analysis window and RNAdo quit when their main window closes, so a window attached to the
    application but hidden could never keep the process running (neither program attaches such a window
    today);
  - the programs MDATK starts (RNAdo, the scripts RNAdo runs, `xdg-open`) are started with `close_fds`,
    stated explicitly (it is Python's default on POSIX, so this was already the case), so none inherits
    the AppImage runtime's keepalive pipe or its fd 1023 on the mount; desktop applications opened through
    Gio are spawned by GLib, which closes them too (checked with GLib 2.80). Tests start each of them while
    holding stand-ins of both descriptors.

### Fixes

- **NAMD MDFF**: the fitted PDB's first line is `REMARK 951 MDFF result (NAMD), …` instead of an unnumbered
  `REMARK   MDFF result …`, which ChimeraX ignores as a bad PDB record (example regenerated).
- **Planner file kind *table*** (`.tsv`, `.csv`) for the decoy tables and reports the new steps pass on.

### Fixes from the release review

The 0.0.12 changes were reviewed before the release; what the review found, fixed (each with a
regression test):

**Model checks and Amber builds**

- **Model check: as many passed models as planned, in every mode.** With *Models that fail: report* or
  *stop* the step wrote one passed file per readable model, while the plan names one per connected model
  (counting the empty slots of an earlier filter and the most a FARFAR2 *Every decoy* set can list): with
  a short decoy set or empty slots among its inputs it exited 0 without the last `passed-NN`, the next
  step stopped on "cannot be read" and the check never read as done. The step now passes `--slots N`
  (`mdatk pdb qc --slots N`, report and stop) and writes the slots no model fills as empty slots that say
  why (`empty slot 6 of 6: only 5 models were given`, `only 1 of 3 inputs were models (2 empty slots)`;
  `pdbtools.empty_slot_text` takes the reason, so the Amber build's copy of the PDB tools changes, as
  `tests/test_golden.py` allows); more models than slots is an error. The passed models of a check fed a
  decoy set carry the empty-slot mark.
- **Model check: a filter below its minimum names no files.** It printed `passed on 1 in 3 slot(s),
  slots 2-3 written empty` before `ERROR: 1 of 3 models pass the gates; at least 2 needed`, and wrote
  nothing; the slots are named only when they are written.
- **Empty slots: the help says which steps skip them.** The *Passed models* tooltip, *Models passed on
  (up to)* and the module documentation said that every MDATK step skips an empty slot; they now name
  the readers that do (model check, decoy interfaces, register census, ensemble comparison, frame check,
  map-model score), the Amber build that stops on one, and that a step reading one model takes the first
  slot, which is empty only when a filter with minimum 0 passed none.
- **Amber build: the O3′–P link check quotes its PDB path.** Without *Prepare the PDB* and *Name
  L-nucleotides* a structure path with spaces was split into several arguments: a clean model stopped a
  *stop* build as a break, and a *warn* build skipped the check. Builds without the check are unchanged.
- **Amber build: the shortest O3′–P link must be below the longest unstrained one** (link check on), as
  in the model check: the helper refused the pair at run time, so the script stopped with a false break
  (stop) or built without a check (warn).

**FARFAR2, run audit and frames**

- **Run audit ignores superseded runs**: `mdatk rosetta audit` (and the step *Rosetta · run audit*) no
  longer reads the runs moved aside into a step folder's `superseded/`. A FARFAR2 step run again under
  constant seeds was reported as *DUPLICATE SEEDS* (failing a strict audit), and the old run's flags were
  mixed into the condition's. A superseded run's own folder given as a root is still audited.
- **FARFAR2's extracted decoys move with their run**: with *Extract every decoy after the runs*, writing
  the step again moves `models/` (every decoy and the extraction logs) into `superseded/<stamp>_<time>/`
  together with the decoy list and table that name them, also when the option has been turned off since
  and, in a plan without step folders, the step's own `<project>_<NN>_models/`. It stayed behind, and the next extraction
  deleted it, so the moved list and table named the new run's models.
- **One extraction folder per step without step folders**: in a plan without step folders, FARFAR2
  (*Extract every decoy*) and FARFAR2 models (*Extract every model*, validity filters) extract into
  `<project>_<NN>_models/` instead of a shared `models/`, where two steps removed each other's decoys and
  their decoy lists named the other step's models. Plans with step folders keep `models/`.
- **Base identity numbered by an existing FASTA**: with an empty *Sequence*, the validity filter of the
  FARFAR2 models step reads the FARFAR2 step's *Existing FASTA* with its chain:range tags (`A:1-6`,
  `B:20-25`). The planner renumbered the chains from 1, so every decoy failed and nothing was clustered.
- **FARFAR2's CPU budget counts its extractions**: with *Extract every decoy* the step books the larger
  of its parallel runs and its *Parallel extractions* (at most one per run), which its decoys script runs
  after the runs; it booked only the runs.
- **Frame check reads each decoy once**: with the FARFAR2 set and its decoy table both connected, every
  decoy was measured (and moved) twice and *Decoys of the table* cut only the table's copies. A model
  given that is also a decoy of the table is now that decoy (its run and tag; the cut leaves it out),
  every file is read once, and folders given are never searched inside `superseded/`.
- **Frame check writes every moved model it promises**: `mdatk frame reframe --slots N` writes exactly N
  numbered models, those no model fills as empty slots (no atoms). The step passes the count its output
  *Moved models* lists, so a set with fewer decoys than runs × models per run (or a cut of the table) no
  longer leaves the step *partial*, waiting for files that never appear.
- **Frame check's moved decoys move with their run**: writing the step again moves its folder of moved
  decoys, `<project>_<NN>_reframed_<stamp>/`, into `superseded/` with the manifest and decoy table that
  point into it (with or without step folders). It stayed, and the next run overwrote it.

**Ensembles and map-model scores**

- **Register census: build conditions from rna_denovo logs only.** The `extract_pdbs` logs MDATK writes
  beside extracted decoys (`models/extract-rNN.log`) were read as build logs, so a register the secondary
  structure imposed was reported UNFORCED, or the group UNKNOWN, and 60 of them filled the cap on the
  logs read; the cap now counts rna_denovo logs only.
- **Register census counts each decoy once** when the decoys are connected and *Also count the decoys the
  decoy table lists* is ticked (a relative and an absolute path of one file were counted twice).
- **Ensemble comparison (pilot): seeds as Rosetta used them.** The seed check compared the `-jran` text;
  it now reads seeds as the run audit does (the logged `real_seed`, `-constant_seed`, `-seed_offset`), so
  a run without `-constant_seed` (Rosetta then ignores `-jran`) is reported as random, not matched, and
  another `-seed_offset` as another seed.
- **Matched runs need the decoy table.** `--matched-runs` without the table's `run` column, or with no
  run in both arms, stops with an error instead of comparing 0 decoys, and the planner asks for the decoy
  table.
- **A reference typed as FASTA** in the census and comparison steps is written beside the script and
  passed with `--fasta` (run together into one shell word, it failed with "the reference sequence is
  empty").
- **A matched-groups suffix starting with `-`** (`pres` / `pres-free`) works: the census script passes
  `--pair-suffix=-free`, which argparse no longer takes for an option.
- **Map-model score and body hybrids: `qscore/` moves with a superseded run** (also after the method was
  changed), so the ChimeraX logs its report names stay those of that run; in a plan without step folders
  each step writes them into its own `<project>_<NN>_qscore/` instead of one `qscore/` all steps shared.
- **Model searches skip `superseded/`**: `mdatk ensemble census -r` and the folders given to `mdatk
  decoys` no longer count the decoys of runs RNAdo moved aside with the current ones.
- **The ChimeraX time limit ends ChimeraX behind a launcher**: ChimeraX runs in a process group of its
  own, which the time limit (`--timeout`) ends whole (`xvfb-run -a chimerax` with the ChimeraX it
  started); a SIGTERM or SIGHUP to MDATK (RNAdo's *Stop*) ends that group too.
- **`mdatk ensemble regions` leaves out tables ChimeraX stopped inside** (no `Overall mean Q-Score` line:
  the log of a failed model, which `map score` keeps): they are listed as not ranked and the command ends
  with status 1, so a failed model no longer ranks on its truncated table.
- **`mdatk map score` of empty slots only** says that every input is an empty slot of a model check (it
  failed with a blank reason).

**Decoy interfaces and DSSR**

- **DSSR engine: stacking inside stems.** DSSR's `stacks` list leaves out stacking within stems, so
  `mdatk pdb annotate --engine dssr` wrote no STACK record for any stem (and the refinement's *Stacking
  from the annotation* restrained loops and junctions only). The stacking inside each stem is now read
  from its consecutive pairs (`stems[*].pairs`, neighbours along each strand); checked on DSSR 2.9.4 JSON.
  Cross-strand stacking inside a stem (only in DSSR's `--non-pair` list) is still not reported.
- **DSSR engine: a relative executable path works.** `--dssr-exe bin/x3dna-dssr` or `./x3dna-dssr` (and a
  planner step's path relative to its folder) failed with 'not found' because DSSR runs in a temporary
  folder; the path is now made absolute first.
- **Decoy interfaces: unreadable FARFAR2 fixed residues are a step error.** With *Rebuilt residues: not
  fixed in the FARFAR2 step*, an `-input_res` MDATK cannot read (`A1-6 B:7-8`) passed every check and then
  aborted the whole `workflow write`; the step now reports it and is skipped.
- **Decoy interfaces: score filters keep the typed threshold.** `--filter` and the census log rounded it
  to 6 digits (`score <= -1500.125` ran as `<=-1500.12`); the exact value is written now.
- **`mdatk pdb interface`: models with the same file name stay apart.** Decoys of different runs
  (`r1/S_1.pdb`, `r2/S_1.pdb`) got the same `model` value and overwrote each other's `--keep` files; they
  are named by path where file names repeat, as in the census, and kept as `r1_S_1_dssr.pdb` and so on.
- **Decoy clusters: exact cut and medoid ties.** Fingerprints exactly at the Jaccard cut (e.g. 9 of 10
  shared at 0.1) were split into two clusters, and exact medoid ties could pick a later member instead of
  the seed (float32 rounding); distances are computed in float64 now.
- **Segment support: strands without chain ids.** In files with blank chain ids (Amber / cpptraj frames)
  consecutive residue numbers across a strand break counted as backbone neighbours, so real contacts were
  dropped and segments ran across strands. A strand now ends at a chain change, a TER record, a 5' end
  without its phosphate or an O3'-P gap over 4 Å; such a file gives the same support as with chain ids.

**RNAdo, GROMACS and menu entries**

- **GROMACS steps that name their GPU** with `-gpu_id` or `-gputasks` among the extra mdrun flags get no
  `CUDA_VISIBLE_DEVICES` from RNAdo or `workflow run` any more: the ids count as the step's own GPU
  (among the GPUs its Environment leaves visible, as NAMD's `+devices`), so `-gpu_id 1` on a two-GPU
  workstation runs on GPU 1 as in 0.0.11 instead of stopping mdrun on an invalid device id.
- **A run that completes while RNAdo asks which completed runs move aside** is no longer moved: only the
  runs the question listed move, and a step whose run completed while the question was open is not
  written (the output log and a message say so; write it again), so a step that reads that run keeps its
  inputs.
- **The Commands tab** no longer lists the case patterns of the set-reading loop (`/*) MODELS+=("$f")`,
  `*) MODELS+=("../$f")`) as commands of the steps that take a set of models; it shows their program
  only.
- **Decoy validation template:** the FARFAR2 models step's decoy table is no longer connected to the
  model check and the decoy-interface step. It lists every decoy by its extracted file, not the cluster
  centres those steps check, so their score columns stayed empty and score limits failed every model.
- **Menu entries of a program whose path holds a `%`** (an AppImage in a folder such as `pct 100%`) now
  appear in GNOME-family menus: the program starts through `env` (`sh` when the path also holds a `=`).
  Before, `mdatk desktop install` reported success and GLib dropped both entries.
- **`mdatk --help`** names ChimeraX's Q-score for the `map` command, not Phenix's.

## 0.0.11 (2026-10-31)

### RNAdo, a program of its own

- The workflow planner, until now the *Project Planner* of the analysis window, becomes **RNAdo, MDATK's
  workflow planner**, a program of its own (`mdatk/gui/rnado.py`). The **script wizard**, the **PDB
  clean-up** and the **structure builder** moved with it from the analysis window into its header bar.
- RNAdo's window holds the plan canvas with its toolbar and details pane, the output log panel (below)
  and a status bar with the planner's status line, the progress bar and *Cancel*. Header bar: *Open*
  with the recent projects, *Save*, the three tools; its menu has *New project*, *Open project…*,
  *Save*, *Save as…*, the tools, *Show the output log*, *Project log…* and *About RNAdo*. Shortcuts:
  Ctrl+N, Ctrl+O, Ctrl+S, Ctrl+Shift+S, Ctrl+Shift+W (wizard), Ctrl+L (log panel).
- RNAdo opens, saves and remembers the same project files (`*.mdatk.toml`) as the analysis window and
  runs without it. Closing RNAdo during a run asks first and stops the scripts.
- The planner, the wizard, the PDB clean-up and the builder use only a small host interface
  (`mdatk/gui/host.py`) and no longer know the analysis window. Background jobs and the run
  confirmation (`mdatk/gui/jobs.py`) are shared by both windows; the runner can mark every line with
  its step and pass standard error apart (command-line output unchanged).
- **Output log panel** (`mdatk/gui/logpanel.py`) at the bottom of RNAdo's window: the planner's
  messages (files written, previous runs checked, runs used or dropped) and every line of the scripts
  it runs, each marked with its step (`[03] …`), with start, finished, failed, cancelled, waiting and
  not-started lines. Standard error is red, lines starting with ERROR (and failed steps) bold red,
  lines starting with WAITING (and waiting steps) amber. *Step* filters by step, *Last run* shows the
  stored log of the selected step's last run (from the project's step log); the panel follows new
  lines, copies, saves or clears them, folds away (Ctrl+L) and unfolds for a run. Lines are inserted in
  batches, the panel keeps the last 5 000–6 000 lines, and a script that prints faster than the panel
  can show gets lines skipped, with a note of how many, so the window stays responsive.
- **Starting RNAdo:**
  - `rnado [PROJECT] [--open wizard|pdb|builder]` and `rnado-gui` (the same without a terminal), new
    entry points of the source installation; `rnado ACTION …` is `mdatk workflow ACTION …` (`tools`,
    `check`, `write`, `run`, `status`, `generate`, `template`, `restraints`);
  - `mdatk rnado …` (the same as `rnado …`), `mdatk gui --rnado` (alias `--planner`) and
    `mdatk gui --open TOOL`, `python -m mdatk.gui.rnado`;
  - in the analysis window, the header-bar menu *RNAdo*: *Open in RNAdo…* (Ctrl+P), *Script wizard /
    PDB clean-up / Structure builder: open in RNAdo…* (Ctrl+Shift+W for the wizard). It saves the
    project first when it has unsaved changes (it asks), then starts RNAdo on the project file
    (`mdatk/gui/launch.py`).
- **The plan's files for the analysis:** *Use the plan's files* (analysis window, Input & cleaning;
  *Use the plan's files as inputs* in its *RNAdo* menu) takes the topology and trajectories that the
  plan's MDATK analysis step reads, from the plan saved in the project file; with several analysis
  steps it asks which. *Use for analysis…* on such a step in RNAdo saves the plan for it.
- **Both programs on one project** (`mdatk/gui/projectui.py`): RNAdo writes only the file's plan (the
  `workflow` section, `projectfile.edit_workflow`) and leaves the analysis settings as the file has
  them; the analysis window saves the plan it finds in the file with its own residue-group renames
  applied, and RNAdo follows those renames in its unsaved plan. Each program notices when the other
  saved (focus, file monitor), takes the other's part, reloads its own when it has no unsaved changes
  there, and otherwise asks in a bar above its content.
- Generated workflow files are the same as in 0.0.10, apart from the version in their headers.

### Logo, icons and stand-alone builds

- RNAdo's logo, wordmark banner and icons (`mdatk/assets`: `rnado-logo.png`, `rnado-wordmark.png`,
  `rnado-icon.svg` and PNGs from 16 to 512 px, 0.8 MB in all) are drawn by
  `packaging/make_rnado_art.py` from a real A-form RNA model: the ideal duplex of `mdatk.builder`,
  read back with `pdbtools`, drawn as P-atom ribbons and C4′–C1′–N1/N3 rungs with beads. The helix
  keeps its geometry: 11 bp per turn, right-handed, pairs 5 Å off the axis and inclined, a narrow, deep
  major groove (11.3 Å) and a wide, shallow minor groove (19.6 Å); the tornado funnel only tapers the
  radius (twist, rise and grooves kept). Reds, red-orange highlights, white and black; the wordmark's
  letters are outlines of Inter Black Italic (SIL Open Font License 1.1, no font file shipped). The
  script is seeded and reproducible; the images are committed, so neither the GUI nor the builds draw
  anything. The wordmark heads the README.
- The stand-alone builds have RNAdo as a second program: `rnado-gui` (windowed; `rnado.ico` on Windows)
  from `packaging/entry_rnado.py`, with `mdatk/assets` bundled. AppImage: `AppImage rnado ARGS` and a
  link named `rnado` start it; the MDATK desktop entry has an *RNAdo (workflow planner)* action, and
  `rnado.desktop` and the hicolor icons are in the AppDir. Windows installer: Start-menu entry *RNAdo*,
  an optional desktop shortcut and an optional launch. `data/rnado.desktop` for source installations;
  `assets/*` is package data. `packaging/smoke_test.py --rnado` starts RNAdo, and the release workflow
  runs it on the Linux jobs.

### Fixes from the release review

- Co-editing:
  - the analysis window's *Save* asks its questions first, then re-reads the file, replays its
    residue-group renames on the plan RNAdo saved and writes, all under one file lock (re-entrant
    within the process). A plan RNAdo saves while *Replace them?* is open is kept; when the analysis
    settings change again meanwhile, the question is asked again;
  - RNAdo's save re-reads the file, takes newer analysis settings, writes the plan and records the
    file state under one lock, so the state it keeps is the one it merged with and wrote;
  - *Reload them* in the analysis window's change bar loads the file as it is: the window's pending
    group renames go with its other unsaved edits instead of being replayed on the plan;
  - RNAdo works out the group renames from the old and new group lists when it takes new analysis
    settings, follows them in its unsaved plan, in the plan last read and in the saved one, and reloads
    the forms of open wizards. A plan changed only by those renames is no conflict;
  - *Save* without edits in the analysis window, when another program saved newer analysis settings,
    reloads them ("Nothing to save") instead of writing the older copy back; *Save as* gives the copy
    the newest settings;
  - *Use the plan's files* keeps the file state, the pending change bar and the settings it offers; a
    save after a shown bar replaces those without asking again, newer ones are asked about;
  - opening a script wizard (stand-alone or a step's) and choosing a tool no longer marks the plan as
    changed, although the wizard fills in defaults.
- Launch and runs:
  - scripts and programs that RNAdo runs get **your own `PYTHONPATH`**: *Open in RNAdo…* from a source
    checkout puts MDATK's folder on RNAdo's `PYTHONPATH` only when it is not a site-packages folder and
    records your value (`MDATK_ORIG_PYTHONPATH`), which `frozen.child_env` gives back, so another Python
    a step runs (ParmEd, a conda environment) no longer imports MDATK's packages;
  - the **output log panel is bounded**: a flush inserts at most 2 000 lines, the newest, with a note
    per step of how many were not shown, and a flooding script pauses 1 ms per 2 000 lines. `seq 1
    1000000` run from RNAdo stalls the window for about 0.1 s at most (was 24 s), and *Cancel* is
    handled on time;
  - **AppImage `rnado` actions**: `AppRun` runs `mdatk rnado …` for a first argument `rnado` and for
    links named `rnado*`, so `AppImage rnado write|run|status|tools …` work; the window opens in that
    process with the program name `rnado`, so its WM_CLASS is RNAdo's;
  - **how RNAdo is started** follows how MDATK runs: the AppImage again only when MDATK itself runs
    from it (its executable lies in `$APPDIR`), with `APPIMAGE_EXTRACT_AND_RUN=1` (and `NO_CLEANUP=1`,
    so RNAdo leaves the shared extracted folder in place) when the AppImage runs extracted, without
    FUSE; otherwise the `rnado-gui` next to the program, or `python -m mdatk.gui.rnado`. The start is
    watched for 3 s: an error status is shown in the analysis window instead of "Started RNAdo", and a
    window closed meanwhile is left alone;
  - **run tokens**: the planner gives a run its token only once the run has started, so a refused
    *Run step…* no longer orphans the markers and spinner of the run in progress.
- The analysis window cancels its pending settings check when it is destroyed (Gtk-CRITICAL warnings).
- Tests: the example test (`test_example_generator`) compares the shipped `examples/namd_mdff` without
  the date in script headers, so it **no longer fails on every day after the example was generated**;
  the test of chain reads replaced while running waits for the read to start.
- `examples/namd_mdff` regenerated (version in the headers).
- Help texts, docstrings and project-file comments say RNAdo for the planner: `mdatk workflow` is "the
  plan of RNAdo, MDATK's workflow planner, and its script generators", and the `workflow` key of a
  project file is commented "RNAdo's plan and script wizard".

## 0.0.10 (2026-10-30)

### L-RNA names before tleap; Windows cursor

- With an own L-RNA library, the Amber build script checks the L residue names **before tleap runs**:
  a standard-library helper (`scripts/lrna_names.py`) reads the L names of the renamed PDB, the units of
  every library the tleap input loads before `loadpdb` (a name in leap's search path is looked up in
  `$AMBERHOME/dat/leap`) and its `addPdbResMap` lines (also over several lines, comments ignored), and
  stops on a name that is neither a unit nor mapped, which tleap would read as the lanthanum or
  lutetium ion (LA / LU). A library it cannot read gives a warning. The planner uses the same parser
  and warns about the names even when an earlier step makes the structure.
- These checks come only with an L-RNA set-up: *My own files* under *L-RNA library*, a structure known
  to have L residues (*Force L for*, an L helix, an input PDB with L sugars), or an own library that
  looks like an L-RNA one (atom types LD / LH / LE, or nucleotide units named the L-RNA ways: LA…,
  A0…, 0A…). D builds with, say, a ligand library, and modified-nucleotide D libraries such as
  `all_modrna08.lib`, are written as before (none of AmberTools' 86 libraries triggers the check;
  Dudek & Trylska's `nucleic12_LRNA.lib` does, and passes).
- In the Windows build the mouse pointer no longer disappears over the structure panel: GDK there maps
  only some cursor names to system cursors and the build leaves the cursor theme out, so
  `mdatk/gui/cursors.py` asks for cursor types with Win32 system cursors (elsewhere the named cursors,
  with the type as fallback).

### Previous runs and separately run production

- **Check previous runs** (planner toolbar, right-click menu, details pane, `mdatk workflow status`)
  finds the outputs of a step's earlier run, also under an earlier time stamp (each iteration of a
  project may get a new one), and marks the box **done**, **partial** or **missing**. Only the files
  the step's scripts make count; Amber MD stage outputs must also end with pmemd's closing lines, so a
  run in progress is partial (`mdatk/workflows/runstate.py`).
- A step can then **use those outputs** (*Use this run*; `adopted` in the project file): the steps after
  it read them, and it is not written or run again (the master script and the summary say so). *Stop
  using it* undoes it; *Run selected steps* / *Run all…* ask whether to skip such steps; the planner
  warns when a file of the run disappears. `mdatk workflow status [--step N] [--adopt | --forget]
  [--json]`; `workflow run` skips steps that use an earlier run and says so.
- A step whose production runs in a separate script no longer reports success before that script has
  finished. Before 0.0.10 the main script of an Amber MD step whose stages were all production held
  only the preamble and "succeeded" without running anything. Now the main script runs the stages
  before production (for GROMACS also the conversion) only while production has not finished, and
  **waits** (exit status 75, naming the script to run); the steps after it do not start, and
  `mdatk workflow run` exits with 75. Once production has finished, the stages after it and the dry
  PDB run and the step succeeds. A main script written by an older version is treated as waiting.
- `mdatk workflow run / write --run --include-separate` and a check box in the run confirmation run the
  separate scripts too, each right after its step's main script (not again once finished).

### Residue groups and located errors

- **Residue groups follow their names and the topology.** Renaming a group renames it everywhere the
  project uses it (analysis groups, selections, panels, charts, representations, PDB export, planner
  steps; `group:Older` and `group:Old_2` stay; notes are never rewritten); empty, invalid and taken
  names are refused. Deleting a group that is still used says where. Choosing a topology fills empty
  or untouched default groups with one group per chain (`mdatk/chaingroups.py`), and *Groups from
  chains* adds the chains no group covers yet. `Project.rename_group` and `Project.group_references`.
- **Selection and group errors say exactly where they are.** Every problem names its setting
  ("Analyses › Hydrogen bonds › Group B: unknown residue group 'Tagret' (defined: Aptamer, Target);
  did you mean 'Target'?"), shows MDAnalysis' own message next to the selection as written and as
  expanded, and says which group or part of a selection matched no atoms (`mdatk/problems.py`:
  `Problem`, `SelectionError` with field path, page, location and token; `Project.problems()`).
- The GUI checks the settings as you type: red count badges on the page buttons, flagged fields with
  the message as tooltip, tinted table cells, and a problems list that jumps to the field.
- **Only what is about to run can block it**: a wrong setting is an error if the run uses it and a
  warning otherwise. The structure view's settings are errors only while the view is rendered, so a
  0.0.7 project with a stale proximity set renders charts and PDBs with the video off again. A group's
  own definition is an error only when something about to run selects it; circular definitions are
  reported at every group of the circle; a failing run on a panel's or chart's groups is reported at
  that panel or chart; a PDB atom subset of only commas is a located error instead of a crash.

### Custom scripts

- **Custom scripts as planner steps** (`user-script`, *Custom*): type a bash or Python script into a
  step, or pick a script file, declare the files it reads and writes, and connect them to other steps
  like any program's files. MDATK copies the script into the step folder, passes the file names as
  `MDATK_IN_<NAME>` / `MDATK_OUT_<NAME>` and `{in:…}` / `{out:…}` arguments, and stops the step when a
  declared output is missing (`mdatk/workflows/userscript.py`).
- The run script removes the declared outputs before the script starts and writes a stamped completion
  marker last, so a stale or partial output never passes; a fixed-name output belongs to the stamp
  whose marker is there. Errors for two steps that write the same output file and for an output that is
  one of the step's own inputs; plain file names may not start with `+`; an optional input with
  several connections and no file expands to no word.
- Wizard: taller boxes for multi-line options (`Option.rows`); the *Files* page follows input ports
  that change through a record list.

### Other fixes

- Chain groups: a chain read or topology timer no longer reaches a project loaded since; renames and
  chain fills retitle only panels with automatic titles (default titles such as "RMSD" stay); *Groups
  from chains* only adds groups; group lists with empty items are renamed and found. VMD keeps chain
  qualifiers and translates chain-group selections, so chain groups colour their own chain.
- `examples/namd_mdff` regenerated (headers, `adopted` field).

## 0.0.8 (2026-10-29)

### MDFF in NAMD on the Amber topology

- New planner step **NAMD · MDFF (Amber topology)** (`namd-mdff`, category *Refinement*): MDFF in NAMD
  on the Amber topology. NAMD reads the parm7 as built (`amber on`, `parmfile`, `ambercoor`), so CP,
  NBfix19, 0BPh, the L-RNA types and dihedrals, the Mg²⁺ pair terms and HMR are kept. A sander / NAMD
  energy comparison proves it before fitting; a mismatch stops the script.
- Solvated (periodic, PME, LJ correction, NVT; default) or vacuum (dry topology, optional GBIS) runs;
  stages: minimisation, MD with a ramp of the map coupling, MDFF and a final minimisation with strong
  coupling. Finished runs are skipped while their inputs are unchanged.
- Its chirality and base-pair restraints **keep L-RNA sugars L**; it never uses VMD's D-only
  stereochemistry tools. The secondary-structure dihedral restraints of the MDFF paper (Trabuco et al.
  2008) hold the torsions of base-paired residues; base pairs are flat-bottom `extraBonds` walls. All
  restraint files are NAMD `extraBonds` files.
- Preparation helper `scripts/namd_mdff.py` (standard library only): reads parm7 / rst7, MRC / CCP4 and
  OpenDX maps, finds the map's frame after tleap moved the solute, writes the potential (cropped and
  tested against the periodic cell exactly as NAMD tests it), the coupling PDB, the restraints and the
  cell; the final model returns to the map's frame with the chains and residue numbers of the
  map-frame model.
- Template *Model + map → Amber build → NAMD MDFF* (rectangular box; NAMD fits the potential into a
  truncated octahedron only with a small margin, and the planner warns about one).
  `examples/namd_mdff/` holds a D + L test case.
- Citations: NAMD (Phillips et al. 2020); MDFF (Trabuco et al. 2008) verified from the paper.

### Generated L-RNA library

- The Amber build can **generate the L-RNA library** from the force field in use (OL3 or OL3 + CP;
  option *L-RNA library*: automatic, generated or own files). The D units are mirrored and named LA LC
  LG LU (LA5 / LA3 / LAN …) inside and out, O5′ and O4′ get their own types (LD, LH; LE with CP), and
  every term of their D twins is repeated with dihedral and improper phases mirrored. A D duplex and
  its mirror image are tested to the same energy. The generator refuses odd-phase terms without a
  twinned type and the generated library with an own CP residue library; *Automatic* leaves own L-RNA
  set-ups alone, so 0.0.7 projects build as before.
- The topology check counts the residues carrying the L types against the L residues of the PDB that
  the final tleap read: D units loaded for L residues stop the build (tleap really loaded L units).
- **Lanthanide trap:** a library whose units are named otherwise (A0 / C0 …) and no `addPdbResMap`
  made tleap build LA and LU as lanthanum and lutetium ions without any warning. The planner reads the
  extra libraries' unit names and the map lines and reports every unmapped L name as an error with the
  map lines to add; the topology check stops on a residue named as a nucleotide with fewer than 10
  atoms, now also for dry L-RNA builds.
- `namd-mdff`: *L residues* declares the L residues by topology number for libraries that write D
  names into their units.

### Regression test and examples

- Steps that existed in 0.0.7 write the same files as before: `tests/golden.py` records every file the
  planner writes and every planner message for every template, every step type (each option changed
  one at a time), typed secondary structures and the example projects (193 cases), and
  `tests/test_golden.py` compares them with the 0.0.7 output. Intended changes: the topology check
  (and the dry L-RNA build script that now runs it).
- `examples/rna_plan.mdatk.toml` and `examples/aptamer_target.mdatk.toml` had been committed empty in
  0.0.4 (and shipped empty since); restored from 0.0.3d.

## 0.0.7 (2026-10-25)

- Independent steps can **run at the same time** (*Parallel* in the planner toolbar, 1–64, saved as
  `workflow.parallel`, default 1; `mdatk workflow run / write --run --parallel N`, checked before any
  file is written). Each step starts when the steps it reads from (through the planner's connections)
  have succeeded; plan order otherwise.
- A failure stops the steps that depend on it; independent steps carry on. Cancel or Ctrl-C stops every
  running script; the final error names each failed script and why. An error of the output itself
  (e.g. `| head` closing the pipe) ends the run instead of hanging. Output lines carry the step number
  when scripts run side by side; the command line writes each line in one piece.
- A box shows an **animated marker** while its script runs: a turning arc, a dashed ring while it waits,
  then a tick, a cross or a dash until the next run. Marks of a run started before a project reload, or
  after the planner closed, are ignored.

## 0.0.6 (2026-10-25)

- The Project Planner can **run a selection of steps**: tick the boxes (check box in the title bar,
  Space on the selected box, the right-click menu, *Tick all steps* / *Untick all steps*), then *Run
  selected steps (N)*: it writes the ticked steps' files and runs their main scripts in plan order,
  whatever order they were ticked in. Notes and the MDATK analysis box have no check box; the ticks are
  not saved. *Run…* is renamed *Run all…*.

## 0.0.5.1 (2026-10-25)

- *Separate HO and O2 Lennard-Jones types first* is on by default for new projects: tleap merges HO with
  water HW, so the NBfix19 HO pairs otherwise apply to every water hydrogen. Projects saved earlier keep
  their setting; turned off, the planner warns.
- On AmberTools 26 the `MG` unit carries type `Mg2+` with charge +2 (checked on a built nanoMg / TIP3P
  topology), so earlier versions applied the published ion terms too; tleap's "MG has a charge of 0"
  message is misleading.
- Two-pass builds leave Mg²⁺ lines under extra ParmEd commands out of the dry pass.
- Topology check: solvent is classified by residue, never by atom type name; the log lists the
  Lennard-Jones types shared by solute and water atoms.

## 0.0.5 (2026-10-25)

- The Amber build **checks the finished topology** (*Check the final topology*, on by default) and stops
  when a box meant to be neutral is not, fewer Mg²⁺ than requested are present, a Mg²⁺ does not carry
  +2, or (microMg / nanoMg) the ion lacks its published Lennard-Jones terms or a solute atom type has no
  Mg²⁺ pair term.
- L-RNA and other non-standard atom types get the Mg²⁺ pair terms of their D-RNA twins (*L-RNA atom
  types*: LD and LE as OS, LH as OH; *Extra Mg²⁺ pair types* for any library). Before 0.0.5, with
  microMg / nanoMg, the combination rule bound Mg²⁺ to the oxygens of an L-RNA library several times
  too strongly.
- The microMg / nanoMg frcmod defines the ion's terms for both type names in use (`Mg2+` and `MG`).
- The L-RNA fixes (atom types, NBfix19 / 0BPh and Mg²⁺ pairs) no longer depend on MDATK naming the L
  residues: an input PDB may already have L names.
- The Mg²⁺ block only adds `changeLJPair` lines, so pair terms set earlier under extra ParmEd commands
  are not reset; planner errors and warnings for types without a reference.
- Documentation: the plan for a future C++ version (`docs/ARCHITECTURE_CPP.md`) updated.

## 0.0.4.1 (2026-10-22)

- The Linux AppImage is built on Rocky Linux 9 (glibc 2.34) with Rocky's Python 3.12 and GTK 3; the 0.0.4
  build needed glibc 2.35 and did not start on RHEL 9. The glibc check now fails the build.
- The AppImage carries the Wayland client libraries in `_internal/fallback` and uses them only on hosts
  without them (minimal or server installs).
- Smoke tests on a clean Rocky Linux 9 and on Ubuntu 22.04; the Windows build keeps MDAnalysis' C
  diagnostics as warnings for GCC 14+. Releases get files only when every build and test passed.
- Planner: selecting a step no longer scrolls the canvas back to the top.
- Citation of PDB 4JRD (Safaee et al. 2013) verified against the paper.

## 0.0.4 (2026-10-21)

- The structure builder also makes **parallel and Hoogsteen-paired duplexes and triplexes**:
  reverse Watson-Crick, Hoogsteen and Hoogsteen/Hoogsteen (4JRD-type) parallel duplexes,
  reverse-Hoogsteen antiparallel duplexes, and triplexes with a Hoogsteen (U·A-U, C+·G-C) or
  reverse-Hoogsteen (A·A-U, G·G-C) third strand; D or L, mismatches reported, C+ written as C.
- PDB clean-up: `--strip-5p` (*Remove 5′-terminal phosphates*) for Amber.
- MDATK runs as a **stand-alone AppImage or Windows program** without Python (PyInstaller, with Barnaba
  and FFmpeg bundled; the Windows build is a beta).
- Value fields (spin buttons, drop-downs, sliders) no longer change when the mouse wheel scrolls past
  them; the canvas keeps wheel zoom.

## 0.0.3d (2026-10-20)

- The Amber build declares the L-RNA library's atom types (LD, LH) for tleap and adds the NBfix19 / 0BPh
  pairs for its bridging-oxygen type, as in the validated hand-written L-RNA ParmEd input; ParmEd skips
  them without L residues.
- Citations of Arnott 1972, Phenix 2019, Coot 2010, AceDRG 2017, eLBOW 2009, Afonine 2018, Casañal 2020,
  CCP4 2023, CCD 2015 and PubChem 2023 verified against the papers.
- `docs/ARCHITECTURE_CPP.md` and `LICENSING.md`: the plan for a future C++ version and its licensing.

## 0.0.3c (2026-10-18)

- MDATK **builds starting structures** (`mdatk build`, the structure builder window, a planner step):
  ideal A-form helices of either chirality from the Arnott fibre monomer, base mutations with
  compensatory partners, rigid-body fits into a map.
- It writes **restraint dictionaries** for the L-RNA terminal residues Coot and Phenix do not know
  (acedrg / phenix.elbow from built-in L SMILES, renamed to the model's atom names; `mdatk dict`;
  PubChem SMILES only on explicit request).
- Planner steps for **real-space refinement** with Phenix 1.21 / 2.x and Coot 0.9 / 1.x (batch), with
  checks for names, terminal variants and dictionaries; templates *ideal helix → Amber MD* and *model +
  map → dictionaries → Phenix → Coot*.
- microMg is available with TIP4P-D as well (the authors' parameter-file values; N. Schwierz confirmed
  that the paper tables carry transcription errors).
- `ccd` scheme: a 5′ end that keeps its phosphate keeps the CCD component name (0G).

## 0.0.3b (2026-10-18)

- MDATK finds **base pairs** in every *n*-th frame (builtin canonical-pair detection or the optional
  [Barnaba](https://github.com/srnas/barnaba)) and shows them in a **secondary-structure arc panel**; a
  **sequence panel** spans the width of the video; both glow and colour with the structure view.
- The structure view can be **zoomed and centred** on a residue or the origin, saved in the project
  (mouse-wheel zoom in the GUI).
- `mdatk pdb annotate` writes base pairs, stacking and dot-bracket into a PDB file as REMARK 950
  records.
- The real-frame preview shows real residue names, chain ids and base pairs.

## 0.0.3a (2026-10-18)

- A project is **one commented TOML file** (`*.mdatk.toml`) holding the settings, the planner workflow,
  notes and a log of every step that ran (moved to `<name>.log.toml` beyond 50 MB); JSON presets are no
  longer read. Generated scripts can be run on request, each run logged in the project file.
- MDATK repairs and renames **PDB files** (ChimeraX damage, L-RNA detected from the sugar chirality,
  Amber / CCD / Rosetta names, chain ids, numbering, SEQRES); this replaces `to_lrna.sh` in the Amber
  build.
- Every script, run log and the *About* dialog **cite** the programs, force fields and parameter sets the
  project actually uses (BibTeX export).
- Amber: OL3 + CP retyped residue library; TIP4P/2005 water with microMg / nanoMg and the authors' Cl⁻.
  The default build (CP library, NBfix19, 0BPh, HMR) and a CP + TIP4P/2005 + nanoMg build were run with
  tleap and ParmEd, and the default build was converted to GROMACS and run with grompp / mdrun.
- GUI: icon bar and a permanent 520 px side bar; *Open project* drop-down with recent projects and the
  demo; nothing loaded at start-up; file choosers name their extensions; PDB clean-up window, project
  log, About with the project's citations.

## 0.0.2c (2026-10-16)

- **3D representations** of residue groups: VMD-style cartoon (New Cartoon with base slabs), ribbon,
  tube, licorice, VDW spheres, surface and trace, drawn by a built-in numpy z-buffer renderer (no OpenGL,
  headless too) or the optional VTK renderer; glow, fog and translucency.
- **Density maps** (MRC / CCP4): a simulated map, a rigid-body fit of the model (locally normalised FFT
  search, refinement) and a projected shadow behind the structure view; `mdatk density`.
- Optimised 12-6 Mg²⁺ models **microMg / nanoMg** in the Amber build, carried over to GROMACS.

## 0.0.2b (2026-10-15)

- MDATK also **writes the inputs and scripts that come before the analysis**: a script wizard and a
  free-form **Project Planner** generate consistently named control files for Rosetta FARFAR2, SimRNA,
  RNAMake, Amber (tleap / ParmEd / pmemd), GROMACS and VMD, from sequence to MD ensemble, with logic
  checks (errors, warnings, infos); files are named `<project>_<step>_<tool>_<time>.<ext>`; command
  line `mdatk workflow`.
- Amber defaults reproduce the reference protocol (OL3 + CP, NBfix19 / 0BPh, SPC/E, min1 → production);
  the GROMACS conversion keeps the off-diagonal Lennard-Jones pairs that ParmEd and acpype drop.

## 0.0.2a (2026-10-15)

Package version 0.0.2.

- The main window is the figure: the same panel code as the video, at its aspect ratio, filled with
  example data or the real reference frame; drag panels to swap them, drag the structure to rotate it.
- Data keys may name their own groups (`contacts@Target+Mg/atom_pairs`); C4′ backbone RMSD; whole-number
  ticks for counts; residue-number tags; framing that never zooms on rotation.
- cpptraj inside a conda environment (`--conda-env`); one timestamp per run on output names.

## 0.0.1 (2026-10-12)

- First release: an offline, Linux-native GTK 3 application with a companion command line that analyses
  MD trajectories of RNA systems (cpptraj cleaning; RMSD, contacts, H-bonds, O–Mg–O bridges and Mg
  occupancy, π–π stacking, intermolecular energy in one streaming pass; cached results) and renders
  multi-panel MP4 videos, time charts and selected-frame PDB files; synthetic demo system.
