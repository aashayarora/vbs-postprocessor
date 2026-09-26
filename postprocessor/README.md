# Candidate post-processor

Per-variation boson-candidate reconstruction for the reconstructed VBS VVH channels,
run as a **separate stage on top of the preselection output** so the preselection code
(and its systematics bookkeeping) stays untouched.

The preselection already:
- selects good objects and stores, for nominal **and every JES/JER variation**, the
  good-jet masks (`FatJet_isGood[_<sfx>]`, `Jet_isGood[_<sfx>]`), varied kinematics
  (`*_pt/mass_<sfx>`), and
- runs the **VBS BDT** for every channel on the FJ-cleaned AK4 jets, storing the tagged pair as all-jet-frame
  indices + score (`vbs_jet1_Jetidx[_<sfx>]`, `vbs_jet2_Jetidx[_<sfx>]`, `vbs_score[_<sfx>]`),
- stores the per-variation channel-pass flags `passes_<channel>_{nom,<sfx>}`, and
- stores the GloParT discriminants `FatJet_Hbb` / `FatJet_Wqq` and the regressed mass
  `FatJet_massGloParT3`.

This stage reads that ntuple and, for each variation, rebuilds the good-object
collections from that variation's masks/kinematics and performs the **boson-candidate +
resolved + m_lb** assignment ported from the old 1lep-cutbased selections. The tagger
scores and the regressed mass are taken from the preselection as-is (all three are
JEC-invariant — see "Input branch names" below), and the VBS pair is taken from the
preselection's stored indices — **the BDT is not re-run here.** Candidates for all
variations are written to one long-format parquet per channel with a `variation` column.

## Input branch names

The preselection's AK8 tagger branches were renamed and the regressed mass redefined;
this stage follows the current names:

| quantity | branch | notes |
|---|---|---|
| H discriminant | `FatJet_Hbb` (was `FatJet_HvsQCD`) | `Xbb / (Xbb + QCD)` |
| V discriminant | `FatJet_Wqq` (was `FatJet_VvsQCD`) | `(Xqq/3 + Xcs) / (Xqq/3 + Xcs + QCD)` |
| regressed mass | `FatJet_massGloParT3` (was `FatJet_globalParT3_mass`) | `massCorrX2p * _FatJet_rawmass` |
| good-fat-jet count | `nfatjet` (was `nFatJets`) | `njet` (FJ-cleaned) unchanged |
| lepton count | `nLep_Sel` | `nMuon_Loose + nElectron_Loose` |

`FatJet_massGloParT3` is now built from the **pre-JEC raw mass** (`massCorrX2p` predicts
the particle-level mass from the raw jet), and it *replaces* rather than stacks on the
JEC/JER mass calibration. It is therefore JES/JER-invariant and is read straight from the
ntuple for every variation. The old recomputation
`massCorrX2p * FatJet_mass * (1 - FatJet_rawFactor)` no longer reproduces it even for the
nominal — up to ~6% off on an ~80 GeV jet — because the preselection re-applies the JEC
stack while `FatJet_rawFactor` still refers to the original NanoAOD mass.

## Channels

| channel     | reconstruction |
|-------------|----------------|
| `0lep_3FJ`  | boosted H (max Hbb) + two boosted V (next-best Wqq, dR>0.8 from H) |
| `1lep_1FJ`  | single fat jet classified H/V + resolved di-jets (top-5 by dR) + m_lb |
| `1lep_2FJ`  | boosted H (max Hbb) + boosted V (max Wqq, dR>0.8 from H) + m_lb |

## Usage

Run inside the project pixi env (provides ROOT, awkward, pyarrow):

```bash
CONDA_OVERRIDE_CUDA=12.0 pixi run --manifest-path env/pixi.toml \
    python postprocess/postprocess.py \
        --channel 1lep_2FJ \
        --input /path/to/presel_1lep_2FJ/*.root \
        --output out/1lep_2FJ.parquet \
        --threads 8
```

- `--input` accepts multiple files / globs.
- `--nominal-only` skips the JES/JER variations.
- Variations are auto-detected from the file (any `FatJet_pt_<sfx>` with matching
  per-variation masks); on data / `--no_systs` preselection output there are none and
  only the nominal tree is produced. The current production carries 24 variations
  (11 regrouped JES sources x Up/Dn, plus jerUp/jerDn) on signal; the background groups
  were produced without them.

`run.sh` post-processes a whole production. The preselection now writes merged job groups
named only by channel, run and submit timestamp:

    <PRESEL_DIR>/merged_<channel>_<r2|r3>_<timestamp>_<channel>/<sample>/output_*.root

so they no longer spell out `sig` / `data` / `bkg` the way the old `*_r3_1lep_2fj_sig_*`
groups did. Each merged group is homogeneous, so `run.sh` resolves the right directory by
looking at the sample names inside it (`VBS*_c2v*` -> sig, `*_NANOAOD` -> data, the rest
-> bkg) instead of hardcoding the timestamps, which change every production. Point it at a
different production with `PRESEL_DIR=... ./run.sh`. Any arguments to `run.sh` are passed on
to every job, so `./run.sh --skip-existing` resumes a production: only preselection files
without a parquet yet are processed. Logs are appended to, not overwritten.

### Batching and crash recovery

In mirror mode (`--output-dir`), the inputs run in batches of `--files-per-process`
(default 50), each batch in a fresh child process. A single long-lived process eventually
segfaults inside awkward's `from_rdataframe`: after a few hundred files a cppyy
`std::string` it caches as a template argument is freed, and a later dict lookup reads it.
This killed the 2026-09-17 r2 1lep_1FJ signal run at file 480 of 1420; the file itself
processes fine alone. A batch that crashes is rerun with `--skip-existing`, so only
unfinished files are redone. After 3 attempts, the files still missing are tried one per
process, so one bad file can't take its batch down. Files that never produce output are
listed at the end, and the exit code is 1.

Parquets are written to `<name>.part` and renamed on completion, so `--skip-existing` never
trusts a half-written file. Cutflow rows are streamed back from each child per finished
file, so a crash loses none. A table from a `--skip-existing` run covers only the newly
processed files, and its header says `INCOMPLETE`; rerun without it for full yields.
`--files-per-process 0` restores the old single-process behaviour.

## Cutflow

`--cutflow` writes a combined table: the preselection stage, parsed from that job group's
condor `.stdout` logs, then the post-processor's own cuts. The logs are looked up under
`$VBSVVH_PRESEL_DIR/condor/jobs/<jobgroup>/` (default: the sibling `cmstas-run3-vbsvvh`
checkout), whose per-group directory names match the ntuple `<jobgroup>` directory names.

The preselection only prints a cutflow table when its jobs were submitted with its own
`--cutflow` flag. The current production was not, so the combined table degrades to the
post-processor stage alone and prints a warning saying so.

`--cutflow-split` takes a regex to write one table per group; use
`'VBS[A-Z]+H(?:_[OS]S)?_c2v[0-9p]+_c3_[0-9p]+'` for per-signal-point cutflows (the signal
samples are now named `VBSWWH_OS_c2v1p0_c3_10p0_UL18`, not `*C2V_*_C3_*`).

Output: `out/<channel>.parquet`, one row per selected event per variation, with the
candidate columns (`boosted_*_candidate_*`, `resolved_*`, `mlb_min`, `vbs_*`), the event
identifiers (`run`, `luminosityBlock`, `event`), and a `variation` field
(`nominal`, `jesAbsoluteUp`, …). The output candidate column names are unchanged by the
input renames — `boosted_h_candidate_part_mass` still carries the regressed mass — so
downstream (abcd, plotter, datacard) reads the same schema as before.

### Weight systematics

Each `weightsyst_<name>` branch is a length-3 `[central, up, down]` vector whose central
element is already folded into `weight`, so the post-processor writes the two RATIO columns
`weightsyst_<name>_syst_up` / `_syst_dn` = varied/central and a varied weight is
`weight * ratio`.

The set is **discovered per input file** rather than hardcoded, because it is not fixed:

- b-tag sources are named per year and per payload — `btag_HF_statistic_2018` vs
  `btag_HF_statistic_2024Prompt`, and Run 2 exposes `as`/`pdf`/`ttbar` where Run 3 exposes
  `pdfas`/`hdamp`/`topmass`/`type3`/`bfragmentation`;
- channels with no configured b-tag working point (e.g. `0lep_3FJ`) get no b-tag or lepton
  SF branches at all — 6 systematics, against 25 for `1lep_1FJ` Run 3 background;
- data carries only `weight`.

Both schemes the preselection offers for the theory nuisances are carried, so the datacard
stage can pick one without reprocessing: `weightsyst_<name>` on its own, and
`weightsyst_<name>_withbSF` correlated with its b-tag response
(`correlateWeightWithBTagSource`) for the full-breakdown b-tag treatment.

The nominal `weight` already includes the central HF and LF b-tag factors; to undo just
that, divide by `weightsyst_btag_HF_correlated[0] * weightsyst_btag_LF_correlated[0]`
(those centrals are not in the parquet — the ratios are — so do it in the preselection
frame if needed).

Note the lepton SFs are no longer separable: the preselection applies one combined SF per
flavor (reco * tight ID * trigger) as `weightsyst_eleSF` / `weightsyst_muoSF`, replacing
`weightsyst_{muon,electron}{id,reco,trigger}`. `abcd/systematics.py` maps these to
`CMS_eff_e` / `CMS_eff_m`.

## Files

- `postprocess.py` — driver: variation detection, per-variation RDF run, `ak.from_rdataframe` → parquet.
- `objects.py` — `build_collections(df, sfx)`: per-variation good-object collections, tagger scores, VBS four-vectors.
- `channels.py` — per-channel candidate reconstruction (`RECONSTRUCTORS`).
- `helpers.h` — ROOT-only C++ helpers (dR / pairing / invariant mass), declared into cling; no ONNX/TMVA/correctionlib.

## Notes / TODO

- **QCD score resampling** stays in the preselection (as agreed): this stage uses whatever
  GloParT scores the preselection wrote. The preselection resamples only the *nominal*
  good-fat-jet `fatjet_Hbb` / `fatjet_Wqq`, so `build_collections` overlays those onto the
  all-fat-jet arrays (`scatter_good_scores`) before slicing with each variation's good-jet
  mask — scores are JEC-invariant, so a jet keeps its resampled value across variations.
- **b-tag nuisances use the full source breakdown** (`SYST_WEIGHTS` in
  `abcd/systematics.py`): the HF shape sources correlated across years, the per-year HF
  statistical component, and LF correlated + per-year. The two summary branches are
  deliberately excluded because they would double-count — `btag_HF_correlated` is the
  quadrature total of the HF sources (verified: rms 0.00704 vs 0.00694 for their
  quadrature sum, correlation 0.999) and `btag_HF_uncorrelated_<year>` duplicates
  `btag_HF_statistic_<year>` (correlation 0.998). The theory/pileup nuisances
  correspondingly use the `_withbSF` variants where they exist, falling back to the plain
  branches for channels with no b-tag working point (`0lep_3FJ`); `preferred_branches`
  resolves that per frame so each nuisance is written exactly once.
- Validated on the current production (`preselection/new/`): all three channels, Run 2 and
  Run 3, signal / data / background; 25 variations on signal; per-variation candidate
  assignment agrees exactly with the ntuple's own `fatjet_massGloParT3` / `fatjet_Hbb` /
  `fatjet_Wqq` for the nominal event set.
- `(run, luminosityBlock, event)` is **not unique** in the signal ntuples (675 unique in
  736 rows in one spot-checked file), so do not use it as a join key on signal.
