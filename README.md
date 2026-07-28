# Candidate post-processor

Per-variation boson-candidate reconstruction for the reconstructed VBS VVH channels,
run as a **separate stage on top of the preselection output** so the preselection code
(and its systematics bookkeeping) stays untouched.

The preselection already:
- selects good objects and stores, for nominal **and every JES/JER variation**, the
  good-jet masks (`FatJet_isGood[_<sfx>]`, `Jet_isGood[_<sfx>]`,
  `Jet_isGoodNoFJClean[_<sfx>]`), varied kinematics (`*_pt/mass_<sfx>`), and
- runs the **VBS BDT** for every channel, storing the tagged pair as all-jet-frame
  indices + score (`vbs_jet1_Jetidx[_<sfx>]`, `vbs_jet2_Jetidx[_<sfx>]`, `vbs_score[_<sfx>]`).

This stage reads that ntuple and, for each variation, rebuilds the good-object
collections from that variation's masks/kinematics, recomputes the GloParT discriminants
(`HvsQCD`, `VvsQCD`, regressed mass) from the stored raw `FatJet_globalParT3_*`, and
performs the **boson-candidate + resolved + m_lb** assignment ported from the old
1lep-cutbased selections. The VBS pair is taken from the preselection's stored indices —
**the BDT is not re-run here.** Candidates for all variations are written to one
long-format parquet per channel with a `variation` column.

## Channels

| channel     | reconstruction |
|-------------|----------------|
| `0lep_3FJ`  | boosted H (max HvsQCD) + two boosted V (next-best VvsQCD, dR>0.8 from H) |
| `1lep_1FJ`  | single fat jet classified H/V + resolved di-jets (top-5 by dR) + m_lb |
| `1lep_2FJ`  | boosted H (max HvsQCD) + boosted V (max VvsQCD, dR>0.8 from H) + m_lb |

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
  only the nominal tree is produced.

Output: `out/<channel>.parquet`, one row per selected event per variation, with the
candidate columns (`boosted_*_candidate_*`, `resolved_*`, `mlb_min`, `vbs_*`), the event
identifiers (`run`, `luminosityBlock`, `event`), and a `variation` field
(`nominal`, `jesAbsoluteUp`, …).

## Files

- `postprocess.py` — driver: variation detection, per-variation RDF run, `ak.from_rdataframe` → parquet.
- `objects.py` — `build_collections(df, sfx)`: per-variation good-object collections, tagger scores, VBS four-vectors.
- `channels.py` — per-channel candidate reconstruction (`RECONSTRUCTORS`).
- `helpers.h` — ROOT-only C++ helpers (dR / pairing / invariant mass), declared into cling; no ONNX/TMVA/correctionlib.

## Notes / TODO

- **QCD score resampling** stays in the preselection (as agreed): this stage uses whatever
  GloParT scores the preselection wrote. If resampling is added there, the corrected
  scores flow through automatically.
- **Event weights** are not yet carried through beyond a `weight` passthrough (copied if
  present). Weight systematics remain per-event branches in the preselection output; decide
  how they should join the candidate parquet.
- Validated end-to-end on synthetic ntuples (all channels, nominal + a synthetic
  variation). Still needs a run on a real preselection output to confirm branch names and
  yields.
