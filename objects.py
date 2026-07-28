"""Per-variation object-collection building for the candidate post-processor.

The preselection writes, for every JES/JER variation ``sfx``:
  * good-object masks at the all-jet level: ``FatJet_isGood_<sfx>``,
    ``Jet_isGood_<sfx>`` (FJ-cleaned), ``Jet_isGoodNoFJClean_<sfx>``;
  * varied kinematics ``FatJet_pt/mass_<sfx>``, ``Jet_pt/mass_<sfx>`` (eta/phi are
    JEC-invariant, stored once without a suffix);
  * the VBS pair the BDT tagged, as all-jet-frame indices + score:
    ``vbs_jet1_Jetidx_<sfx>``, ``vbs_jet2_Jetidx_<sfx>``, ``vbs_score_<sfx>``.
Nominal columns carry no suffix (``FatJet_isGood``, ``FatJet_pt``, ``vbs_jet1_Jetidx`` …).

``build_collections`` slices those into the lowercase good-object collections
(``fatjet_*``, ``jet_*``, ``jetNoFJClean_*``) and the VBS four-vectors that the channel
reconstruction expects — using the given variation's masks/kinematics so the SAME
reconstruction code produces per-variation candidates. The nominal lowercase collections
already exist in the preselection output, so every column is defined via
define-or-redefine.
"""
from __future__ import annotations


def _c(base: str, sfx: str) -> str:
    """Column name for variation ``sfx`` (``""`` == nominal)."""
    return f"{base}_{sfx}" if sfx else base


# Per-event weight systematics written by the preselection as length-3 branches
# weightsyst_<name> = [nominal, up, down] (weights.cpp applyMCWeights). The nominal
# element is already folded into `weight`, so the varied weight is weight * (elem/nominal).
WEIGHT_SYSTS = ["pileup", "muonid", "muonreco", "muontrigger", "electronid",
                "electronreco", "electrontrigger", "l1prefiring",
                "PSFSR", "PSISR", "muF", "muR"]


def add_weight_columns(df):
    """Define the per-event weight columns for the output and return their names.

    Passes through the nominal `weight` (and `ewkweight`) and, for each length-3
    weightsyst_<name> present, writes the up/down RATIO columns
    weightsyst_<name>_syst_up / _dn = varied/nominal (== 1 where nominal is 0 or
    non-finite). These are flat floats, so parquet consumers read them directly;
    the nominal is already in `weight`, so a varied weight is `weight * ratio`.

    Weight systematics are computed from nominal-object SFs (leptons/pileup), which
    are unchanged by JES/JER, so the same values are carried for every variation.
    """
    cols = set(str(c) for c in df.GetColumnNames())
    out = [w for w in ("weight", "ewkweight") if w in cols]
    for name in WEIGHT_SYSTS:
        br = f"weightsyst_{name}"
        if br not in cols:
            continue
        up, dn = f"{br}_syst_up", f"{br}_syst_dn"
        safe = f"({br}.size() == 3 && {br}[0] != 0.f && std::isfinite({br}[0]))"
        df = df.Define(up, f"{safe} ? (float)({br}[1] / {br}[0]) : 1.0f")
        df = df.Define(dn, f"{safe} ? (float)({br}[2] / {br}[0]) : 1.0f")
        out += [up, dn]
    return df, out


def _def(df, name: str, expr: str):
    """Define ``name``, or Redefine it if the preselection already wrote that column."""
    if name in df.GetColumnNames():
        return df.Redefine(name, expr)
    return df.Define(name, expr)


def build_collections(df, sfx: str):
    """Define/redefine fatjet_*, jet_*, jetNoFJClean_*, counts, tagger scores and the VBS
    four-vectors for variation ``sfx`` on top of a preselection-output RDataFrame.

    All target columns use plain (non-suffixed) names so the channel reconstruction is
    variation-agnostic; call this once per variation on a fresh RDataFrame.
    """
    fj_mask = _c("FatJet_isGood", sfx)
    fj_pt, fj_mass = _c("FatJet_pt", sfx), _c("FatJet_mass", sfx)
    jt_mask = _c("Jet_isGood", sfx)
    jt_pt, jt_mass = _c("Jet_pt", sfx), _c("Jet_mass", sfx)
    jnf_mask = _c("Jet_isGoodNoFJClean", sfx)

    # --- good fat jets (H/V candidate pool) ---
    df = _def(df, "fatjet_pt", f"{fj_pt}[{fj_mask}]")
    df = _def(df, "fatjet_eta", f"FatJet_eta[{fj_mask}]")
    df = _def(df, "fatjet_phi", f"FatJet_phi[{fj_mask}]")
    df = _def(df, "fatjet_mass", f"{fj_mass}[{fj_mask}]")
    df = _def(df, "fatjet_msoftdrop", f"FatJet_msoftdrop[{fj_mask}]")
    df = _def(df, "fatjet_tau1", f"FatJet_tau1[{fj_mask}]")
    df = _def(df, "fatjet_tau2", f"FatJet_tau2[{fj_mask}]")
    df = _def(df, "nFatJets", f"Sum({fj_mask})")
    # GloParT-regressed mass (follows the varied FatJet_mass), computed on all fat jets then masked.
    df = _def(df, "fatjet_globalParT3_mass",
              f"(FatJet_globalParT3_massCorrX2p * {fj_mass} * (1 - FatJet_rawFactor))[{fj_mask}]")
    # H/V discriminants: use the preselection's resampled scores for EVERY variation, never
    # recompute from the raw FatJet_globalParT3_* (which would discard the data-driven QCD score
    # resampling applied in the preselection, applyQCDScoreResampling for 0lep_3FJ). The
    # preselection resamples only the NOMINAL good-fat-jet scores (fatjet_HvsQCD/VvsQCD), so
    # overlay them onto the all-fat-jet FatJet_HvsQCD/VvsQCD and slice that with each variation's
    # good-jet mask. Scores are JEC-invariant, so a jet keeps its resampled value across
    # variations; a jet good only under a variation (pT near threshold, not nominal-good) falls
    # back to the unresampled all-fat-jet score.
    df = _def(df, "_FatJet_HvsQCD_res", "scatter_good_scores(FatJet_HvsQCD, fatjet_HvsQCD, FatJet_isGood)")
    df = _def(df, "_FatJet_VvsQCD_res", "scatter_good_scores(FatJet_VvsQCD, fatjet_VvsQCD, FatJet_isGood)")
    df = _def(df, "fatjet_HvsQCD", f"_FatJet_HvsQCD_res[{fj_mask}]")
    df = _def(df, "fatjet_VvsQCD", f"_FatJet_VvsQCD_res[{fj_mask}]")

    # --- good AK4 jets (FJ-cleaned): resolved di-jets and mlb ---
    df = _def(df, "jet_pt", f"{jt_pt}[{jt_mask}]")
    df = _def(df, "jet_eta", f"Jet_eta[{jt_mask}]")
    df = _def(df, "jet_phi", f"Jet_phi[{jt_mask}]")
    df = _def(df, "jet_mass", f"{jt_mass}[{jt_mask}]")
    df = _def(df, "jet_btagUParTAK4B", f"Jet_btagUParTAK4B[{jt_mask}]")
    df = _def(df, "jet_isLooseBTag", f"Jet_isLooseBTag[{jt_mask}]")
    df = _def(df, "njet", f"Sum({jt_mask})")

    # --- good AK4 jets (NOT FJ-cleaned): VBS pool / 1lep_2FJ + 0lep_3FJ jet counting ---
    df = _def(df, "jetNoFJClean_pt", f"{jt_pt}[{jnf_mask}]")
    df = _def(df, "jetNoFJClean_eta", f"Jet_eta[{jnf_mask}]")
    df = _def(df, "jetNoFJClean_phi", f"Jet_phi[{jnf_mask}]")
    df = _def(df, "jetNoFJClean_mass", f"{jt_mass}[{jnf_mask}]")
    df = _def(df, "njetNoFJClean", f"Sum({jnf_mask})")

    # --- VBS pair: rebuild four-vectors from the preselection's stored all-jet indices ---
    df = _def(df, "vbs_jet1_idx", _c("vbs_jet1_Jetidx", sfx))
    df = _def(df, "vbs_jet2_idx", _c("vbs_jet2_Jetidx", sfx))
    df = _def(df, "vbs_score", _c("vbs_score", sfx))
    df = _def(df, "vbs_jet1_pt",   f"vbs_jet1_idx >= 0 ? {jt_pt}[vbs_jet1_idx]   : -999.f")
    df = _def(df, "vbs_jet1_eta",  f"vbs_jet1_idx >= 0 ? Jet_eta[vbs_jet1_idx]   : -999.f")
    df = _def(df, "vbs_jet1_phi",  f"vbs_jet1_idx >= 0 ? Jet_phi[vbs_jet1_idx]   : -999.f")
    df = _def(df, "vbs_jet1_mass", f"vbs_jet1_idx >= 0 ? {jt_mass}[vbs_jet1_idx] : -999.f")
    df = _def(df, "vbs_jet2_pt",   f"vbs_jet2_idx >= 0 ? {jt_pt}[vbs_jet2_idx]   : -999.f")
    df = _def(df, "vbs_jet2_eta",  f"vbs_jet2_idx >= 0 ? Jet_eta[vbs_jet2_idx]   : -999.f")
    df = _def(df, "vbs_jet2_phi",  f"vbs_jet2_idx >= 0 ? Jet_phi[vbs_jet2_idx]   : -999.f")
    df = _def(df, "vbs_jet2_mass", f"vbs_jet2_idx >= 0 ? {jt_mass}[vbs_jet2_idx] : -999.f")
    df = _def(df, "vbs_mjj", "vbs_jet1_idx >= 0 ? (ROOT::Math::PtEtaPhiMVector(vbs_jet1_pt, vbs_jet1_eta, vbs_jet1_phi, vbs_jet1_mass) + "
                             "ROOT::Math::PtEtaPhiMVector(vbs_jet2_pt, vbs_jet2_eta, vbs_jet2_phi, vbs_jet2_mass)).M() : -999.f")
    df = _def(df, "vbs_detajj", "vbs_jet1_idx >= 0 ? std::abs(vbs_jet1_eta - vbs_jet2_eta) : -999.f")
    return df
