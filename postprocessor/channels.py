"""Boson-candidate reconstruction for the post-processor, ported from the (old)
1lep-cutbased selections. Each function assumes ``build_collections`` has already defined
the per-variation ``fatjet_*`` / ``jet_*`` / ``jetNoFJClean_*`` collections, tagger scores
(``fatjet_HvsQCD`` / ``fatjet_VvsQCD``), counts (``nFatJets`` / ``njet`` / ``njetNoFJClean``)
and the VBS four-vectors (``vbs_*``, rebuilt from the preselection's stored indices).

Each ``reconstruct_*`` applies the channel's jet-multiplicity gate (matching the
preselection channel definition) and returns ``(df, candidate_columns)`` where
``candidate_columns`` are the flat per-event columns to write out.

The dR / pairing / invariant-mass calls (``VdR``, ``getJetPairs``, ``VVInvariantMass``,
``VVInvariantPt``, ``VVDeltaR``, ``VInvariantMass``) resolve to the C++ helpers declared
from ``helpers.h``. The GloParT score expressions match preselection AK8JetsSelection.
"""
from __future__ import annotations

# Kinematics stored for each boosted candidate (suffix appended to the candidate name).
_CAND_KIN = ["eta", "phi", "mass", "msoftdrop", "part_mass", "pt", "tau21"]

# Selected-lepton kinematics stored per flavor (electron / muon) for the 1-lepton channels.
_LEPTON_KIN = ["pt", "eta", "phi", "mass"]
_LEPTON_COLS = [f"{fl}_{k}" for fl in ("electron", "muon", "lepton") for k in _LEPTON_KIN]

_MET_COLS = ["met_pt", "met_phi"]


def add_lepton_scalars(df):
    """Store the single selected lepton's kinematics per flavor as scalars. A 1-lepton
    event has exactly one electron OR one muon (the preselection requires it), so the
    absent flavor is filled with -999. Scalar-izes the jagged NanoAOD electron_*/muon_*
    inputs in place — they carry the full lepton collection but are not otherwise used in
    the reconstruction, which takes the lepton from the combined lepton_* collection."""
    for flavor, count in (("electron", "nelectron"), ("muon", "nmuon")):
        for k in _LEPTON_KIN:
            col = f"{flavor}_{k}"
            df = df.Redefine(col, f"{count} > 0 ? (float){col}[0] : -999.0f")
    return df


def reconstruct_0lep_3FJ(df, cutflow=None):
    """0-lepton, 3 fat jets: boosted H (max HvsQCD) + two boosted V (next-best VvsQCD,
    dR>0.8 from H). VBS pair is the preselection-tagged pair (FJ-cleaned AK4).

    The channel event set (leptons==0 && nFatJets>=3) is applied upstream via the
    preselection's passes_0lep_3FJ_<sfx> flag. VBS jets are pre-tagged in the
    preselection, so no njetNoFJClean cut is needed here."""
    # boosted H candidate = highest HvsQCD fat jet
    df = (
        df.Define("_best_h_idx", "fatjet_HvsQCD.size() != 0 ? ArgMax(fatjet_HvsQCD) : 999.0")
          .Define("boosted_h_candidate_score", "_best_h_idx != 999.0 ? fatjet_HvsQCD[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_found", "boosted_h_candidate_score > 0")
          .Define("boosted_h_candidate_eta", "boosted_h_candidate_found ? fatjet_eta[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_phi", "boosted_h_candidate_found ? fatjet_phi[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_mass", "boosted_h_candidate_found ? fatjet_mass[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_msoftdrop", "boosted_h_candidate_found ? fatjet_msoftdrop[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_part_mass", "boosted_h_candidate_found ? fatjet_globalParT3_mass[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_pt", "boosted_h_candidate_found ? fatjet_pt[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_tau21", "boosted_h_candidate_found ? fatjet_tau2[_best_h_idx] / fatjet_tau1[_best_h_idx] : -999.0f")
    )
    df = df.Filter("boosted_h_candidate_found", "boosted H candidate")
    if cutflow is not None:
        cutflow.add(df, "boosted H candidate")

    # boosted V candidates = top-2 VvsQCD fat jets at dR > 0.8 from the H candidate
    df = (
        df.Define("_fatjet_h_dR", "VdR(fatjet_eta, fatjet_phi, boosted_h_candidate_eta, boosted_h_candidate_phi)")
          .Define("_boosted_v_candidate_jets", "_fatjet_h_dR >= 0.8")
          .Define("_find_w_idx", "fatjet_VvsQCD[_boosted_v_candidate_jets].size() != 0")
          .Define("_find_w_idxs", "fatjet_VvsQCD[_boosted_v_candidate_jets].size() > 1")
          .Define("_best_w_idxs", "Argsort(-fatjet_VvsQCD[_boosted_v_candidate_jets])")
    )
    for v, needflag in (("v1", "_find_w_idx"), ("v2", "_find_w_idxs")):
        k = "0" if v == "v1" else "1"
        df = (
            df.Define(f"boosted_{v}_candidate_score", f"{needflag} ? fatjet_VvsQCD[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
              .Define(f"boosted_{v}_candidate_found", f"boosted_{v}_candidate_score > 0")
              .Define(f"boosted_{v}_candidate_eta", f"boosted_{v}_candidate_found ? fatjet_eta[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
              .Define(f"boosted_{v}_candidate_phi", f"boosted_{v}_candidate_found ? fatjet_phi[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
              .Define(f"boosted_{v}_candidate_mass", f"boosted_{v}_candidate_found ? fatjet_mass[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
              .Define(f"boosted_{v}_candidate_msoftdrop", f"boosted_{v}_candidate_found ? fatjet_msoftdrop[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
              .Define(f"boosted_{v}_candidate_part_mass", f"boosted_{v}_candidate_found ? fatjet_globalParT3_mass[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
              .Define(f"boosted_{v}_candidate_pt", f"boosted_{v}_candidate_found ? fatjet_pt[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
              .Define(f"boosted_{v}_candidate_tau21", f"boosted_{v}_candidate_found ? fatjet_tau2[_boosted_v_candidate_jets][_best_w_idxs[{k}]] / fatjet_tau1[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
        )
    df = df.Filter("boosted_v1_candidate_found && boosted_v2_candidate_found", "boosted V candidates")
    if cutflow is not None:
        cutflow.add(df, "boosted V candidates")

    # Require >=2 FJ-cleaned AK4 jets — the collection the preselection tagged the VBS pair on.
    # The VBS tagger sentinels idx = -1 (rather than filtering) when the collection has < 2 jets,
    # so without this ~40% of 0lep_3FJ events carry all-(-999) VBS features and collapse the
    # downstream VBS BDT.
    df = df.Filter("njet >= 2", "njet >= 2 (VBS pair)")
    if cutflow is not None:
        cutflow.add(df, "njet >= 2 (VBS pair)")

    cols = (
        ["boosted_h_candidate_score", "boosted_h_candidate_found"]
        + [f"boosted_h_candidate_{k}" for k in _CAND_KIN]
        + [f"boosted_{v}_candidate_score" for v in ("v1", "v2")]
        + [f"boosted_{v}_candidate_found" for v in ("v1", "v2")]
        + [f"boosted_{v}_candidate_{k}" for v in ("v1", "v2") for k in _CAND_KIN]
        + _VBS_COLS
        + _MET_COLS
    )
    return df, cols


def reconstruct_1lep_1FJ(df, cutflow=None):
    """1-lepton, 1 fat jet, >=4 FJ-cleaned AK4 jets: classify the fat jet as H or V,
    then resolved di-jet system (AK4 cleaned vs boson candidates + VBS) and m_lb.

    The channel event set (nFatJets==1 && njet>=4) is applied upstream via the
    preselection's passes_1lep_1FJ_<sfx> flag, which guarantees exactly one good fat
    jet for this variation, so fatjet_*[0] below is safe."""
    # single fat jet classified as H or V (nFatJets==1 guaranteed by the pass flag)
    df = (
        df.Define("fatjet_is_h", "fatjet_HvsQCD[0] > fatjet_VvsQCD[0]")
          .Define("fatjet_is_v", "!fatjet_is_h")
          .Define("boosted_h_candidate_eta", "fatjet_is_h ? fatjet_eta[0] : -999.0f")
          .Define("boosted_h_candidate_phi", "fatjet_is_h ? fatjet_phi[0] : -999.0f")
          .Define("boosted_h_candidate_mass", "fatjet_is_h ? fatjet_mass[0] : -999.0f")
          .Define("boosted_h_candidate_msoftdrop", "fatjet_is_h ? fatjet_msoftdrop[0] : -999.0f")
          .Define("boosted_h_candidate_part_mass", "fatjet_is_h ? fatjet_globalParT3_mass[0] : -999.0f")
          .Define("boosted_h_candidate_pt", "fatjet_is_h ? fatjet_pt[0] : -999.0f")
          .Define("boosted_h_candidate_tau21", "fatjet_is_h ? fatjet_tau2[0] / fatjet_tau1[0] : -999.0f")
          .Define("boosted_h_candidate_score", "fatjet_is_h ? fatjet_HvsQCD[0] : -999.0f")
          .Define("boosted_h_candidate_v_score", "fatjet_is_h ? fatjet_VvsQCD[0] : -999.0f")
          .Define("boosted_v_candidate_eta", "fatjet_is_v ? fatjet_eta[0] : -999.0f")
          .Define("boosted_v_candidate_phi", "fatjet_is_v ? fatjet_phi[0] : -999.0f")
          .Define("boosted_v_candidate_mass", "fatjet_is_v ? fatjet_mass[0] : -999.0f")
          .Define("boosted_v_candidate_msoftdrop", "fatjet_is_v ? fatjet_msoftdrop[0] : -999.0f")
          .Define("boosted_v_candidate_part_mass", "fatjet_is_v ? fatjet_globalParT3_mass[0] : -999.0f")
          .Define("boosted_v_candidate_pt", "fatjet_is_v ? fatjet_pt[0] : -999.0f")
          .Define("boosted_v_candidate_tau21", "fatjet_is_v ? fatjet_tau2[0] / fatjet_tau1[0] : -999.0f")
          .Define("boosted_v_candidate_score", "fatjet_is_v ? fatjet_VvsQCD[0] : -999.0f")
          .Define("boosted_v_candidate_h_score", "fatjet_is_v ? fatjet_HvsQCD[0] : -999.0f")
    )

    # resolved di-jet system: FJ-cleaned AK4 jets cleaned vs boson candidates and VBS jets
    df = (
        df.Define("jet_v_dR", "VdR(jet_eta, jet_phi, boosted_v_candidate_eta, boosted_v_candidate_phi)")
          .Define("jet_h_dR", "VdR(jet_eta, jet_phi, boosted_h_candidate_eta, boosted_h_candidate_phi)")
          .Define("jet_vbs1_dR", "VdR(jet_eta, jet_phi, vbs_jet1_eta, vbs_jet1_phi)")
          .Define("jet_vbs2_dR", "VdR(jet_eta, jet_phi, vbs_jet2_eta, vbs_jet2_phi)")
          .Define("_resolved_candidate_jets",
                  "abs(jet_eta) <= 2.5 && jet_v_dR >= 0.8 && jet_h_dR >= 0.8 && "
                  "jet_vbs1_dR >= 0.4 && jet_vbs2_dR >= 0.4")
          .Define("_resolved_candidate_pt", "jet_pt[_resolved_candidate_jets]")
          .Define("_resolved_candidate_eta", "jet_eta[_resolved_candidate_jets]")
          .Define("_resolved_candidate_phi", "jet_phi[_resolved_candidate_jets]")
          .Define("_resolved_candidate_mass", "jet_mass[_resolved_candidate_jets]")
          .Define("_resolved_candidate_pairs", "getJetPairs(_resolved_candidate_pt)")
          .Define("_rc_pairs1_pt", "Take(_resolved_candidate_pt, _resolved_candidate_pairs[0], -999.0f)")
          .Define("_rc_pairs2_pt", "Take(_resolved_candidate_pt, _resolved_candidate_pairs[1], -999.0f)")
          .Define("_rc_pairs1_eta", "Take(_resolved_candidate_eta, _resolved_candidate_pairs[0], -999.0f)")
          .Define("_rc_pairs2_eta", "Take(_resolved_candidate_eta, _resolved_candidate_pairs[1], -999.0f)")
          .Define("_rc_pairs1_phi", "Take(_resolved_candidate_phi, _resolved_candidate_pairs[0], -999.0f)")
          .Define("_rc_pairs2_phi", "Take(_resolved_candidate_phi, _resolved_candidate_pairs[1], -999.0f)")
          .Define("_rc_pairs1_mass", "Take(_resolved_candidate_mass, _resolved_candidate_pairs[0], -999.0f)")
          .Define("_rc_pairs2_mass", "Take(_resolved_candidate_mass, _resolved_candidate_pairs[1], -999.0f)")
          .Define("_rc_ptjj", "VVInvariantPt(_rc_pairs1_pt, _rc_pairs1_eta, _rc_pairs1_phi, _rc_pairs1_mass, _rc_pairs2_pt, _rc_pairs2_eta, _rc_pairs2_phi, _rc_pairs2_mass)")
          .Define("_rc_mjj", "VVInvariantMass(_rc_pairs1_pt, _rc_pairs1_eta, _rc_pairs1_phi, _rc_pairs1_mass, _rc_pairs2_pt, _rc_pairs2_eta, _rc_pairs2_phi, _rc_pairs2_mass)")
          .Define("_rc_dR", "VVDeltaR(_rc_pairs1_eta, _rc_pairs1_phi, _rc_pairs2_eta, _rc_pairs2_phi)")
          .Define("_sorted_resolved_dR", "Argsort(_rc_dR)")
    )
    # top-5 resolved di-jets ordered by dR
    for n in range(1, 6):
        i = n - 1
        df = (
            df.Define(f"resolved_mjj_{n}", f"_rc_mjj.size() > {i} ? _rc_mjj[_sorted_resolved_dR[{i}]] : -999.0f")
              .Define(f"resolved_dR_{n}",  f"_rc_dR.size() > {i} ? _rc_dR[_sorted_resolved_dR[{i}]] : -999.0f")
              .Define(f"resolved_ptjj_{n}", f"_rc_ptjj.size() > {i} ? _rc_ptjj[_sorted_resolved_dR[{i}]] : -999.0f")
        )
    df = df.Filter("Sum(_resolved_candidate_jets) >= 2", "resolved candidate pair")
    if cutflow is not None:
        cutflow.add(df, "resolved candidate pair")

    # m_lb: min invariant mass of (loose-b-tagged resolved jet, lepton)
    df = (
        df.Define("_mlb_candidate_jets", "_resolved_candidate_jets && jet_isLooseBTag")
          .Define("_mlb_jets_pt", "jet_pt[_mlb_candidate_jets]")
          .Define("_mlb_jets_eta", "jet_eta[_mlb_candidate_jets]")
          .Define("_mlb_jets_phi", "jet_phi[_mlb_candidate_jets]")
          .Define("_mlb_jets_mass", "jet_mass[_mlb_candidate_jets]")
          .Define("_mlb", "VInvariantMass(_mlb_jets_pt, _mlb_jets_eta, _mlb_jets_phi, _mlb_jets_mass, "
                          "lepton_pt[0], lepton_eta[0], lepton_phi[0], lepton_mass[0])")
          .Define("mlb_min", "_mlb.size() > 0 ? Min(_mlb) : -999.0f")
    )
    df = add_lepton_scalars(df)

    cols = (
        ["boosted_h_candidate_score", "boosted_h_candidate_v_score"]
        + [f"boosted_h_candidate_{k}" for k in _CAND_KIN if k != "part_mass"] + ["boosted_h_candidate_part_mass"]
        + ["boosted_v_candidate_score", "boosted_v_candidate_h_score"]
        + [f"boosted_v_candidate_{k}" for k in _CAND_KIN if k != "part_mass"] + ["boosted_v_candidate_part_mass"]
        + [f"resolved_mjj_{n}" for n in range(1, 6)]
        + [f"resolved_dR_{n}" for n in range(1, 6)]
        + [f"resolved_ptjj_{n}" for n in range(1, 6)]
        + ["mlb_min"]
        + _LEPTON_COLS
        + _VBS_COLS
        + _MET_COLS
    )
    return df, cols


def reconstruct_1lep_2FJ(df, cutflow=None):
    """1-lepton, >=2 fat jets: boosted H (max HvsQCD) + boosted V (max VvsQCD, dR>0.8 from
    H). VBS pair is the preselection-tagged pair (non-FJ-cleaned AK4). Adds m_lb.

    The channel event set (nFatJets>=2 && njet>=2) is applied upstream via the
    preselection's passes_1lep_2FJ_<sfx> flag. VBS jets are pre-tagged in the
    preselection, so no njetNoFJClean cut is needed here."""
    df = (
        df.Define("_best_h_idx", "fatjet_HvsQCD.size() != 0 ? ArgMax(fatjet_HvsQCD) : 999.0")
          .Define("boosted_h_candidate_score", "_best_h_idx != 999.0 ? fatjet_HvsQCD[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_v_score", "_best_h_idx != 999.0 ? fatjet_VvsQCD[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_found", "boosted_h_candidate_score > 0")
          .Define("boosted_h_candidate_eta", "boosted_h_candidate_found ? fatjet_eta[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_phi", "boosted_h_candidate_found ? fatjet_phi[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_mass", "boosted_h_candidate_found ? fatjet_mass[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_msoftdrop", "boosted_h_candidate_found ? fatjet_msoftdrop[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_part_mass", "boosted_h_candidate_found ? fatjet_globalParT3_mass[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_pt", "boosted_h_candidate_found ? fatjet_pt[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_tau21", "boosted_h_candidate_found ? fatjet_tau2[_best_h_idx] / fatjet_tau1[_best_h_idx] : -999.0f")
    )
    df = df.Filter("boosted_h_candidate_found", "boosted H candidate")
    if cutflow is not None:
        cutflow.add(df, "boosted H candidate")

    df = (
        df.Define("_fatjet_h_dR", "VdR(fatjet_eta, fatjet_phi, boosted_h_candidate_eta, boosted_h_candidate_phi)")
          .Define("_boosted_v_candidate_jets", "_fatjet_h_dR >= 0.8")
          .Define("_best_w_idx", "fatjet_VvsQCD[_boosted_v_candidate_jets].size() != 0 ? ArgMax(fatjet_VvsQCD[_boosted_v_candidate_jets]) : -1")
          .Define("boosted_v_candidate_score", "_best_w_idx != -1 ? fatjet_VvsQCD[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_h_score", "_best_w_idx != -1 ? fatjet_HvsQCD[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_found", "boosted_v_candidate_score > 0")
          .Define("boosted_v_candidate_eta", "boosted_v_candidate_found ? fatjet_eta[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_phi", "boosted_v_candidate_found ? fatjet_phi[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_mass", "boosted_v_candidate_found ? fatjet_mass[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_msoftdrop", "boosted_v_candidate_found ? fatjet_msoftdrop[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_part_mass", "boosted_v_candidate_found ? fatjet_globalParT3_mass[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_pt", "boosted_v_candidate_found ? fatjet_pt[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_tau21", "boosted_v_candidate_found ? fatjet_tau2[_boosted_v_candidate_jets][_best_w_idx] / fatjet_tau1[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
    )
    df = df.Filter("boosted_v_candidate_found", "boosted V candidate")
    if cutflow is not None:
        cutflow.add(df, "boosted V candidate")

    # m_lb: min invariant mass of (loose-b-tagged AK4 cleaned vs VBS + boson candidates, lepton)
    df = (
        df.Define("_jet_vbs1_dR", "VdR(jet_eta, jet_phi, vbs_jet1_eta, vbs_jet1_phi)")
          .Define("_jet_vbs2_dR", "VdR(jet_eta, jet_phi, vbs_jet2_eta, vbs_jet2_phi)")
          .Define("_jet_h_dR", "VdR(jet_eta, jet_phi, boosted_h_candidate_eta, boosted_h_candidate_phi)")
          .Define("_jet_v_dR", "VdR(jet_eta, jet_phi, boosted_v_candidate_eta, boosted_v_candidate_phi)")
          .Define("_mlb_candidate_jets",
                  "abs(jet_eta) <= 2.5 && _jet_vbs1_dR >= 0.4 && _jet_vbs2_dR >= 0.4 && "
                  "_jet_h_dR >= 0.8 && _jet_v_dR >= 0.8 && jet_isLooseBTag")
          .Define("_mlb_jets_pt", "jet_pt[_mlb_candidate_jets]")
          .Define("_mlb_jets_eta", "jet_eta[_mlb_candidate_jets]")
          .Define("_mlb_jets_phi", "jet_phi[_mlb_candidate_jets]")
          .Define("_mlb_jets_mass", "jet_mass[_mlb_candidate_jets]")
          .Define("_mlb", "VInvariantMass(_mlb_jets_pt, _mlb_jets_eta, _mlb_jets_phi, _mlb_jets_mass, "
                          "lepton_pt[0], lepton_eta[0], lepton_phi[0], lepton_mass[0])")
          .Define("mlb_min", "_mlb.size() > 0 ? Min(_mlb) : -999.0f")
    )
    df = add_lepton_scalars(df)

    cols = (
        ["boosted_h_candidate_score", "boosted_h_candidate_v_score", "boosted_h_candidate_found"]
        + [f"boosted_h_candidate_{k}" for k in _CAND_KIN]
        + ["boosted_v_candidate_score", "boosted_v_candidate_h_score", "boosted_v_candidate_found"]
        + [f"boosted_v_candidate_{k}" for k in _CAND_KIN]
        + ["mlb_min"]
        + _LEPTON_COLS
        + _VBS_COLS
        + _MET_COLS
    )
    return df, cols


# VBS pair columns (rebuilt in build_collections from the preselection's stored indices).
_VBS_COLS = [
    "vbs_score", "vbs_mjj", "vbs_detajj",
    "vbs_jet1_pt", "vbs_jet1_eta", "vbs_jet1_phi", "vbs_jet1_mass",
    "vbs_jet2_pt", "vbs_jet2_eta", "vbs_jet2_phi", "vbs_jet2_mass",
]

RECONSTRUCTORS = {
    "0lep_3FJ": reconstruct_0lep_3FJ,
    "1lep_1FJ": reconstruct_1lep_1FJ,
    "1lep_2FJ": reconstruct_1lep_2FJ,
}

# Preselection channel gate, matching definePerVariationPassFlags in selections.cpp.
# The AUTHORITATIVE per-variation event set is the stored passes_<channel>_<sfx> flag;
# the driver selects on that. These expressions (evaluated on the variation's rebuilt
# nFatJets/njet) are only a fallback for input ntuples that predate the flags.
CHANNEL_GATE = {
    "0lep_3FJ": "(nMuon_Loose == 0) && (nElectron_Loose == 0) && (nFatJets >= 3)",
    "1lep_1FJ": "(nFatJets == 1) && (njet >= 4)",
    "1lep_2FJ": "(nFatJets >= 2) && (njet >= 2)",
}
