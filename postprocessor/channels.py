from __future__ import annotations

_CAND_KIN = ["eta", "phi", "mass", "msoftdrop", "part_mass", "pt", "tau21"]

_LEPTON_KIN = ["pt", "eta", "phi", "mass"]
_LEPTON_COLS = [f"{fl}_{k}" for fl in ("electron", "muon", "lepton") for k in _LEPTON_KIN]

_MET_COLS = ["met_pt", "met_phi"]


def add_lepton_scalars(df):
    for flavor, count in (("electron", "nelectron"), ("muon", "nmuon")):
        for k in _LEPTON_KIN:
            col = f"{flavor}_{k}"
            df = df.Redefine(col, f"{count} > 0 ? (float){col}[0] : -999.0f")
    return df


def reconstruct_0lep_3FJ(df, cutflow=None):
    df = (
        df.Define("_best_h_idx", "fatjet_Hbb.size() != 0 ? ArgMax(fatjet_Hbb) : 999.0")
          .Define("boosted_h_candidate_score", "_best_h_idx != 999.0 ? fatjet_Hbb[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_found", "boosted_h_candidate_score > 0")
          .Define("boosted_h_candidate_eta", "boosted_h_candidate_found ? fatjet_eta[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_phi", "boosted_h_candidate_found ? fatjet_phi[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_mass", "boosted_h_candidate_found ? fatjet_mass[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_msoftdrop", "boosted_h_candidate_found ? fatjet_msoftdrop[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_part_mass", "boosted_h_candidate_found ? fatjet_massGloParT3[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_pt", "boosted_h_candidate_found ? fatjet_pt[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_tau21", "boosted_h_candidate_found ? fatjet_tau2[_best_h_idx] / fatjet_tau1[_best_h_idx] : -999.0f")
    )
    df = df.Filter("boosted_h_candidate_found", "boosted H candidate")
    if cutflow is not None:
        cutflow.add(df, "boosted H candidate")

    # boosted V candidates = top-2 Wqq fat jets at dR > 0.8 from the H candidate
    df = (
        df.Define("_fatjet_h_dR", "VdR(fatjet_eta, fatjet_phi, boosted_h_candidate_eta, boosted_h_candidate_phi)")
          .Define("_boosted_v_candidate_jets", "_fatjet_h_dR >= 0.8")
          .Define("_find_w_idx", "fatjet_Wqq[_boosted_v_candidate_jets].size() != 0")
          .Define("_find_w_idxs", "fatjet_Wqq[_boosted_v_candidate_jets].size() > 1")
          .Define("_best_w_idxs", "Argsort(-fatjet_Wqq[_boosted_v_candidate_jets])")
    )
    for v, needflag in (("v1", "_find_w_idx"), ("v2", "_find_w_idxs")):
        k = "0" if v == "v1" else "1"
        df = (
            df.Define(f"boosted_{v}_candidate_score", f"{needflag} ? fatjet_Wqq[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
              .Define(f"boosted_{v}_candidate_found", f"boosted_{v}_candidate_score > 0")
              .Define(f"boosted_{v}_candidate_eta", f"boosted_{v}_candidate_found ? fatjet_eta[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
              .Define(f"boosted_{v}_candidate_phi", f"boosted_{v}_candidate_found ? fatjet_phi[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
              .Define(f"boosted_{v}_candidate_mass", f"boosted_{v}_candidate_found ? fatjet_mass[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
              .Define(f"boosted_{v}_candidate_msoftdrop", f"boosted_{v}_candidate_found ? fatjet_msoftdrop[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
              .Define(f"boosted_{v}_candidate_part_mass", f"boosted_{v}_candidate_found ? fatjet_massGloParT3[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
              .Define(f"boosted_{v}_candidate_pt", f"boosted_{v}_candidate_found ? fatjet_pt[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
              .Define(f"boosted_{v}_candidate_tau21", f"boosted_{v}_candidate_found ? fatjet_tau2[_boosted_v_candidate_jets][_best_w_idxs[{k}]] / fatjet_tau1[_boosted_v_candidate_jets][_best_w_idxs[{k}]] : -999.0f")
        )
    df = df.Filter("boosted_v1_candidate_found && boosted_v2_candidate_found", "boosted V candidates")
    if cutflow is not None:
        cutflow.add(df, "boosted V candidates")

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

    The channel event set (nfatjet==1 && njet>=4) is applied upstream via the
    preselection's passes_1lep_1FJ_<sfx> flag, which guarantees exactly one good fat
    jet for this variation, so fatjet_*[0] below is safe."""
    # The 1lep-cutbased lepton selection. The preselection's 1lep_1FJ filter drops its
    # nElectron_Veto terms (1lep_2FJ keeps them), which lets a veto electron ride along with
    # the muon and even become lepton_pt[0].
    df = df.Filter(
        "(nMuon_Loose == 1 && nMuon_Tight == 1 && nElectron_Veto == 0 && nElectron_Loose == 0 && nElectron_Tight == 0) || "
        "(nMuon_Loose == 0 && nMuon_Tight == 0 && nElectron_Veto == 1 && nElectron_Loose == 1 && nElectron_Tight == 1)",
        "1-lepton selection (electron veto)")
    if cutflow is not None:
        cutflow.add(df, "1-lepton selection (electron veto)")

    df = (
        df.Define("fatjet_is_h", "fatjet_Hbb[0] > fatjet_Wqq[0]")
          .Define("fatjet_is_v", "!fatjet_is_h")
          .Define("boosted_h_candidate_eta", "fatjet_is_h ? fatjet_eta[0] : -999.0f")
          .Define("boosted_h_candidate_phi", "fatjet_is_h ? fatjet_phi[0] : -999.0f")
          .Define("boosted_h_candidate_mass", "fatjet_is_h ? fatjet_mass[0] : -999.0f")
          .Define("boosted_h_candidate_msoftdrop", "fatjet_is_h ? fatjet_msoftdrop[0] : -999.0f")
          .Define("boosted_h_candidate_part_mass", "fatjet_is_h ? fatjet_massGloParT3[0] : -999.0f")
          .Define("boosted_h_candidate_pt", "fatjet_is_h ? fatjet_pt[0] : -999.0f")
          .Define("boosted_h_candidate_tau21", "fatjet_is_h ? fatjet_tau2[0] / fatjet_tau1[0] : -999.0f")
          .Define("boosted_h_candidate_score", "fatjet_is_h ? fatjet_Hbb[0] : -999.0f")
          .Define("boosted_h_candidate_v_score", "fatjet_is_h ? fatjet_Wqq[0] : -999.0f")
          .Define("boosted_v_candidate_eta", "fatjet_is_v ? fatjet_eta[0] : -999.0f")
          .Define("boosted_v_candidate_phi", "fatjet_is_v ? fatjet_phi[0] : -999.0f")
          .Define("boosted_v_candidate_mass", "fatjet_is_v ? fatjet_mass[0] : -999.0f")
          .Define("boosted_v_candidate_msoftdrop", "fatjet_is_v ? fatjet_msoftdrop[0] : -999.0f")
          .Define("boosted_v_candidate_part_mass", "fatjet_is_v ? fatjet_massGloParT3[0] : -999.0f")
          .Define("boosted_v_candidate_pt", "fatjet_is_v ? fatjet_pt[0] : -999.0f")
          .Define("boosted_v_candidate_tau21", "fatjet_is_v ? fatjet_tau2[0] / fatjet_tau1[0] : -999.0f")
          .Define("boosted_v_candidate_score", "fatjet_is_v ? fatjet_Wqq[0] : -999.0f")
          .Define("boosted_v_candidate_h_score", "fatjet_is_v ? fatjet_Hbb[0] : -999.0f")
    )

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
    df = (
        df.Define("_best_h_idx", "fatjet_Hbb.size() != 0 ? ArgMax(fatjet_Hbb) : 999.0")
          .Define("boosted_h_candidate_score", "_best_h_idx != 999.0 ? fatjet_Hbb[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_v_score", "_best_h_idx != 999.0 ? fatjet_Wqq[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_found", "boosted_h_candidate_score > 0")
          .Define("boosted_h_candidate_eta", "boosted_h_candidate_found ? fatjet_eta[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_phi", "boosted_h_candidate_found ? fatjet_phi[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_mass", "boosted_h_candidate_found ? fatjet_mass[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_msoftdrop", "boosted_h_candidate_found ? fatjet_msoftdrop[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_part_mass", "boosted_h_candidate_found ? fatjet_massGloParT3[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_pt", "boosted_h_candidate_found ? fatjet_pt[_best_h_idx] : -999.0f")
          .Define("boosted_h_candidate_tau21", "boosted_h_candidate_found ? fatjet_tau2[_best_h_idx] / fatjet_tau1[_best_h_idx] : -999.0f")
    )
    df = df.Filter("boosted_h_candidate_found", "boosted H candidate")
    if cutflow is not None:
        cutflow.add(df, "boosted H candidate")

    df = (
        df.Define("_fatjet_h_dR", "VdR(fatjet_eta, fatjet_phi, boosted_h_candidate_eta, boosted_h_candidate_phi)")
          .Define("_boosted_v_candidate_jets", "_fatjet_h_dR >= 0.8")
          .Define("_best_w_idx", "fatjet_Wqq[_boosted_v_candidate_jets].size() != 0 ? ArgMax(fatjet_Wqq[_boosted_v_candidate_jets]) : -1")
          .Define("boosted_v_candidate_score", "_best_w_idx != -1 ? fatjet_Wqq[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_h_score", "_best_w_idx != -1 ? fatjet_Hbb[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_found", "boosted_v_candidate_score > 0")
          .Define("boosted_v_candidate_eta", "boosted_v_candidate_found ? fatjet_eta[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_phi", "boosted_v_candidate_found ? fatjet_phi[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_mass", "boosted_v_candidate_found ? fatjet_mass[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_msoftdrop", "boosted_v_candidate_found ? fatjet_msoftdrop[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_part_mass", "boosted_v_candidate_found ? fatjet_massGloParT3[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_pt", "boosted_v_candidate_found ? fatjet_pt[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
          .Define("boosted_v_candidate_tau21", "boosted_v_candidate_found ? fatjet_tau2[_boosted_v_candidate_jets][_best_w_idx] / fatjet_tau1[_boosted_v_candidate_jets][_best_w_idx] : -999.0f")
    )
    df = df.Filter("boosted_v_candidate_found", "boosted V candidate")
    if cutflow is not None:
        cutflow.add(df, "boosted V candidate")

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

CHANNEL_GATE = {
    "0lep_3FJ": "(nLep_Sel == 0) && (nfatjet >= 3) && (njet >= 2)",
    "1lep_1FJ": "(nfatjet == 1) && (njet >= 4)",
    "1lep_2FJ": "(nfatjet >= 2) && (njet >= 2)",
}
