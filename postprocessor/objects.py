from __future__ import annotations


def _c(base: str, sfx: str) -> str:
    """Column name for variation ``sfx`` (``""`` == nominal)."""
    return f"{base}_{sfx}" if sfx else base

WEIGHTSYST_PREFIX = "weightsyst_"


def discover_weight_systs(cols):
    return sorted(c for c in cols if c.startswith(WEIGHTSYST_PREFIX))


def add_weight_columns(df):
    cols = set(str(c) for c in df.GetColumnNames())
    out = [w for w in ("weight", "ewkweight") if w in cols]
    for br in discover_weight_systs(cols):
        up, dn = f"{br}_syst_up", f"{br}_syst_dn"
        safe = f"({br}.size() == 3 && {br}[0] != 0. && std::isfinite({br}[0]))"
        df = df.Define(up, f"{safe} ? (float)({br}[1] / {br}[0]) : 1.0f")
        df = df.Define(dn, f"{safe} ? (float)({br}[2] / {br}[0]) : 1.0f")
        out += [up, dn]
    return df, out


def _def(df, name: str, expr: str):
    if name in df.GetColumnNames():
        return df.Redefine(name, expr)
    return df.Define(name, expr)


def build_collections(df, sfx: str):
    fj_mask = _c("FatJet_isGood", sfx)
    fj_pt, fj_mass = _c("FatJet_pt", sfx), _c("FatJet_mass", sfx)
    jt_mask = _c("Jet_isGood", sfx)
    jt_pt, jt_mass = _c("Jet_pt", sfx), _c("Jet_mass", sfx)

    # --- good fat jets (H/V candidate pool) ---
    df = _def(df, "fatjet_pt", f"{fj_pt}[{fj_mask}]")
    df = _def(df, "fatjet_eta", f"FatJet_eta[{fj_mask}]")
    df = _def(df, "fatjet_phi", f"FatJet_phi[{fj_mask}]")
    df = _def(df, "fatjet_mass", f"{fj_mass}[{fj_mask}]")
    df = _def(df, "fatjet_msoftdrop", f"FatJet_msoftdrop[{fj_mask}]")
    df = _def(df, "fatjet_tau1", f"FatJet_tau1[{fj_mask}]")
    df = _def(df, "fatjet_tau2", f"FatJet_tau2[{fj_mask}]")
    df = _def(df, "nfatjet", f"Sum({fj_mask})")
    df = _def(df, "fatjet_massGloParT3", f"FatJet_massGloParT3[{fj_mask}]")
    df = _def(df, "_FatJet_Hbb_res", "scatter_good_scores(FatJet_Hbb, fatjet_Hbb, FatJet_isGood)")
    df = _def(df, "_FatJet_Wqq_res", "scatter_good_scores(FatJet_Wqq, fatjet_Wqq, FatJet_isGood)")
    df = _def(df, "fatjet_Hbb", f"_FatJet_Hbb_res[{fj_mask}]")
    df = _def(df, "fatjet_Wqq", f"_FatJet_Wqq_res[{fj_mask}]")

    df = _def(df, "jet_pt", f"{jt_pt}[{jt_mask}]")
    df = _def(df, "jet_eta", f"Jet_eta[{jt_mask}]")
    df = _def(df, "jet_phi", f"Jet_phi[{jt_mask}]")
    df = _def(df, "jet_mass", f"{jt_mass}[{jt_mask}]")
    df = _def(df, "jet_btagUParTAK4B", f"Jet_btagUParTAK4B[{jt_mask}]")
    df = _def(df, "jet_isLooseBTag", f"Jet_isLooseBTag[{jt_mask}]")
    df = _def(df, "njet", f"Sum({jt_mask})")

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
