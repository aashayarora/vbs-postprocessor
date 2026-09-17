"""The per-event systematic weight variations carried through to the datacards.

The preselection stores each of these as a length-3 branch
``[nominal, up, down]`` and folds *only the nominal element* into the event
weight (see ``preselection/src/weights.cpp`` ``applyMCWeights``):

    weight = xsecweight * genWeight * prod(weight_<syst>[0])

So a varied event weight is ``weight * weight_<syst>[k] / weight_<syst>[0]``.
We store those two ratios rather than the raw elements: they are what the
datacard multiplies by, they are independent of the nominal weight, and they
sit near 1.0 so they compress well.

``weight_btagging_sf_HF``/``_LF`` and ``weight_l1prefiring`` are commented out
of the preselection (both the Define and the weight product), so they are not
in the ntuples and are absent here. ``weight_ewk`` is a plain scalar with no
up/down and is likewise not a systematic.
"""

# Signal process tag for the theory nuisances (QCDscale/PS), correlated across all
# channels of this analysis (e.g. QCDscale_fac_vbsvvh, ps_fsr_vbsvvh).
PROC_BASE = "vbsvvh"

# Branch name -> (combine nuisance name, scope). These follow the standard CMS
# correlation convention and are CORRELATED across channels — NOT scoped per
# channel/scan (only the stat / control-ABCD / tagger nuisances stay per-channel).
#   scope "era"  -> append the centre-of-mass era tag (13TeV / 13p6TeV)
#   scope "corr" -> fixed name, correlated across channels and eras
#   scope "proc" -> append PROC_BASE (theory nuisances scoped to the signal process)
SYST_WEIGHTS = {
    "weightsyst_muF":             ("QCDscale_fac",          "proc"),
    "weightsyst_muR":             ("QCDscale_ren",          "proc"),
    "weightsyst_PSISR":           ("ps_isr",                "proc"),
    "weightsyst_PSFSR":           ("ps_fsr",                "proc"),
    "weightsyst_pileup":          ("CMS_pileup",            "era"),
    "weightsyst_l1prefiring":     ("CMS_l1_ecal_prefiring", "corr"),
    "weightsyst_muonid":          ("CMS_eff_m_id",          "corr"),
    "weightsyst_muonreco":        ("CMS_eff_m_reco",        "corr"),
    "weightsyst_muontrigger":     ("CMS_eff_m_trigger",     "corr"),
    "weightsyst_electronid":      ("CMS_eff_e_id",          "corr"),
    "weightsyst_electronreco":    ("CMS_eff_e_reco",        "corr"),
    "weightsyst_electrontrigger": ("CMS_eff_e_trigger",     "corr"),
}

# NOTE: there is deliberately no acceptance-only treatment here. Every variation,
# theory ones included, enters the datacard as the raw per-region varied/nominal
# yield ratio, so it carries its full normalization + acceptance effect. This
# follows the Run 2 semileptonic datacard script. An earlier version of this file
# divided out the inclusive ratio for muF/muR/PSISR/PSFSR, which suppressed the
# ~20% muF effect down to ~2% because the denominator was the already-preselected
# sample rather than the generated sum of weights.

UP_SUFFIX = "_syst_up"
DN_SUFFIX = "_syst_dn"


def ratio_columns(branch):
    """The (up, down) ratio column names written for one systematic branch."""
    return branch + UP_SUFFIX, branch + DN_SUFFIX


def all_ratio_columns():
    cols = []
    for branch in SYST_WEIGHTS:
        cols.extend(ratio_columns(branch))
    return cols


def era_suffix(proc_name):
    """Map a process name like '0lep_3fj_r3' to its centre-of-mass energy tag."""
    name = str(proc_name).lower()
    if "r2" in name.split("_"):
        return "13TeV"
    if "r3" in name.split("_"):
        return "13p6TeV"
    return None


def nuisance_name(branch, proc_name, scan_name=None):
    """Combine nuisance name for a weight systematic, e.g. CMS_pileup_13p6TeV,
    CMS_eff_m_id, QCDscale_fac_vbsvvh.

    CORRELATED across channels (the CMS convention), so NOT scoped by channel/scan:
    experimental SFs use fixed CMS names (pileup is per-era), the theory nuisances are
    scoped to the signal process. ``scan_name`` is accepted for call compatibility but
    is unused (correlated nuisances must share a name across scans).
    """
    base, scope = SYST_WEIGHTS[branch]
    if scope == "era":
        era = era_suffix(proc_name)
        return f"{base}_{era}" if era else base
    if scope == "proc":
        return f"{base}_{PROC_BASE}"
    return base


def jec_nuisance_name(source, proc_name, year=None):
    """Combine nuisance name for a JES regrouped source or JER (a `variation` label with
    Up/Dn stripped: 'jesAbsolute', 'jesAbsoluteYear', 'jer'). CORRELATED across channels,
    matching the CMS convention:

      * JES sources without the 'Year' tag correlate across years: CMS_scale_j_Absolute.
      * JES '*Year' regrouped sources are decorrelated per data-taking year: pass ``year``
        to get CMS_scale_j_Absolute_2018 (falls back to the era tag if year is None).
      * JER is per-era: CMS_res_j_13p6TeV.
    """
    era = era_suffix(proc_name)
    if source == "jer":
        return f"CMS_res_j_{era}" if era else "CMS_res_j"
    stem = source[3:] if source.startswith("jes") else source
    if stem.endswith("Year"):
        stem = stem[:-4]
        tag = year if year else era
        return f"CMS_scale_j_{stem}_{tag}" if tag else f"CMS_scale_j_{stem}"
    return f"CMS_scale_j_{stem}"
