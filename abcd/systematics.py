"""The per-event systematic weight variations carried through to the datacards.

The preselection stores each of these as a length-3 branch
``[nominal, up, down]`` and folds *only the nominal element* into the event
weight (see ``preselection/src/weights.cpp`` ``applyMCWeights``):

    weight = xsecweight * genWeight * prod(weight_<syst>[0])

So a varied event weight is ``weight * weight_<syst>[k] / weight_<syst>[0]``.
We store those two ratios rather than the raw elements: they are what the
datacard multiplies by, they are independent of the nominal weight, and they
sit near 1.0 so they compress well.

``weight_ewk`` is a plain scalar with no up/down and is not a systematic.

Which branches a given ntuple carries varies by channel, run and sample kind: the
b-tag sources are named per year and per payload (``btag_HF_statistic_2018`` vs
``btag_HF_statistic_2024Prompt``; Run 2 exposes ``as``/``pdf``/``ttbar`` where Run 3
exposes ``pdfas``/``hdamp``/``topmass``/``type3``/``bfragmentation``), channels with
no configured b-tag working point get none at all, and data carries only ``weight``.
The post-processor therefore DISCOVERS the ``weightsyst_*`` branches per input rather
than taking a fixed list, and every ratio column it finds is written. This table is
the separate question of which of those get a combine nuisance; branches absent from
a frame are skipped (``available_systematics``), so entries here are safe but a
branch missing from this table is silently dropped from the cards.
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
    "weightsyst_l1prefiring":     ("CMS_l1_ecal_prefiring", "corr"),
    # The preselection now applies one combined lepton SF per flavor (reco * tight ID *
    # trigger, lepSFWrapper in selections.cpp) and writes it as a single weightsyst_eleSF /
    # weightsyst_muoSF branch, replacing the old per-component
    # weightsyst_{muon,electron}{id,reco,trigger}. The components are no longer separable,
    # so each flavor contributes one nuisance.
    "weightsyst_muoSF":           ("CMS_eff_m",             "corr"),
    "weightsyst_eleSF":           ("CMS_eff_e",             "corr"),
}

# --- theory / pileup: plain and _withbSF variants of the same nuisance ------------------
# Each of these variations is written twice by the preselection: the plain branch, and a
# _withbSF companion that carries the variation correlated with its b-tag response
# (correlateWeightWithBTagSource in weights.cpp). The full b-tag breakdown prescribes the
# _withbSF ones, but they are written ONLY where b-tag SFs were applied -- 0lep_3FJ has no
# configured working point and carries the plain branches alone. So both names map to the
# SAME nuisance and `preferred_branches` keeps whichever a given frame actually has: the
# nuisance is neither written twice nor silently lost. Never list a pair as two nuisances.
_WITHBSF_PAIRS = {
    "weightsyst_muF":    ("QCDscale_fac", "proc"),
    "weightsyst_muR":    ("QCDscale_ren", "proc"),
    "weightsyst_PSISR":  ("ps_isr",       "proc"),
    "weightsyst_PSFSR":  ("ps_fsr",       "proc"),
    "weightsyst_pileup": ("CMS_pileup",   "era"),
}
# plain branch -> its _withbSF companion
WITHBSF_OF = {plain: plain + "_withbSF" for plain in _WITHBSF_PAIRS}
for _plain, _spec in _WITHBSF_PAIRS.items():
    SYST_WEIGHTS[_plain] = _spec
    SYST_WEIGHTS[_plain + "_withbSF"] = _spec

# --- AK4 b-tagging: the full per-source breakdown ---------------------------------------
# The preselection writes both a summary and a breakdown of the b-tag response; taking
# both would double-count, so this is the breakdown ONLY:
#
#   * HF shape sources, correlated across years (BTV convention). The payloads name them
#     differently per run -- Run 2 exposes as / pdf / ttbar, Run 3 pdfas / hdamp /
#     topmass / type3 / bfragmentation -- and a given file carries only its own set, so
#     all of them are listed and the absent ones are skipped per frame.
#   * The per-year statistical component, decorrelated by year.
#   * LF correlated + the per-year LF component.
#
# DELIBERATELY EXCLUDED, both verified against the ntuples:
#   * weightsyst_btag_HF_correlated -- the quadrature TOTAL of the HF shape sources
#     (rms 0.00704 vs 0.00694 for their quadrature sum, correlation 0.999), so it would
#     double-count the whole HF breakdown.
#   * weightsyst_btag_HF_uncorrelated_<year> -- a near-duplicate of
#     HF_statistic_<year> (correlation 0.998); the breakdown uses the statistic one.
BTAG_YEARS = ["2016preVFP", "2016postVFP", "2017", "2018", "2024Prompt"]
BTAG_HF_SOURCES = [
    "as", "pdf", "ttbar",                                     # Run 2 payloads
    "pdfas", "hdamp", "topmass", "type3", "bfragmentation",   # Run 3 payloads
]
for _src in BTAG_HF_SOURCES:
    SYST_WEIGHTS[f"weightsyst_btag_HF_{_src}"] = (f"CMS_btag_HF_{_src}", "corr")
SYST_WEIGHTS["weightsyst_btag_LF_correlated"] = ("CMS_btag_LF", "corr")
for _yr in BTAG_YEARS:
    SYST_WEIGHTS[f"weightsyst_btag_HF_statistic_{_yr}"] = (f"CMS_btag_HF_stat_{_yr}", "corr")
    SYST_WEIGHTS[f"weightsyst_btag_LF_uncorrelated_{_yr}"] = (f"CMS_btag_LF_stat_{_yr}", "corr")

# The b-tag payload's own response to JES/JER. These are companions of the b-tag SF, NOT
# of the analysis JES/JER nuisances (which are separate event sets in the `variation`
# column, named by jec_nuisance_name), so they get their own names and must not be merged
# into CMS_scale_j_* / CMS_res_j_*. Run 3 writes one inclusive BTV JES-total; Run 2 writes
# no BTV JES branch at all.
SYST_WEIGHTS["weightsyst_jes"] = ("CMS_btag_jes", "era")
SYST_WEIGHTS["weightsyst_jer"] = ("CMS_btag_jer", "era")


def preferred_branches(present):
    """Filter a list of present branches down to one branch per nuisance.

    Drops the plain variant of any _withbSF pair whose companion is also present, so a
    frame carrying both contributes that nuisance once (from the b-tag-correlated
    version) while a frame carrying only the plain branch still contributes it.
    """
    present = list(present)
    have = set(present)
    drop = {plain for plain, wb in WITHBSF_OF.items() if wb in have and plain in have}
    return [b for b in present if b not in drop]


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
