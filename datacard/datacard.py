import os
import sys
import argparse
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
from scipy.stats import gamma

# The systematics definition lives with the code that writes the predictions, so
# the branch list and nuisance naming cannot drift between the two.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "abcd"))
from systematics import SYST_WEIGHTS, jec_nuisance_name, nuisance_name, ratio_columns


def get_poisson_uncertainty(n):
    """
    Calculate Poisson uncertainties for integer observation n using Gamma distribution.
    Matches the logic from the old script.
    """
    n = round(n)
    up = round(gamma.ppf(1 - (1 - 0.9973) / 2, n + 1))
    dn = round(gamma.ppf((1 - 0.9973) / 2, n)) if n > 0 else 0
    return up, dn

def get_yields_from_df(df, cuts):
    """
    Get the yield (sum of weights) from a pandas DataFrame applying the given cuts.
    """
    df_cut = df.query(cuts)
    return df_cut["weight"].sum()

def get_data_yields_from_df(df, cuts):
    """
    Get data yields from a pandas DataFrame applying the given cuts.
    Returns integer sum.
    """
    df_cut = df.query(cuts)
    return len(df_cut)

def available_systematics(df):
    """The systematic branches whose up/down ratio columns are present in ``df``."""
    present = []
    for branch in SYST_WEIGHTS:
        up_col, dn_col = ratio_columns(branch)
        if up_col in df.columns and dn_col in df.columns:
            present.append(branch)
    return present


LEPTON_TAGS = {"0lep": "ZeroLep", "1lep": "OneLep", "2lep": "TwoLep"}


def lepton_tag(process_name):
    """'ZeroLep' / 'OneLep' / ... from a process name like '0lep_3fj_r2'.

    Only used to name the background process and the ABCD rateParams. Combine
    treats those as opaque strings, so the tag is cosmetic -- but every card
    used to say OneLep regardless of channel, which was simply wrong for the
    0-lepton ones. Anything unrecognised keeps the old OneLep spelling.
    """
    return LEPTON_TAGS.get(process_name.split("_")[0], "OneLep")


def kappa_cell(kv):
    """The datacard cell for one region's ``(kappa_dn, kappa_up)``, '-' if absent."""
    return "-" if kv is None else f"{kv[0]:.4f}/{kv[1]:.4f}"


def contributes(per_region):
    """True if a nuisance has any visible effect across its regions.

    The test is on the *written* cell rather than the raw ratio: a kappa of
    1.00003 rounds to 1.0000 in the card and is exactly as inert as a true 1.0,
    so both must be judged the same way.

    Dropping the inert ones matters. A nuisance that is 1.0 in every bin is an
    exactly flat direction in the likelihood, and once the per-scan cards are
    combined a single such nuisance makes the covariance matrix singular: HESSE
    fails (fit status 300, covQual 0), no parameter gets a usable error, and
    every combine Impacts fit then has no +-sigma to shift to and is dropped.
    """
    return any(kappa_cell(kv) not in ("-", "1.0000/1.0000")
               for kv in per_region.values())


def compute_syst_kappas(sig_df, region_dfs, proc_name, scan_name):
    """lnN kappas for every systematic, per ABCD region, for the signal.

    Only signal is affected: the background rate is a rateParam driven from the
    data control regions, so it carries no MC systematic.

    kappa is the raw varied-over-nominal yield ratio in the region, with no
    inclusive normalization divided out: every variation carries its full effect,
    normalization and acceptance together. This matches the convention of the
    Run 2 semileptonic datacard script, which took the plain per-region ratio for
    both the weight and the JES/JER systematics. (That script also symmetrized to
    a single ``1 + max(|dn|, |up|)`` lnN; we keep the asymmetric dn/up form, which
    combine reads natively and which preserves the two directions.)

    Returns ``{nuisance: {region: (kappa_dn, kappa_up)}}`` for every systematic
    whose columns are present and which has an effect in at least one region.
    """
    kappas = {}

    for branch in available_systematics(sig_df):
        up_col, dn_col = ratio_columns(branch)

        per_region = {}
        for region_id, df_r in region_dfs.items():
            w = df_r["weight"].to_numpy()
            nom = w.sum()
            if nom <= 0 or len(df_r) == 0:
                per_region[region_id] = None
                continue
            k_up = (w * df_r[up_col].to_numpy()).sum() / nom
            k_dn = (w * df_r[dn_col].to_numpy()).sum() / nom
            if not (np.isfinite(k_up) and np.isfinite(k_dn) and k_up > 0 and k_dn > 0):
                per_region[region_id] = None
                continue
            per_region[region_id] = (k_dn, k_up)

        # A systematic that comes out to 1 in every region (muR, or the lepton
        # SFs in a 0-lepton channel) is dropped: it constrains nothing, and
        # leaving it in makes the combined likelihood singular -- see
        # contributes(). The cost is that an omitted line no longer distinguishes
        # "evaluated, no effect" from "never computed"; the summary printed after
        # the card is written records which were dropped.
        if not contributes(per_region):
            continue

        kappas[nuisance_name(branch, proc_name, scan_name)] = per_region

    return kappas


def _jec_region_kappas(nom, sub, region_final_cuts):
    """Per-region (kappa_dn, kappa_up) for one JES/JER source from its up/dn subframes.

    ``sub`` maps 'up'/'dn' -> the variation's signal frame (already restricted to any
    year). A missing direction contributes a ratio of 1 (no variation) on that side.
    """
    per_region = {}
    for region_id, cut in region_final_cuts.items():
        nom_y = nom.query(cut)["weight"].sum()
        if nom_y <= 0:
            per_region[region_id] = None
            continue

        def _region_yield(direction):
            s = sub.get(direction)
            if s is None or len(s) == 0:
                return nom_y  # missing direction -> no variation on that side
            return s.query(cut)["weight"].sum()

        k_up = _region_yield("up") / nom_y
        k_dn = _region_yield("dn") / nom_y
        if not (np.isfinite(k_up) and np.isfinite(k_dn) and k_up > 0 and k_dn > 0):
            per_region[region_id] = None
            continue
        per_region[region_id] = (k_dn, k_up)
    return per_region


def compute_jec_kappas(sig_df, region_final_cuts, proc_name):
    """lnN kappas for the JES/JER (kinematic) variations, per ABCD region, for signal.

    Each variation is a separate signal event set carried in the ``variation`` column
    (candidate post-processor + per-variation inference). The kappa in a region is the
    varied-over-nominal signal yield there, applying the same region cut to that
    variation's rows — which carry that variation's own dnn/bdt/candidate scores.

    JES '*Year' regrouped sources are decorrelated per data-taking year when a ``year``
    column is present (one nuisance per year, restricted to that year's events); all
    other JES sources and JER correlate across years. Returns
    ``{nuisance: {region: (kappa_dn, kappa_up)}}``.
    """
    if "variation" not in sig_df.columns:
        return {}
    varcol = sig_df["variation"].astype(str)
    nom = sig_df[varcol == "nominal"]
    if len(nom) == 0:
        return {}
    has_year = "year" in sig_df.columns

    # Group variation labels into sources by stripping the Up/Dn direction.
    sources = {}
    for v in set(varcol.unique()):
        if v == "nominal":
            continue
        if v.endswith("Up"):
            sources.setdefault(v[:-2], {})["up"] = v
        elif v.endswith("Dn"):
            sources.setdefault(v[:-2], {})["dn"] = v

    kappas = {}
    for source, ud in sorted(sources.items()):
        stem = source[3:] if source.startswith("jes") else source
        per_year = stem.endswith("Year") and has_year
        sub_all = {d: sig_df[varcol == v] for d, v in ud.items()}

        year_groups = ([(y, str(y)) for y in sorted(set(nom["year"].astype(str)))]
                       if per_year else [(None, None)])
        for year_val, year_tag in year_groups:
            if year_val is None:
                nom_g, sub_g = nom, sub_all
            else:
                nom_g = nom[nom["year"].astype(str) == year_val]
                sub_g = {d: s[s["year"].astype(str) == year_val] for d, s in sub_all.items()}
            per_region = _jec_region_kappas(nom_g, sub_g, region_final_cuts)
            if contributes(per_region):
                kappas[jec_nuisance_name(source, proc_name, year=year_tag)] = per_region
    return kappas


def build_kin_terms(scan_info):
    """
    Build the list of kinematic score selections (>=) that define the phase space.
    Supports both the single-V phase space (v_cut / boosted_v_candidate_score, e.g. 1FJ/2FJ
    channels) and the two-V phase space (v1_cut, v2_cut / boosted_v{1,2}_candidate_score,
    e.g. the 3FJ channels). The ABCD axes (dnn_score, bdt_score) are handled separately.
    """
    terms = [f"boosted_h_candidate_score >= {scan_info['h_cut']}"]
    if "v1_cut" in scan_info or "v2_cut" in scan_info:
        terms.append(f"boosted_v1_candidate_score >= {scan_info['v1_cut']}")
        terms.append(f"boosted_v2_candidate_score >= {scan_info['v2_cut']}")
    else:
        terms.append(f"boosted_v_candidate_score >= {scan_info['v_cut']}")
    return terms

def create_abcd_datacard_single(process_name, out_name, scan_name, scan_info, orthogonality_cuts, sig_df, data_df, combination="and", unblind=False):
    """
    Create a single datacard for a specific scan.
    """

    # The full frame (all variations) drives the JES/JER kappas; the nominal subset
    # drives the region yields, weight systematics and signal stat uncertainty.
    full_sig_df = sig_df
    if "variation" in sig_df.columns:
        sig_df = sig_df[sig_df["variation"] == "nominal"]
    if data_df is not None and "variation" in data_df.columns:
        data_df = data_df[data_df["variation"] == "nominal"]

    yields = {scan_name: {}}
    observations = {scan_name: {}}
    poisson_errs = {scan_name: {}}

    # Base cuts
    dnn_cut = scan_info["dnn_cut"]
    vbs_cut = scan_info["vbs_cut"]

    # Kinematic phase space (h + V candidate scores), shared by all ABCD regions.
    # For single-fatjet channels (combination == "or") an event has either an H or a
    # V candidate (never both), so the candidate-score terms are OR'd; otherwise AND'd.
    kin_combine = " or " if combination == "or" else " and "
    kin_cut = "(" + kin_combine.join(build_kin_terms(scan_info)) + ")"

    A_cut = f"{kin_cut} and (dnn_score > {dnn_cut}) and (bdt_score > {vbs_cut})"
    B_cut = f"{kin_cut} and (dnn_score > {dnn_cut}) and (bdt_score < {vbs_cut})"
    C_cut = f"{kin_cut} and (dnn_score < {dnn_cut}) and (bdt_score > {vbs_cut})"
    D_cut = f"{kin_cut} and (dnn_score < {dnn_cut}) and (bdt_score < {vbs_cut})"
    
    cuts_dict = {"A": A_cut, "B": B_cut, "C": C_cut, "D": D_cut}

    ortho_str = ""
    if orthogonality_cuts:
        ortho_str = " and ".join([f"not ({cut})" for cut in orthogonality_cuts]) + " and "

    region_sig_dfs = {}
    region_final_cuts = {}
    for region_id in ["A", "B", "C", "D"]:
        final_cut = ortho_str + cuts_dict[region_id]
        region_final_cuts[region_id] = final_cut

        # Data
        obs_val = get_data_yields_from_df(data_df, final_cut)
        if region_id == "A" and not unblind:
            observations[scan_name][region_id] = 1 # blind SR
            poisson_errs[scan_name][region_id] = (1, 1)
        else:
            observations[scan_name][region_id] = int(obs_val)
            poisson_errs[scan_name][region_id] = get_poisson_uncertainty(obs_val)
            
        # Signal
        # compute signal sum weights and sum weights^2 to get stat uncertainty
        df_sig_cut = sig_df.query(final_cut)
        region_sig_dfs[region_id] = df_sig_cut
        sig_sumw = df_sig_cut["weight"].sum()
        sig_sumw2 = (df_sig_cut["weight"] ** 2).sum()
        yields[scan_name][region_id] = sig_sumw
        # multiplicative lnN: 1 + sqrt(sumw2)/sumw (if sumw > 0)
        if sig_sumw > 0:
            sig_err = 1.0 + np.sqrt(sig_sumw2) / sig_sumw
        else:
            sig_err = 1.0
        # store per-region signal stat uncertainty
        poisson_errs[scan_name][f"sig_{region_id}"] = sig_err

    # Write datacard in old format (backgrounds first then signals)

    # if output dir doesn't exist then create it
    out_dir = os.path.dirname(out_name)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(out_name, "w+") as f:
        f.write("imax 4 number of channels\n")
        f.write("jmax 1 number of backgrounds\n")
        f.write("kmax * number of nuisance parameters\n")
        f.write("-" * 150 + "\n")

        # Observation block
        f.write(f"{'bin':<60}{'A':<20}{'B':<20}{'C':<20}{'D':<20}\n")
        f.write(f"{'observation':<60}{observations[scan_name]['A']:<20}{observations[scan_name]['B']:<20}{observations[scan_name]['C']:<20}{observations[scan_name]['D']:<20}\n")
        f.write("-" * 150 + "\n")

        # Process block
        lep = lepton_tag(process_name)
        bkg_name = f"TotalBkg_{lep}_{process_name}_{scan_name}"
        f.write(f"{'bin':<60}{'A':<50}{'B':<50}{'C':<50}{'D':<50}{'A':<20}{'B':<20}{'C':<20}{'D':<20}\n")
        f.write(f"{'process':<60}{bkg_name:<50}{bkg_name:<50}{bkg_name:<50}{bkg_name:<50}{'TotalSig':<20}{'TotalSig':<20}{'TotalSig':<20}{'TotalSig':<20}\n")
        f.write(f"{'process':<60}{'1':<50}{'1':<50}{'1':<50}{'1':<50}{'0':<20}{'0':<20}{'0':<20}{'0':<20}\n")
        f.write(f"{'rate':<60}{1:<50}{1:<50}{1:<50}{1:<50}{yields[scan_name]['A']:<20.5f}{yields[scan_name]['B']:<20.5f}{yields[scan_name]['C']:<20.5f}{yields[scan_name]['D']:<20.5f}\n")
        f.write("-" * 150 + "\n")

        # Systematics
        f.write(f"{f'CMS_{process_name}_{scan_name}_control_abcd_syst':<50}{'lnN':<10}{'1.35':<50}{'-':<50}{'-':<50}{'-':<50}{'-':<20}{'-':<20}{'-':<20}{'-':<20}\n")
        f.write(f"{'lumi_13TeV_correlated':<50}{'lnN':<10}{'-':<50}{'-':<50}{'-':<50}{'-':<50}{'1.016':<20}{'1.016':<20}{'1.016':<20}{'1.016':<20}\n")
        
        # Signal stat systematics
        sigA_err = poisson_errs[scan_name]['sig_A']
        sigB_err = poisson_errs[scan_name]['sig_B']
        sigC_err = poisson_errs[scan_name]['sig_C']
        sigD_err = poisson_errs[scan_name]['sig_D']
        # One line per region, touching only that region. A region with no signal
        # yield gets an error of exactly 1 and is skipped for the same reason the
        # weight systematics are -- see contributes().
        dropped = []
        for i, (region_id, err) in enumerate(
                zip("ABCD", (sigA_err, sigB_err, sigC_err, sigD_err))):
            name = f"CMS_{process_name}_{scan_name}_mcstat_Region{region_id}"
            if f"{err:.5f}" == "1.00000":
                dropped.append(name)
                continue
            cells = ["-"] * 4
            cells[i] = f"{err:.5f}"
            f.write(f"{name:<50}{'lnN':<10}{'-':<50}{'-':<50}{'-':<50}{'-':<50}"
                    f"{cells[0]:<20}{cells[1]:<20}{cells[2]:<20}{cells[3]:<20}\n")

        # Weight-based systematics, signal only (the background rate is a
        # rateParam measured from the data control regions). Asymmetric lnN is
        # written in combine's "down/up" form.
        syst_kappas = compute_syst_kappas(sig_df, region_sig_dfs, process_name, scan_name)
        n_inert_weight = len(available_systematics(sig_df)) - len(syst_kappas)
        for nuisance, per_region in sorted(syst_kappas.items()):
            cells = [kappa_cell(per_region.get(r)) for r in ["A", "B", "C", "D"]]
            f.write(f"{nuisance:<50}{'lnN':<10}{'-':<50}{'-':<50}{'-':<50}{'-':<50}"
                    f"{cells[0]:<20}{cells[1]:<20}{cells[2]:<20}{cells[3]:<20}\n")
        if not syst_kappas:
            if not available_systematics(sig_df):
                print(f"  WARNING: {scan_name}: signal predictions carry no systematic weight "
                      f"columns (re-run inference to add them); card has only the hardcoded nuisances")
            else:
                print(f"  WARNING: {scan_name}: systematic columns present but no region has a "
                      f"usable signal yield; no weight systematics written")

        # JES/JER (kinematic) systematics, signal only: per-region varied/nominal yield
        # ratios computed from the per-variation event sets carried in `variation`.
        jec_kappas = compute_jec_kappas(full_sig_df, region_final_cuts, process_name)
        for nuisance, per_region in sorted(jec_kappas.items()):
            cells = [kappa_cell(per_region.get(r)) for r in ["A", "B", "C", "D"]]
            f.write(f"{nuisance:<50}{'lnN':<10}{'-':<50}{'-':<50}{'-':<50}{'-':<50}"
                    f"{cells[0]:<20}{cells[1]:<20}{cells[2]:<20}{cells[3]:<20}\n")

        f.write("-" * 150 + "\n")

        # RateParams
        f.write(f"A_{lep}_{process_name}_{scan_name} rateParam       A                   {bkg_name}     (@0*@1/@2)\tB_{lep}_{process_name}_{scan_name},C_{lep}_{process_name}_{scan_name},D_{lep}_{process_name}_{scan_name}\n")
        for r in ['B', 'C', 'D']:
            o = observations[scan_name][r]
            up, dn = poisson_errs[scan_name][r]
            f.write(f"{r}_{lep}_{process_name}_{scan_name} rateParam       {r}                   {bkg_name}     {o}\t[{dn},{up}]\n")

    if dropped or n_inert_weight:
        bits = []
        if n_inert_weight:
            bits.append(f"{n_inert_weight} weight syst")
        if dropped:
            bits.append(", ".join(n.rsplit("_", 1)[-1] for n in dropped))
        print(f"  {scan_name}: dropped as having no effect: {'; '.join(bits)}")
    print(f"Datacard written to {out_name}")

def create_abcd_datacards(out_name_prefix, scans, sig_df, data_df, process_name, combination, unblind=False):
    """
    Create individual datacards for all scans using DataFrame inputs and cuts.
    """
    
    orthogonality_cuts = []
    
    for scan_name, scan_info in scans.items():
        out_name = f"{out_name_prefix}_{scan_name}.dat" if out_name_prefix else f"datacard_{scan_name}.dat"
        
        create_abcd_datacard_single(process_name, out_name, scan_name, scan_info, orthogonality_cuts, sig_df, data_df, combination, unblind)

        combine = " or " if combination == "or" else " and "
        ortho_cut = "(" + combine.join(build_kin_terms(scan_info)) + ")"
        orthogonality_cuts.append(ortho_cut)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ABCD datacard from DataFrames")
    parser.add_argument("--sig", default="signal.parquet", help="Signal predictions file (.parquet)")
    parser.add_argument("--data", default="data.parquet", help="Data predictions file (.parquet)")
    parser.add_argument("--out", default="datacard.txt", help="Output datacard text file")
    parser.add_argument("--config", default="regions.yaml", help="Yaml config containing scans")
    parser.add_argument("--unblind", action="store_true", help="Unblind SR")
    parser.add_argument("--proc", default="vbsvvh1lep", help="Process name for datacard (used in systematics naming)")
    parser.add_argument("--combination", default="and", choices=["and", "or"], help="Use AND or OR combination of orthogonality cuts for multiple scans (default is AND)")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config {args.config} not found. Ensure you created it.")
        exit(1)
        
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    scans = config.get("channels", {})
    
    # Open trees
    if os.path.exists(args.sig) and os.path.exists(args.data):
        sig_df = pd.read_parquet(args.sig)
        data_df = pd.read_parquet(args.data)

        # If the predictions contain both signal and background (label column), filter label==1
        if 'label' in sig_df.columns:
            sig_df = sig_df[sig_df['label'] == 1].copy()

        # Ensure weight column exists for signal; if not, assume weight=1
        if 'weight' not in sig_df.columns:
            sig_df['weight'] = 1.0

        create_abcd_datacards(args.out, scans, sig_df, data_df, args.proc, args.combination, args.unblind)
    else:
        print("Signal or Data files not found. Provide them via --sig and --data")

