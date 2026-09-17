import pandas as pd, yaml, numpy as np
channels = {
  "0lep_3fj_r2": ("0LEP_3FJ_RUN2","and"),"0lep_3fj_r3": ("0LEP_3FJ_RUN3","and"),
  "1lep_1fj_r2": ("1LEP_1FJ_RUN2","or"),"1lep_1fj_r3": ("1LEP_1FJ_RUN3","or"),
  "1lep_2fj_r2": ("1LEP_2FJ_RUN2","and"),"1lep_2fj_r3": ("1LEP_2FJ_RUN3","and"),
}
def terms(s):
    t=[f"boosted_h_candidate_score >= {s['h_cut']}"]
    if "v1_cut" in s or "v2_cut" in s:
        t+=[f"boosted_v1_candidate_score >= {s['v1_cut']}",f"boosted_v2_candidate_score >= {s['v2_cut']}"]
    else: t.append(f"boosted_v_candidate_score >= {s['v_cut']}")
    return t
def cutset(s,comb):
    kc=" or " if comb=="or" else " and "; kin="("+kc.join(terms(s))+")"
    d=s['dnn_cut']; v=s['vbs_cut']
    return {"A":f"{kin} and (dnn_score > {d}) and (vbs_detajj > {v})",
            "B":f"{kin} and (dnn_score > {d}) and (vbs_detajj < {v})",
            "C":f"{kin} and (dnn_score < {d}) and (vbs_detajj > {v})",
            "D":f"{kin} and (dnn_score < {d}) and (vbs_detajj < {v})"}
want=["weight","label","dnn_score","vbs_detajj","boosted_h_candidate_score",
      "boosted_v_candidate_score","boosted_v1_candidate_score","boosted_v2_candidate_score"]
results={f"Scan{i}":{} for i in range(1,6)}
for ch,(d,comb) in channels.items():
    base=f"{d}/single/version_0"
    cfg=yaml.safe_load(open(f"{base}/regions.yaml"))["channels"]
    head=pd.read_csv(f"{base}/predictions_single.csv",nrows=0).columns
    need=[c for c in want if c in head]
    mc=pd.read_csv(f"{base}/predictions_single.csv",usecols=need)
    dcols=pd.read_csv(f"{base}/predictions_single_data.csv",nrows=0).columns
    dneed=[c for c in need if c!="label" and c in dcols]
    data=pd.read_csv(f"{base}/predictions_single_data.csv",usecols=dneed)
    sig=mc[mc['label']==1]; bkg=mc[mc['label']==0]
    for sn,s in cfg.items():
        if sn not in results: continue
        cuts=cutset(s,comb); rec={}
        for r in "ABCD":
            ss=sig.query(cuts[r])["weight"].sum()
            bsub=bkg.query(cuts[r])["weight"]; bb=bsub.sum(); be=np.sqrt((bsub**2).sum())
            rec[r]=(ss,bb,be,len(data.query(cuts[r])))
        results[sn][ch]=rec
def f(x): return f"{x:.3f}"
def esc(c): return c.replace("_","\\_")
print(r"% Yields per scan. Region A data BLINDED -> ABCD pred Nhat_A = N_B N_C / N_D.")
print(r"% Signal = sum w (label==1); Bkg(MC) = sum w +/- sqrt(sum w^2) (label==0); Data = event count.")
for sn in [f"Scan{i}" for i in range(1,6)]:
    chans=[c for c in channels if c in results[sn]]
    print(r"\begin{table}[htbp]\centering\footnotesize")
    print(r"\caption{Yields in "+sn+r" by channel and ABCD region. Region A data blinded; $\hat{N}_A=N_B N_C/N_D$.}")
    print(r"\begin{tabular}{l l r r r}")
    print(r"\hline\hline")
    print(r"Channel & Region & Signal & Bkg (MC) & Data \\")
    print(r"\hline")
    for ch in chans:
        rec=results[sn][ch]
        for i,r in enumerate("ABCD"):
            ss,bb,be,nd=rec[r]
            chcell=esc(ch) if i==0 else ""
            if r=="A":
                NB,NC,ND=rec["B"][3],rec["C"][3],rec["D"][3]
                pred=NB*NC/ND if ND else float('nan')
                datacell=f"blind ($\\hat{{N}}_A={pred:.2f}$)"
            else: datacell=str(nd)
            print(f"{chcell} & {r} & {f(ss)} & ${f(bb)}\\pm{f(be)}$ & {datacell} \\\\")
        print(r"\hline")
    print(r"\hline\end{tabular}\end{table}")
    print()