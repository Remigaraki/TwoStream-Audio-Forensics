"""Fresh paired utterance bootstrap, seed 42; not recovered original intervals."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
models=['A0','A1','B0','C1','C2']
conditions=['clean','opus_16','opus_32','opus_64','mp3_64','mp3_128','aac_128']
prepared={}
for cond in conditions:
    for model in models:
        df=pd.read_csv(root/f'results/preds_{model}_{cond}.csv').sort_values('utterance_id')
        y=df.true_label.to_numpy(); s=df.score.to_numpy()
        order=np.argsort(s,kind='stable'); sy=y[order]; ss=s[order]
        ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
        prepared[model,cond]=(order,sy,ends)
n=len(y); rng=np.random.default_rng(42)
eers=np.empty((1000,len(conditions),len(models)))
dcfs=np.empty((2000,len(conditions),len(models)))
for rep in range(2000):
    weights=np.bincount(rng.integers(0,n,n),minlength=n)
    for ci,cond in enumerate(conditions):
        for mi,model in enumerate(models):
            order,sy,ends=prepared[model,cond]
            w=weights[order]; nb=w[sy==0].sum(); ns=w[sy==1].sum()
            accepted_spoof=np.r_[0,np.cumsum(w*sy)[ends]]/ns
            rejected_bona=1-np.r_[0,np.cumsum(w*(1-sy))[ends]]/nb
            dcfs[rep,ci,mi]=np.min(1.9*rejected_bona+accepted_spoof)
            if rep<1000:
                i=np.argmin(abs(rejected_bona-accepted_spoof))
                eers[rep,ci,mi]=(rejected_bona[i]+accepted_spoof[i])/2*100
    if (rep+1)%500==0: print('resamples',rep+1,flush=True)
rows=[]
for ci,cond in enumerate(conditions):
    for a,b in [('C1','A1'),('C1','C2'),('A1','A0'),('C1','B0')]:
        ai=models.index(a); bi=models.index(b)
        lo,hi=np.quantile(eers[:,ci,ai]-eers[:,ci,bi],[.025,.975])
        dl,dh=np.quantile(dcfs[:,ci,ai]-dcfs[:,ci,bi],[.025,.975])
        rows.append(dict(condition=cond,comparison=f'{a} minus {b}',eer_difference_ci_low=lo,
                         eer_difference_ci_high=hi,min_dcf_difference_ci_low=dl,min_dcf_difference_ci_high=dh))
pd.DataFrame(rows).to_csv(root/'manuscript_audit/fresh_bootstrap_differences.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
