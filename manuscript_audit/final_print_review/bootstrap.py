"""Fresh bootstrap sensitivity check; does not recover original printed intervals."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[2]; out=Path(__file__).resolve().parent
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
rows=[]; individual=[]
for ci,cond in enumerate(conditions):
    for mi,model in enumerate(models):
        lo,hi=np.quantile(eers[:,ci,mi],[.025,.975]); dl,dh=np.quantile(dcfs[:,ci,mi],[.025,.975])
        individual.append(dict(model=model,condition=cond,eer_low=lo,eer_high=hi,dcf_low=dl,dcf_high=dh))
    for a,b in [('C1','A1'),('C1','C2'),('A1','A0'),('C1','B0')]:
        ai=models.index(a); bi=models.index(b)
        lo,hi=np.quantile(eers[:,ci,ai]-eers[:,ci,bi],[.025,.975])
        dl,dh=np.quantile(dcfs[:,ci,ai]-dcfs[:,ci,bi],[.025,.975])
        rows.append(dict(condition=cond,comparison=f'{a} minus {b}',eer_low=lo,eer_high=hi,dcf_low=dl,dcf_high=dh))
pd.DataFrame(rows).to_csv(out/'fresh_difference_intervals.csv',index=False)
pd.DataFrame(individual).to_csv(out/'fresh_individual_intervals.csv',index=False)
np.savez_compressed(out/'fresh_bootstrap_samples.npz',eers=eers,dcfs=dcfs,models=models,conditions=conditions,seed=42)
print(pd.DataFrame(rows).to_string(index=False))
