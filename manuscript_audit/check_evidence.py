"""Independent manuscript evidence audit; leaves source results untouched."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.metrics import compute_eer as canonical_eer

out = ROOT / 'manuscript_audit'
frames, rows = {}, []
reference = None
for path in sorted((ROOT/'results').glob('preds_*.csv')):
    model, condition = path.stem[6:].split('_', 1)
    df = pd.read_csv(path).sort_values('utterance_id').reset_index(drop=True)
    assert len(df) == 18769 and df.utterance_id.is_unique
    assert set(df.true_label) == {0, 1}
    assert np.isfinite(df.score).all() and df.score.between(0, 1).all()
    identity = df[['utterance_id','true_label']]
    if reference is None: reference = identity
    assert reference.equals(identity), path
    y, s = df.true_label.to_numpy(), df.score.to_numpy()
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    fnr = 1-tpr
    i = np.argmin(np.abs(fpr-fnr))
    eer = (fpr[i]+fnr[i])/2
    dcf = np.min(1.9*fpr+fnr)
    ceer, cthr = canonical_eer(y, s)
    frames[model, condition] = (y, s, (s>=cthr).astype(int))
    rows.append(dict(model=model, condition=condition, n=len(df), bona=int((y==0).sum()),
                     spoof=int((y==1).sum()), independent_eer=eer, canonical_eer=ceer,
                     independent_min_dcf=dcf, ties=len(s)-len(np.unique(s)),
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
metrics=pd.DataFrame(rows)
metrics.to_csv(out/'independent_metrics.csv',index=False)
existing=pd.read_csv(ROOT/'results/tables/metrics_long.csv')
joined=metrics.merge(existing,on=['model','condition'],suffixes=('_audit','_repo'))
print('FILES',len(metrics),'ALIGNMENT AND LABELS PASS; bona/spoof',rows[0]['bona'],rows[0]['spoof'])
print('MAX EER DIFF',float(abs(joined.independent_eer-joined.eer).max()),
      'MAX DCF DIFF',float(abs(joined.independent_min_dcf-joined.min_dcf).max()))
print(metrics.pivot(index='model',columns='condition',values='independent_eer').mul(100).round(4).to_string())
mc=[]
for r in pd.read_csv(ROOT/'results/tables/table_mcnemar.csv').to_dict('records'):
    a,b=r['comparison'].split(' vs ')
    y,_,pa=frames[a,r['condition']]; _,_,pb=frames[b,r['condition']]
    n01=int(((pa==y)&(pb!=y)).sum()); n10=int(((pa!=y)&(pb==y)).sum())
    stat=(abs(n01-n10)-1)**2/(n01+n10)
    assert n01==r['only_first_correct'] and n10==r['only_second_correct']
    assert abs(stat-r['chi2'])<1e-9
    mc.append(dict(comparison=r['comparison'],condition=r['condition'],n01=n01,n10=n10,
                   chi2=stat,p_value=float(chi2.sf(stat,1))))
mc=pd.DataFrame(mc)
for pair in mc.comparison.unique():
    subset=mc[mc.comparison==pair].sort_values('p_value').copy()
    p=subset.p_value.to_numpy(); adjusted=np.minimum(1,np.maximum.accumulate(p*np.arange(len(p),0,-1)))
    mc.loc[subset.index,'holm_p_within_7_conditions']=adjusted
mc.to_csv(out/'independent_mcnemar.csv',index=False)
print('ALL',len(mc),'MCNEMAR COUNTS AND STATISTICS MATCH')
print(mc[mc.comparison.isin(['C1 vs A1','C2 vs C1'])].to_string(index=False))

import torch
checks={}
for name,path in [('A0','setup_a/best_a0_final_eer0156.pt'),('A1','setup_a/best_a_ep13_eer0105_0713_0401.pt'),
                  ('B0','setup_b/best_b_ep4_eer0759_0624_1407.pt'),('B1','setup_b/best_b1_k64synthetic_ep9_eer0604.pt'),
                  ('C0','setup_c/best_c_ep12_eer0566_0710_0112.pt'),('C1','setup_c/best_c1_attention.pt'),('C2','setup_c/best_c2_concat.pt')]:
    checkpoint=torch.load(ROOT/'checkpoints'/path,map_location='cpu',weights_only=False)
    state=checkpoint.get('model_state',checkpoint.get('model_state_dict',checkpoint.get('state_dict',checkpoint)))
    checks[name]=state
    metadata={k:v for k,v in checkpoint.items() if not isinstance(v,(dict,torch.Tensor))}
    print('CHECKPOINT',name,str(metadata)[:1200])
c1,c2=checks['C1'],checks['C2']
for stream in ['stream1.','stream2.']:
    keys=[k for k in c1 if k.startswith(stream)]
    print('STREAM EQUALITY',stream,len(keys),sum(torch.equal(c1[k],c2[k]) for k in keys))
keys=[k for k in c1 if k.startswith('stream1.') and c1[k].is_floating_point()]
x=torch.cat([c1[k].flatten().double() for k in keys]); z=torch.cat([checks['A1'][k].flatten().double() for k in keys])
print('C1/A1 stream1 cosine including float buffers',torch.nn.functional.cosine_similarity(x,z,dim=0).item())
print('DONE')
