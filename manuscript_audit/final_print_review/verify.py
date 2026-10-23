"""Read-only verification of research artifacts for the final print manuscript."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_curve
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.utils.metrics import compute_eer

frames, rows = {}, []
reference = None
for path in sorted((ROOT / 'results').glob('preds_*.csv')):
    model, condition = path.stem[6:].split('_', 1)
    df = pd.read_csv(path).sort_values('utterance_id').reset_index(drop=True)
    assert len(df) == 18769 and df.utterance_id.is_unique
    identity = df[['utterance_id', 'true_label']]
    if reference is None: reference = identity
    assert reference.equals(identity)
    y, s = df.true_label.to_numpy(), df.score.to_numpy()
    assert set(y) == {0, 1} and np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all()
    fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
    fnr = 1 - tpr
    i = np.argmin(abs(fpr - fnr))
    eer = (fpr[i] + fnr[i]) / 2
    dcf = np.min(1.9 * fpr + fnr)
    _, threshold = compute_eer(y, s)
    frames[model, condition] = (y, s, (s >= threshold).astype(int))
    rows.append(dict(model=model, condition=condition, n=len(y), bona=int((y==0).sum()),
                     spoof=int((y==1).sum()), eer=eer, min_dcf=dcf,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT/'verified_metrics.csv', index=False)
repo = pd.read_csv(ROOT/'results/tables/metrics_long.csv')
joined = metrics.merge(repo, on=['model','condition'], suffixes=('_audit','_repo'))
print('49 CSV pairing/score checks PASS; classes', rows[0]['bona'], rows[0]['spoof'])
print('Max absolute metric differences', abs(joined.eer_audit-joined.eer_repo).max(),
      abs(joined.min_dcf_audit-joined.min_dcf_repo).max())
tests=[]
for r in pd.read_csv(ROOT/'results/tables/table_mcnemar.csv').to_dict('records'):
    a,b=r['comparison'].split(' vs ')
    y,_,pa=frames[a,r['condition']]; _,_,pb=frames[b,r['condition']]
    n01=int(((pa==y)&(pb!=y)).sum()); n10=int(((pa!=y)&(pb==y)).sum())
    stat=(abs(n01-n10)-1)**2/(n01+n10)
    assert n01==r['only_first_correct'] and n10==r['only_second_correct']
    assert abs(stat-r['chi2']) < 1e-9
    tests.append(dict(comparison=r['comparison'],condition=r['condition'],first_only=n01,
                      second_only=n10,chi2=stat,p=float(chi2.sf(stat,1))))
tests=pd.DataFrame(tests)
for pair in tests.comparison.unique():
    sub=tests[tests.comparison==pair].sort_values('p')
    tests.loc[sub.index,'holm_p']=np.minimum(1,np.maximum.accumulate(sub.p.to_numpy()*np.arange(len(sub),0,-1)))
tests.to_csv(OUT/'verified_mcnemar.csv',index=False)
print('All',len(tests),'McNemar count/statistic checks PASS')
print(tests[tests.comparison.isin(['C1 vs A1','C2 vs C1','B1 vs B0'])].to_string(index=False))

paths={'A0':'setup_a/best_a0_final_eer0156.pt','A1':'setup_a/best_a_ep13_eer0105_0713_0401.pt',
       'B0':'setup_b/best_b_ep4_eer0759_0624_1407.pt','B1':'setup_b/best_b1_k64synthetic_ep9_eer0604.pt',
       'C0':'setup_c/best_c_ep12_eer0566_0710_0112.pt','C1':'setup_c/best_c1_attention.pt',
       'C2':'setup_c/best_c2_concat.pt'}
states={}
for model,path in paths.items():
    ck=torch.load(ROOT/'checkpoints'/path,map_location='cpu',weights_only=False)
    state=ck['model_state']; states[model]=state
    opt=ck['optimizer_state']; ids=opt['param_groups'][0]['params']
    step=sorted(set(float(v['step']) for v in opt['state'].values()))
    print('CKPT',model,'epoch',ck['epoch'],'val_eer',ck['val_eer'],'opt_params',len(ids),
          'steps',step,'lr',opt['param_groups'][0]['lr'],'wd',opt['param_groups'][0]['weight_decay'])
    print('opt_shapes',model,[list(v['exp_avg'].shape) for v in opt['state'].values()])
for stream in ['stream1.','stream2.']:
    keys=[k for k in states['C1'] if k.startswith(stream)]
    assert all(torch.equal(states['C1'][k],states['C2'][k]) for k in keys)
    print('Identical C1/C2',stream,len(keys),'tensors')
keys=[k for k in states['C1'] if k.startswith('stream1.') and states['C1'][k].is_floating_point()]
x=torch.cat([states['C1'][k].flatten().double() for k in keys]); z=torch.cat([states['A1'][k].flatten().double() for k in keys])
print('C1/A1 cosine including floating buffers',torch.nn.functional.cosine_similarity(x,z,dim=0).item())
from src.stream1.rawnet2 import RawNet2Encoder
enc=RawNet2Encoder(); pkeys=['stream1.'+k for k,_ in enc.named_parameters()]
x=torch.cat([states['C1'][k].flatten().double() for k in pkeys]); z=torch.cat([states['A1'][k].flatten().double() for k in pkeys])
print('C1/A1 cosine parameters only',torch.nn.functional.cosine_similarity(x,z,dim=0).item())
print('B0 alternative checkpoint',torch.load(ROOT/'checkpoints/setup_b/best_b_ep3_eer0750_0713_0222.pt',map_location='cpu',weights_only=False)['val_eer'])
print('DONE')
