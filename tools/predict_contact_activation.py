"""Feasible secant geometry guess across activation distances; never transfer history."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from packing_fem.rigid_inclusion import make_periodic,make_contact
from packing_fem.explicit_material import geometry
from tools.assess_contact_repair import PHYSICAL


def predict(first,second,target,out):
    first=Path(first);second=Path(second);out=Path(out)
    if out.exists():raise FileExistsError(out)
    histories=[];states=[];configs=[];receipts=[]
    for run in [first,second]:
        c=json.loads((run/'config.json').read_text());configs.append(c)
        audit=json.loads((run/'audit.json').read_text());last=audit['states'][-1]
        path=run/f"state-{last['step']:04d}.npz"
        if not audit['passed'] or hashlib.sha256(path.read_bytes()).hexdigest()!=last['state_sha256']:raise ValueError('Unaudited source state')
        with np.load(run/'state-0000.npz') as f:histories.append(dict(f))
        with np.load(path) as f:states.append(dict(f))
        receipts.append(dict(run=str(run),state_sha256=last['state_sha256'],dhat_mm=c['ipc_dhat_mm']))
    old=histories[0]
    if set(old)!=set(histories[1]) or any(not np.array_equal(old[k],histories[1][k]) for k in old):raise ValueError('Accepted material histories differ')
    a,b=states
    for state in states:
        if not np.array_equal(state['X'],old['X']) or not np.array_equal(state['t'],old['t']):raise ValueError('Source solve changed the reference chart')
    if any(configs[0].get(k)!=configs[1].get(k) for k in PHYSICAL):raise ValueError('Source material or physical conditions differ')
    if not np.array_equal(a['q'][-6:],b['q'][-6:]):raise ValueError('Source macroscopic strains differ')
    da,db=[c['ipc_dhat_mm'] for c in configs]
    if not 0<target<db<da:raise ValueError('Expected decreasing activation distances')
    ratio=(target-db)/(db-da);direction=ratio*(b['q']-a['q']);direction[-6:]=0.
    c=dict(configs[1],ipc_dhat_mm=target);p=make_periodic(old['X'],c['length_mm'],c);contact=make_contact(old['X'],old['t'],p,c)
    bound=contact.ccd(b['q'],direction);fraction=1. if bound>=1 else .9*bound
    for _ in range(20):
        guess=b['q']+fraction*direction
        try:volume=geometry(p.positions(guess),old['t'])[0];break
        except ValueError:fraction*=.5
    else:raise ValueError('No positive-volume secant guess')
    np.savez_compressed(out,q=guess,oldq=old['q'],X=old['X'],t=old['t'],b=old['b'],ep=old['ep'])
    report=dict(sources=receipts,target_dhat_mm=target,secant_ratio=ratio,feasible_fraction=fraction,
        minimum_element_volume_mm3=float(volume.min()),advanced_material_history_used=False,
        history_source=str(first/'state-0000.npz'),geometry_only=True)
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--first',required=True);p.add_argument('--second',required=True);p.add_argument('--target',type=float,required=True);p.add_argument('--out',required=True);a=p.parse_args();print(predict(a.first,a.second,a.target,a.out))
