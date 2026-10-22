import numpy as np,json
from pathlib import Path
from tools.run_barrier_compaction import fixture
from packing_fem.explicit_material import geometry
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.area_contact import AreaContact
root=Path('results/refined-contact-probe-002');root.mkdir(exist_ok=False)
with np.load('exports/contact-cycle-cfl-001/fine/run/last-state.npz') as z:
 X=z['reference_points'];x=z['points'];t=z['tets'];F=deformation(z['generalized_position'][-6:])
D=np.stack([X[t[:,k]]-X[t[:,0]] for k in (1,2,3)],axis=2);inv=np.linalg.inv(D)
rows=[]
for n in (1,2,3):
 Y,T=fixture(n);weights=[];owners=[]
 for y in Y:
  v=np.einsum('tij,tj->ti',inv,y-X[t[:,0]]);w=np.column_stack((1-v.sum(1),v));valid=np.flatnonzero(w.min(1)>-1e-9)
  if not len(valid):raise ValueError('outside')
  k=valid[0];owners.append(k);weights.append(w[k])
 y=np.einsum('ni,nij->nj',np.array(weights),x[t[np.array(owners)]]);p=Periodic(Y,4.)
 c=AreaContact(Y,T,p,20.,.3,.1,thickness=.02,max_candidates=1000000)
 try:
  force,extra,h,r=c.evaluate(y,y*0,F*4,F*0,0.);row=dict(n=n,nodes=len(Y),tets=len(T),report=r,error=None)
 except ValueError as e:row=dict(n=n,nodes=len(Y),tets=len(T),error=str(e),type=type(e).__name__)
 print(row,flush=True);rows.append(row)
 np.savez_compressed(root/f'mesh-{n}.npz',reference_points=Y,points=y,tets=T)
(root/'summary.json').write_text(json.dumps(rows,indent=2))
