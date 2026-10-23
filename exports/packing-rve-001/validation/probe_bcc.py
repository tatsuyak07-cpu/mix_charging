import numpy as np,time,json
from itertools import product
from packing_fem.sintered import draw,mesh,connectivity
from packing_fem.explicit_material import hex_to_tet
from packing_fem.explicit_periodic import Periodic
from packing_fem.area_contact import AreaContact
for n in (12,16,20,24):
 best=None
 for L in np.linspace(11.8,12.5,141):
  h=L/n;mask=np.zeros((n,n,n),bool)
  for p in (np.zeros(3),np.ones(3)*L/2):draw(mask,p,p,5.,h)
  for end in product((-.5,.5),repeat=3):draw(mask,np.zeros(3),L*np.array(end),1.,h)
  phi=1-mask.mean();err=abs(phi-.4)
  if best is None or err<best[0]:best=(err,L,mask.copy())
 err,L,mask=best;h=L/n;X,hexes,_=mesh(mask,h);t=hex_to_tet(X,hexes)
 row=dict(n=n,L=L,phi=1-mask.mean(),error=err,nodes=len(X),tets=len(t),connectivity=connectivity(mask));print(json.dumps(row),flush=True)
 st=time.perf_counter()
 try:
  contact=AreaContact(X,t,Periodic(X,L),20.,.3,.1,thickness=.02,max_candidates=2000000,multiple_patches=True,edge_stationarity_filter=True)
  f,e,hist,r=contact.evaluate(X,X*0,np.eye(3)*L,np.zeros((3,3)),0.);row.update(elapsed=time.perf_counter()-st,contact=r,search=contact.search_counts)
 except Exception as e:row.update(elapsed=time.perf_counter()-st,error=str(e))
 print(json.dumps(row),flush=True)
 np.savez_compressed(f'results/packing-rve-development-001/bcc-{n}.npz',points=X,tets=t,mask=mask,L=L)
