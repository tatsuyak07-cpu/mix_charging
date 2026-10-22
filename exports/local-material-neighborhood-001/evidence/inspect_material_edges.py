import numpy as np
from tools.run_barrier_compaction import fixture
from packing_fem.explicit_periodic import Periodic
from packing_fem.area_contact import AreaContact
from packing_fem.barrier_contact import feature,relative
X,t=fixture(4);c=AreaContact(X,t,Periodic(X,4),20.,.3,.1,thickness=.02,max_candidates=1000000)
try:c.evaluate(X,X*0,np.eye(3)*4,np.zeros((3,3)),0.)
except ValueError as e:print(e)
selected=set()
for key,ids,kind,shift,_,_ in c.guard.neighbors:
 p=X[ids].copy();p[1 if kind=='pt' else 2:]+=4*np.asarray(shift)
 _,w=feature(p,kind);r=relative(p,kind,feature(p,kind)[0]);d=np.linalg.norm(r)
 if d>=.02:continue
 if kind=='pt':
  if not c.material_neighborhood.local(key,ids,kind,shift,p,w,X,np.eye(3)*4):selected.add(int(ids[0]))
for key,ids,kind,shift,_,_ in c.guard.neighbors:
 if kind!='ee':continue
 p=X[ids].copy();p[2:]+=4*np.asarray(shift)
 branch,w=feature(p,kind);d=np.linalg.norm(relative(p,kind,branch))
 if d>=.02 or any(int(k) in selected for k in ids):continue
 if c.material_neighborhood.local(('ee_material',*map(int,ids),*map(int,shift)),ids,kind,shift,p,w,X,np.eye(3)*4):continue
 print('distance',d,'ids',ids,'points',p,'weights',w,'endpoints',c.material_neighborhood.endpoints(p,w),flush=True)
 break
