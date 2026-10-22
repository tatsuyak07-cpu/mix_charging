import numpy as np,json
from tools.run_barrier_compaction import fixture
from packing_fem.explicit_periodic import Periodic
from packing_fem.area_contact import AreaContact
from packing_fem.barrier_contact import feature,relative
X,t=fixture(4);c=AreaContact(X,t,Periodic(X,4.),20.,.3,.1,thickness=.02,max_candidates=1000000)
try:c.evaluate(X,X*0,np.eye(3)*4,np.zeros((3,3)),0.)
except ValueError:pass
sel={}
for key,ids,kind,shift,_,_ in c.guard.neighbors:
 if kind!='pt':continue
 z=X[ids].copy();z[1:]+=np.array(shift)*4
 branch,w=feature(z,'pt');r=relative(z,'pt',branch);d=np.linalg.norm(r)
 if d<.02:sel.setdefault(int(ids[0]),[]).append((d,key,ids,shift,z,r))
for node,rows in sel.items():
 best=min(rows,key=lambda r:(r[0],r[1]));d=best[0];tol=64*np.finfo(float).eps*max(4,d);ptol=max(4e-10,4*np.sqrt(2*d*tol))
 ties=[r for r in rows if abs(r[0]-d)<=tol]
 if any(np.linalg.norm(r[-1]-best[-1])>ptol for r in ties):
  print('source',node,X[node],flush=True)
  for row in ties[:8]:print('distance',row[0],'target',row[4][1:].tolist(),'offset',row[-1].tolist(),flush=True)
  break
