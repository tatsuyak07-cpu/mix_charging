import numpy as np,json
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from packing_fem.sintered import mesh
from packing_fem.explicit_material import geometry,hex_to_tet
from packing_fem.explicit_periodic import deformation
root=Path('results/extension-probe-001');root.mkdir(exist_ok=False);rows=[]
source=Path('exports/edge-stationarity-002/preflight')
with np.load(source/'shape-source/last-state.npz') as z:H=4*deformation(z['generalized_position'][-6:])
for n in (1,2,3,4,6):
 with np.load(source/f'mesh-{n}.npz') as z:X=z['reference_points'];x=z['points'];t=z['tets']
 axes=[np.unique(X[:,d]) for d in range(3)];_,hexes,ijk=mesh(np.ones(tuple(len(v)-1 for v in axes),bool),1.)
 Y=np.column_stack([axes[d][ijk[:,d]] for d in range(3)]);T=hex_to_tet(Y,hexes);ids=np.ravel_multi_index(tuple(np.searchsorted(axes[d],X[:,d]) for d in range(3)),tuple(len(v) for v in axes));known=np.zeros(len(Y),bool);known[ids]=True;ghost=np.flatnonzero(~known)
 A=H/4;u=np.zeros_like(Y);u[ids]=x-X@A.T
 V,g,D=geometry(Y,T);K=np.einsum('n,nai,nbi->nab',V,g,g);I=np.repeat(T,4,axis=1).ravel();J=np.tile(T,(1,4)).ravel();K=coo_matrix((K.ravel(),(I,J)),shape=(len(Y),len(Y))).tocsr()
 if len(ghost):u[ghost]=spsolve(K[ghost][:,ghost],-(K[ghost][:,ids]@u[ids]))
 y=Y@A.T+u;F=(y[T][:,1:]-y[T][:,:1]).swapaxes(1,2)@np.linalg.inv(D);G=np.linalg.inv(A)@F;values=np.linalg.eigvalsh((G+G.swapaxes(1,2))/2)
 row=dict(n=n,ghost_nodes=len(ghost),margin=float(values.min()),minJ=float(np.linalg.det(F).min()));rows.append(row);print(row,flush=True)
(root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
