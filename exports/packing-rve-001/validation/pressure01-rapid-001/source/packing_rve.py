"""Small BCC sintered-sphere RVE for development, not a random packing RVE.

Continuous geometry is two periodic 10 mm spheres joined to all eight BCC
neighbors by 2 mm cylinders. A fixed cell and voxel phase are held across mesh
levels. Actual discretized porosity and neck resolution are always reported.
"""
from itertools import product
import numpy as np
from .sintered import draw,mesh,connectivity
from .explicit_material import hex_to_tet,geometry


def build(n=16,L=12.1,diameter=10.,neck=2.,phase=(0.,0.,0.)):
    if not isinstance(n,int) or n<4:raise ValueError('At least four grid cells required')
    if not np.isfinite(L) or not 0<neck<diameter<L:raise ValueError('Require positive neck < diameter < cell length')
    if np.sqrt(3)*L/2<diameter:raise ValueError('BCC sphere cores overlap')
    phase=np.asarray(phase,float)
    if phase.shape!=(3,) or not np.all(np.isfinite(phase)):raise ValueError('Finite three-component phase required')
    h=L/n;mask=np.zeros((n,n,n),bool);centers=np.array([[0.,0.,0.],[.5,.5,.5]])*L+phase
    for center in centers:draw(mask,center,center,diameter/2,h)
    core_volume=float(mask.sum()*h**3)
    for end in product((-.5,.5),repeat=3):draw(mask,phase,phase+L*np.array(end),neck/2,h)
    X,hexes,ijk=mesh(mask,h);t=hex_to_tet(X,hexes);components,rank=connectivity(mask)
    metadata=dict(scope=__doc__,cell_length_mm=L,diameter_mm=diameter,neck_diameter_mm=neck,centers=(centers%L).tolist(),periodic_bonds=8,grid_n=n,voxel_mm=h,cells_across_neck=neck/h,nodes=len(X),tetrahedra=len(t),initial_porosity=float(1-mask.mean()),mesh_solid_volume_mm3=float(geometry(X,t)[0].sum()),sphere_voxel_volume_mm3=core_volume,added_neck_volume_mm3=float(mask.sum()*h**3-core_volume),periodic_components=components,winding_rank=rank,phase_mm=phase.tolist(),statistical_isotropy_verified=False,mesh_convergence_verified=False,requested_100mm_model=False)
    return X,t,mask,metadata
