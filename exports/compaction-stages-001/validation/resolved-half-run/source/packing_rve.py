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


def continuum_porosity(L,diameter=10.,neck=2.):
    """Exact union volume when diagonal tubes overlap only inside their end cores.

    Exposed volume per bond = integral_0^a 2*pi*r*(d-2*sqrt(R²-r²)) dr.
    The small-neck restriction excludes intersections of adjacent exposed tubes.
    """
    R=diameter/2;a=neck/2;distance=np.sqrt(3)*L/2
    if not 0<a<R/np.sqrt(3) or L<=diameter or distance<diameter:
        raise ValueError('Union formula outside disjoint-core/small-neck domain')
    exposed=np.pi*a*a*distance-4*np.pi/3*(R**3-(R*R-a*a)**1.5)
    return float(1-(2*4*np.pi*R**3/3+8*exposed)/L**3)
