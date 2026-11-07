"""Seeded periodic hard-sphere geometry generation, not a compression solver.

A soft overlap objective is used only to place centers. Stress / deformation
are subsequently computed by solid FEM. Units: mm. No lattice is initialized.
"""
from itertools import combinations
import numpy as np
from scipy.optimize import minimize,brentq


def edges_and_distances(f,L):
    pairs=np.array(list(combinations(range(len(f)),2)),int)
    delta=f[pairs[:,1]]-f[pairs[:,0]];shift=-np.rint(delta).astype(int)
    vectors=(delta+shift)*L
    return pairs,shift,vectors,np.linalg.norm(vectors,axis=1)


def graph_audit(n,bonds,shifts):
    neighbors=[[] for _ in range(n)]
    for (i,j),s in zip(bonds,shifts):neighbors[i].append((j,s));neighbors[j].append((i,-s))
    offsets={};cycles=[];components=0
    for root in range(n):
        if root in offsets:continue
        components+=1;offsets[root]=np.zeros(3,int);queue=[root]
        while queue:
            i=queue.pop()
            for j,s in neighbors[i]:
                proposed=offsets[i]+s
                if j not in offsets:offsets[j]=proposed;queue.append(j)
                elif np.any(proposed-offsets[j]):cycles.append(proposed-offsets[j])
    rank=int(np.linalg.matrix_rank(cycles)) if cycles else 0
    return components,rank,np.array([len(row) for row in neighbors])


def neck_added_volume(distance,R,r):
    # Integrate the part of a center-to-center cylinder outside its end spheres.
    return np.pi*r*r*np.asarray(distance)-(4*np.pi/3)*(R**3-(R*R-r*r)**1.5)


def generate(c,attempt_directory=None):
    n=c['particle_count'];D=c['diameter_mm'];R=D/2;r=c['neck_diameter_mm']/2
    if n<16 or not 0<r<R:raise ValueError('Require >=16 spheres and a neck narrower than the sphere')
    rng=np.random.default_rng(c['packing_seed']);L=(n*(4*np.pi*R**3/3)/.6)**(1/3)
    initial=rng.random((n,3));f=initial.copy();pairs=np.array(list(combinations(range(n),2)))
    log=[]
    for diameter in np.linspace(.5*D,D,21):
        def objective(flat):
            points=flat.reshape(-1,3);delta=points[pairs[:,1]]-points[pairs[:,0]];delta-=np.rint(delta);v=delta*L
            dist=np.linalg.norm(v,axis=1);overlap=np.maximum(diameter-dist,0)
            local=-(overlap/np.maximum(dist,1e-30))[:,None]*v*L
            g=np.zeros_like(points);np.add.at(g,pairs[:,1],local);np.add.at(g,pairs[:,0],-local)
            return .5*float(overlap@overlap),g.ravel()
        result=minimize(objective,f.ravel(),jac=True,method='L-BFGS-B',options=dict(maxiter=10000,gtol=1e-9,ftol=1e-15,maxls=50))
        f=np.mod(result.x.reshape(-1,3),1.)
        log.append(dict(diameter_mm=float(diameter),objective=float(result.fun),iterations=int(result.nit),optimizer_success=bool(result.success)))
    pairs,shifts,vectors,dist=edges_and_distances(f,L)
    if attempt_directory is not None:
        from pathlib import Path
        import json
        directory=Path(attempt_directory);directory.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(directory/'overlap-relaxation.npz',initial=initial,fractional_centers=f,length_mm=L)
        (directory/'optimizer.json').write_text(json.dumps(log,indent=2))
    maximum_overlap=max(0.,float(D-dist.min()))
    if maximum_overlap>1e-6:raise ValueError(f'Packing overlap {maximum_overlap} mm; seed {c["packing_seed"]}')
    # Tiny clearance avoids OCC tangency ambiguity; the CAD porosity calibration
    # below expands further and never shrinks overlapping spheres into place.
    L*=max(1.,D/dist.min())*(1+1e-8)
    pairs,shifts,vectors,dist=edges_and_distances(f,L)
    selected=dist<=D+c['neck_selection_gap_mm'];bonds=pairs[selected];images=shifts[selected]
    components,rank,coordination=graph_audit(n,bonds,images)
    if components!=1 or rank!=3:raise ValueError(f'Non-spanning sinter graph: {components} components, winding rank {rank}')
    lengths=np.linalg.norm((f[bonds[:,1]]+images-f[bonds[:,0]]),axis=1)
    Vs=n*4*np.pi*R**3/3;solid=lambda length:Vs+neck_added_volume(length*lengths,R,r).sum()
    residual=lambda length:(1-c['requested_initial_porosity'])*length**3-solid(length)
    if residual(L)>0:raise ValueError('Calibration would require shrinking the nonoverlapping packing')
    L=brentq(residual,L,L*1.02,xtol=1e-12)
    pairs,shifts,vectors,dist=edges_and_distances(f,L)
    gaps=L*lengths-D
    if gaps.max()>c['maximum_neck_gap_mm']:raise ValueError('Calibration exceeds the specified near-neighbor neck gap')
    if L<=2*(D+c['maximum_neck_gap_mm']):raise ValueError('Cell too small for unique nearest-image bonds')
    directions=(f[bonds[:,1]]+images-f[bonds[:,0]])/lengths[:,None]
    fabric=directions.T@directions/len(bonds)
    return dict(fractional_centers=f,initial_fractional_centers=initial,bonds=bonds,bond_images=images,
                length_mm=float(L),metadata=dict(particle_count=n,packing_seed=c['packing_seed'],
                generation='uniform random centers + periodic overlap minimization; no lattice initialization',
                minimum_center_distance_mm=float(dist.min()),maximum_overlap_mm=max(0.,float(D-dist.min())),
                neck_count=len(bonds),maximum_neck_gap_mm=float(gaps.max()),minimum_coordination=int(coordination.min()),
                mean_coordination=float(coordination.mean()),periodic_components=components,winding_rank=rank,
                analytic_porosity=float(1-solid(L)/L**3),fabric_eigenvalues=np.linalg.eigvalsh(fabric).tolist(),
                statistical_isotropy_verified=False,packing_optimization_log=log))
