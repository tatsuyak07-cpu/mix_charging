"""Periodic voxel FEM of spheres joined by explicit solid cylindrical necks.

This is a developer-defined trial geometry, not a simulation of sintering.
No contact ties across empty space and no frictional particle surrogate.
"""
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.ndimage import label,generate_binary_structure
from scipy.spatial import cKDTree
from .geometry import make_packing,sphere_volume


def validate(c):
    for key in ('length_mm','diameter_mm','neck_diameter_mm','mesh_mm','young_mpa','target_mpa','porosity_tolerance'):
        if c.get(key) is None or not np.isfinite(c[key]) or c[key]<=0:
            raise ValueError(f'Sintered geometry requires positive {key}')
    if c['neck_diameter_mm']>=c['diameter_mm']:
        raise ValueError('Neck must be narrower than the sphere')
    if not 0<c['porosity']<1 or not -1<c['poisson']<.5:
        raise ValueError('Invalid porosity or Poisson ratio')
    for key in ('neck_neighbors','max_geometry_iterations','max_dofs'):
        if not isinstance(c.get(key),int) or c[key]<1:
            raise ValueError(f'{key} must be a positive integer')
    if not 0<c.get('stress_tolerance',.02)<1:
        raise ValueError('stress_tolerance must be between zero and one')


def draw(mask,start,end,radius,h):
    """Rasterize the union; periodic index wrapping, never erase solid cells."""
    lo=np.floor((np.minimum(start,end)-radius)/h-.5).astype(int)
    hi=np.ceil((np.maximum(start,end)+radius)/h-.5).astype(int)
    indices=np.stack(np.meshgrid(*[np.arange(a,b+1) for a,b in zip(lo,hi)],indexing='ij'),axis=-1).reshape(-1,3)
    p=(indices+.5)*h
    vector=end-start
    if np.dot(vector,vector)==0:
        distance=np.linalg.norm(p-start,axis=1)
        hit=distance<=radius
    else:
        t=(p-start)@vector/np.dot(vector,vector)
        distance=np.linalg.norm(p-start-t[:,None]*vector,axis=1)
        hit=(distance<=radius)&(t>=0)&(t<=1)
    chosen=indices[hit]%mask.shape[0]
    mask[chosen[:,0],chosen[:,1],chosen[:,2]]=True


def voxel_geometry(c,count):
    L,d=c['length_mm'],c['diameter_mm']
    cores_porosity=1-count*sphere_volume(d)/L**3
    x=make_packing(L,d,cores_porosity,c['seed'])
    n=int(np.ceil(L/c['mesh_mm']));h=L/n
    mask=np.zeros((n,n,n),dtype=bool)
    for p in x:
        draw(mask,p,p,d/2,h)
    core_volume=float(mask.sum()*h**3)
    tree=cKDTree(x,boxsize=L)
    _,neighbors=tree.query(x,k=min(len(x),c['neck_neighbors']+1))
    bonds=set()
    if neighbors.ndim==1:neighbors=neighbors[:,None]
    for i,row in enumerate(neighbors):
        for j in row:
            if i!=j:bonds.add(tuple(sorted((i,int(j)))))
    for i,j in sorted(bonds):
        delta=x[j]-x[i];delta-=L*np.rint(delta/L)
        draw(mask,x[i],x[i]+delta,c['neck_diameter_mm']/2,h)
    return x,mask,h,core_volume,len(bonds)


def connectivity(mask):
    """Face-connected components on a torus and independent winding cycles."""
    labels,nlabels=label(mask,generate_binary_structure(3,1))
    parent=np.arange(nlabels+1);offset=np.zeros((nlabels+1,3),dtype=int)
    def find(a):
        if parent[a]!=a:
            p=parent[a];root,v=find(p);offset[a]+=v;parent[a]=root
        return int(parent[a]),offset[a].copy()
    cycles=[]
    for axis in range(3):
        first=np.take(labels,0,axis=axis).ravel();last=np.take(labels,-1,axis=axis).ravel()
        pairs=np.unique(np.stack((last,first),axis=1),axis=0)
        shift=np.eye(3,dtype=int)[axis]
        for a,b in pairs:
            if a==0 or b==0:continue
            ra,va=find(int(a));rb,vb=find(int(b))
            if ra==rb:
                cycle=vb-va-shift
                if cycle.any():cycles.append(cycle)
            else:
                parent[rb]=ra;offset[rb]=shift+va-vb
    roots={find(i)[0] for i in range(1,nlabels+1)}
    rank=int(np.linalg.matrix_rank(np.asarray(cycles,dtype=float))) if cycles else 0
    return len(roots),rank


def geometry(c):
    validate(c)
    nominal=int(np.floor((1-c['porosity'])*c['length_mm']**3/sphere_volume(c['diameter_mm'])+.5))
    if nominal<2:raise ValueError('Sintered trial requires at least two spheres')
    count=nominal;best=None;seen=set();attempts=[]
    for _ in range(c['max_geometry_iterations']):
        if count in seen:break
        seen.add(count)
        x,mask,h,core_volume,nbonds=voxel_geometry(c,count)
        phi=1-float(mask.mean());error=abs(phi-c['porosity'])
        attempts.append({'particles':count,'porosity':phi})
        if best is None or error<best[0]:best=(error,x,mask,h,core_volume,nbonds)
        if error<=c['porosity_tolerance']:break
        count=max(2,min(nominal,int(np.floor(count*(1-c['porosity'])/(1-phi)+.5))))
    error,x,mask,h,core_volume,nbonds=best
    components,rank=connectivity(mask)
    estimate={'model_type':'sintered','particles':len(x),'sphere_only_nominal_count':nominal,
              'diameter_mm':c['diameter_mm'],'neck_diameter_mm':c['neck_diameter_mm'],
              'neck_bonds':nbonds,'grid_cells_per_axis':mask.shape[0],'voxel_mm':h,
              'solid_elements':int(mask.sum()),'initial_porosity':1-float(mask.mean()),
              'target_porosity':c['porosity'],'porosity_error':error,
              'sphere_voxel_volume_mm3':core_volume,
              'added_neck_voxel_volume_mm3':float(mask.sum()*h**3-core_volume),
              'periodic_components':components,'winding_rank':rank,
              'cells_across_neck':c['neck_diameter_mm']/h,'geometry_attempts':attempts,
              'geometry_model':'Random nonoverlapping spheres + nearest-neighbor straight cylindrical neck union, voxelized',
              'mesh_convergence':'unverified'}
    return x,mask,h,estimate


def mesh(mask,h):
    occupied=np.argwhere(mask)
    corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                      [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    vertices=(occupied[:,None,:]+corners[None,:,:]).reshape(-1,3)
    integers,inverse=np.unique(vertices,axis=0,return_inverse=True)
    return integers*h,inverse.reshape(-1,8),integers


def hex_quadrature(points,elements):
    from .continuum import jacobians
    return np.linalg.det(jacobians(points,elements))


def prepare(directory,c,allow_large=False):
    directory=Path(directory)
    x,mask,h,estimate=geometry(c)
    points,elements,integers=mesh(mask,h)
    estimate.update(nodes=len(points)+6,upper_bound_dofs=3*len(points)+6)
    (directory/'estimate.json').write_text(json.dumps(estimate,indent=2)+'\n')
    np.savez_compressed(directory/'sintered-geometry.npz',centers=x,mask=mask,points=points,elements=elements)
    print(json.dumps(estimate,indent=2),flush=True)
    return points,elements,integers,estimate


def write_model(directory,c,points,elements,integers,estimate,allow_large=False):
    if estimate['porosity_error']>c['porosity_tolerance']:
        raise ValueError('Cannot match neck-inclusive porosity at this cell size/resolution; inspect estimate.json')
    if estimate['periodic_components']!=1 or estimate['winding_rank']!=3:
        raise ValueError('Skeleton is not face-connected/percolating in 3 axes; refine neck mesh or revise geometry')
    if estimate['upper_bound_dofs']>c['max_dofs'] and not allow_large:
        raise ValueError('Sintered FEM exceeds max_dofs; inspect estimate.json before --allow-large')
    n=estimate['grid_cells_per_axis'];nn=len(points)
    macro_scale=float(c.get('macro_scale',n))
    if not np.isfinite(macro_scale) or macro_scale<=0:raise ValueError('Invalid macro_scale')
    macros=np.arange(nn+1,nn+7)
    lines=['*HEADING','Sintered solid voxel FEM; mm N MPa; linear static periodic stress control','*NODE']
    lines += [f'{i},'+','.join(map(str,p)) for i,p in enumerate(points,1)]
    lines += [f'{node},0,0,0' for node in macros]
    lines += ['*ELEMENT,TYPE=C3D8,ELSET=SOLID']
    lines += [f'{i},'+','.join(map(str,e+1)) for i,e in enumerate(elements,1)]
    lines += ['*SOLID SECTION,ELSET=SOLID,MATERIAL=MAT','*MATERIAL,NAME=MAT','*ELASTIC',
              f"{c['young_mpa']},{c['poisson']}",'*NSET,NSET=PHYSICAL,GENERATE',f'1,{nn},1',
              '*NSET,NSET=MACRO',','.join(map(str,macros)),'*EQUATION']
    # Physical q=(Exx,Eyy,Ezz,Exy,Exz,Eyz)*L; macro nodes store scale*q.
    macro_matrix=np.array([[0,3,4],[3,1,5],[4,5,2]])
    representatives={}
    for node,p in enumerate(integers,1):
        key=tuple(p%n)
        if key not in representatives:
            representatives[key]=(node,p)
            continue
        rep,rp=representatives[key];shift=(p-rp)//n
        for axis in range(3):
            terms=[(node,axis+1,1),(rep,axis+1,-1)]
            for b in range(3):
                if shift[b]:terms.append((int(macros[macro_matrix[axis,b]]),1,-int(shift[b])/macro_scale))
            lines.append(str(len(terms)))
            for k in range(0,len(terms),4):
                lines.append(','.join(str(v) for term in terms[k:k+4] for v in term))
    anchor=next(iter(representatives.values()))[0]
    lines += ['*BOUNDARY',f'{anchor},1,3,0']
    lines += [f'{node},2,3,0' for node in macros]
    solver=c.get('linear_solver','SPOOLES')
    if solver not in ('SPOOLES','ITERATIVE CHOLESKY','ITERATIVE SCALING'):
        raise ValueError('Unsupported linear_solver')
    lines += ['*STEP',f'*STATIC,SOLVER={solver}','*CLOAD',f"{macros[2]},1,{-c['target_mpa']*c['length_mm']**2/macro_scale}",
              '*NODE PRINT,NSET=PHYSICAL','U','*NODE PRINT,NSET=MACRO','U',
              '*EL PRINT,ELSET=SOLID','S,E','*EL PRINT,ELSET=SOLID,TOTALS=ONLY','ELSE',
              '*END STEP']
    with (Path(directory)/'sintered.inp').open('x') as stream:stream.write('\n'.join(lines)+'\n')
    metadata={'config':c,'estimate':estimate,'macro_nodes':macros.tolist(),
              'strain_order':['xx','yy','zz','xy','xz','yz'],'model_type':'sintered',
              'macro_scale':macro_scale}
    (Path(directory)/'sintered-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


def solve(directory,solver,timeout):
    from .cli import solver_environment
    from .results import read_dat,find
    import shutil
    directory=Path(directory).resolve()
    executable=shutil.which(solver)
    if executable is None:raise ValueError('CalculiX unavailable')
    if (directory/'sintered.log').exists():raise ValueError('Refusing to overwrite archived sintered run')
    metadata=json.loads((directory/'sintered-metadata.json').read_text())
    cap=metadata['config'].get('solver_memory_gib')
    command=[executable,'-i','sintered']
    if cap is not None:
        if not np.isfinite(cap) or cap<=0:raise ValueError('solver_memory_gib must be positive')
        command=['/bin/bash','-c','ulimit -v "$1"; exec "$2" -i sintered',
                 'fem-limit',str(int(cap*1024**2)),executable]
    with (directory/'sintered.log').open('x') as stream:
        try:
            r=subprocess.run(command,cwd=directory,env=solver_environment(),
                             stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
            code=r.returncode
        except subprocess.TimeoutExpired:code=124
    log=(directory/'sintered.log').read_text(errors='replace')
    completed=code==0 and 'Job finished' in log and '*ERROR' not in log
    result={'status':'diagnostics_only','solver_exit_code':code,'solver_completed':completed,
            'mesh_convergence':'unverified'}
    if completed:
        m=json.loads((directory/'sintered-metadata.json').read_text());c=m['config']
        with np.load(directory/'sintered-geometry.npz') as g:
            points,elements=g['points'],g['elements']
        frames=list(read_dat(directory/'sintered.dat'))
        _,s=frames[-1]
        u=find(s,'displacements','PHYSICAL');u=u[np.argsort(u[:,0])][:,1:4]
        macro=find(s,'displacements','MACRO');macro=macro[np.argsort(macro[:,0])][:,1]/m.get('macro_scale',1.)
        E=np.array([[macro[0],macro[3],macro[4]],[macro[3],macro[1],macro[5]],
                    [macro[4],macro[5],macro[2]]])/c['length_mm']
        F=np.eye(3)+E
        weights=hex_quadrature(points,elements)
        stress=find(s,'stresses','SOLID');stress=stress[np.lexsort((stress[:,1],stress[:,0]))]
        if len(stress)!=len(elements)*8 or len(u)!=len(points):raise ValueError('Incomplete FEM output')
        mean=(stress[:,2:8]*weights.ravel()[:,None]).sum(axis=0)/c['length_mm']**3
        strains=find(s,'strains','SOLID')[:,-6:]
        deformed=hex_quadrature(points+u,elements)
        phi=1-deformed.sum()/(np.linalg.det(F)*c['length_mm']**3)
        tolerance=c.get('stress_tolerance',.02)*c['target_mpa']
        checks={'solver_completed':completed,'stress_target':abs(-mean[2]-c['target_mpa'])<tolerance,
                'other_stresses_zero':np.max(np.abs(mean[[0,1,3,4,5]]))<tolerance,
                'positive_jacobians':bool(np.all(deformed>0)),
                'positive_cell':bool(np.linalg.det(F)>0),'physical_porosity':0<phi<1,
                'small_material_strain':bool(np.max(np.abs(strains))<.01)}
        result.update(checks={k:bool(v) for k,v in checks.items()},initial_porosity=m['estimate']['initial_porosity'],
                      final_porosity=float(phi),macro_strain=E.tolist(),mean_stress_mpa=mean.tolist(),
                      axial_compression_strain=float(-E[2,2]))
        if all(checks.values()):result['status']='equilibrium_candidate_requires_refinement'
        import meshio
        meshio.write(directory/'sintered-deformed.vtu',meshio.Mesh(points+u,[('hexahedron',elements)],
                     point_data={'displacement_mm':u},cell_data={'stress_mpa':[stress[:,2:8].reshape(-1,8,6).mean(axis=1)]}))
        history=[]
        for scale in np.linspace(0,1,11):
            vv=hex_quadrature(points+scale*u,elements).sum()
            box=np.linalg.det(np.eye(3)+scale*E)*c['length_mm']**3
            history.append([scale*c['target_mpa'],scale*(-E[2,2]),1-vv/box])
        np.savetxt(directory/'sintered-linear-history.csv',history,delimiter=',',
                   header='compression_MPa,axial_strain,porosity',comments='')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        history=np.asarray(history)
        fig,axes=plt.subplots(1,2,figsize=(9,3.5))
        axes[0].plot(history[:,1],history[:,0])
        axes[0].set(xlabel='Axial compression strain',ylabel='Compression stress [MPa]')
        axes[1].plot(history[:,0],100*history[:,2])
        axes[1].set(xlabel='Compression stress [MPa]',ylabel='Porosity [%]')
        fig.suptitle('Sintered trial: proportional loading of linear FEM solution')
        fig.tight_layout();fig.savefig(directory/'sintered-linear-history.png',dpi=160);plt.close(fig)
    with (directory/'sintered-summary.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2),flush=True)
    return result
