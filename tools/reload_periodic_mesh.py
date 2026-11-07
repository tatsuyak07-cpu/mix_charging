"""Canonicalize Gmsh periodic coordinates; preserve the original mesh file.

Reject corrections larger than a specified numerical roundoff budget. No
material geometry, volume or physical parameters are adjusted to converge FEM.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from packing_fem.explicit_material import geometry,oriented
from packing_fem.adaptive_tet import quality
from tools.build_periodic_sintered import topology


def reload_mesh(source,out,c,cad):
    import gmsh,meshio
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    raw=meshio.read(source/'mesh.msh');before=0.
    for dim,tags,A,pairs in raw.gmsh_periodic:
        if dim!=2:continue
        A=np.asarray(A).reshape(4,4);pairs=np.asarray(pairs)
        before=max(before,float(np.max(abs(raw.points[pairs[:,0]]-(raw.points[pairs[:,1]]@A[:3,:3].T+A[:3,3])))))
    if before>1e-6:raise ValueError(f'Periodic mismatch exceeds numerical correction budget: {before}')
    gmsh.initialize();gmsh.option.setNumber('General.Terminal',0)
    try:
        gmsh.open(str(source/'mesh.msh'));gmsh.write(str(out/'mesh.msh'))
        tags,x,_=gmsh.model.mesh.getNodes();x=np.asarray(x).reshape(-1,3);lookup={int(tag):i for i,tag in enumerate(tags)}
        types,_,conn=gmsh.model.mesh.getElements(3)
        if list(types)!=[4]:raise ValueError('Expected T4 elements')
        rawt=np.asarray(conn[0]).reshape(-1,4);used=np.unique(rawt);mapping={int(tag):i for i,tag in enumerate(used)}
        X=x[[lookup[int(tag)] for tag in used]];t=oriented(X,np.array([[mapping[int(tag)] for tag in row] for row in rawt]))
        error=0.;pairs_count=0
        for _,slave in gmsh.model.getEntities(2):
            master,sn,mn,A=gmsh.model.mesh.getPeriodicNodes(2,slave)
            if master==slave or not len(sn):continue
            A=np.asarray(A).reshape(4,4);pairs_count+=1
            error=max(error,float(np.max(abs(x[[lookup[int(tag)] for tag in sn]]-(x[[lookup[int(tag)] for tag in mn]]@A[:3,:3].T+A[:3,3])))))
        L=c['length_mm'];components,rank=topology(X,t,L)
        if components!=1 or rank!=3 or error>L*1e-10:raise ValueError('Periodic mesh topology / coordinates failed')
        volume=float(geometry(X,t)[0].sum());meta=dict(particle_count=c['particle_count'],packing_seed=c['packing_seed'],cell_length_mm=L,nodes=len(X),tetrahedra=len(t),initial_porosity=1-volume/L**3,cad_porosity=cad['cad_porosity'],geometry_porosity_error=abs(1-volume/L**3-cad['cad_porosity']),minimum_quality=float(quality(X,t).min()),periodic_components=components,winding_rank=rank,periodic_node_error_mm=error,periodic_surface_pairs=pairs_count,original_periodic_error_mm=before,numerical_coordinate_budget_mm=1e-6,statistical_isotropy_verified=False,mesh_convergence_verified=False)
        np.savez_compressed(out/'mesh.npz',points=X,tets=t);(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');(out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
        shutil.copy2(source/'geometry.brep',out/'geometry.brep');return meta
    finally:gmsh.finalize()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--config',required=True);a=p.parse_args()
    c=json.loads(Path(a.config).read_text());cad=json.loads(Path(a.source,'cad-volume.json').read_text());print(json.dumps(reload_mesh(a.source,a.out,c,cad)),flush=True)
