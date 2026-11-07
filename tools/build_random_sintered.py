"""Periodic CAD / T4 mesh of a seeded random sintered sphere packing."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from packing_fem.explicit_material import geometry,oriented
from packing_fem.adaptive_tet import quality
from tools.build_periodic_sintered import topology


def image_shifts(minimum,maximum,L):
    lo=np.ceil(-maximum/L).astype(int);hi=np.floor((L-minimum)/L).astype(int)
    return itertools.product(*[range(a,b+1) for a,b in zip(lo,hi)])


def build(c,packing,directory,geometry_path=None):
    import gmsh
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    (directory/'requested-config.json').write_text(json.dumps(c,indent=2)+'\n')
    L=float(packing['length_mm']);R=c['diameter_mm']/2;r=c['neck_diameter_mm']/2
    centers=packing['fractional_centers']*L;bonds=packing['bonds'];images=packing['bond_images']
    gmsh.initialize();gmsh.logger.start()
    try:
        gmsh.option.setNumber('General.Terminal',0);gmsh.model.add('random-periodic-sintered')
        occ=gmsh.model.occ;solids=[]
        if geometry_path is not None:
            occ.importShapes(str(geometry_path));occ.synchronize();clipped=gmsh.model.getEntities(3)
        else:
            for center in centers:
                for s in image_shifts(center-R,center+R,L):solids.append((3,occ.addSphere(*(center+L*np.array(s)),R)))
            for (i,j),image in zip(bonds,images):
                a=centers[i];b=centers[j]+L*image;delta=b-a
                for s in image_shifts(np.minimum(a,b)-r,np.maximum(a,b)+r,L):
                    solids.append((3,occ.addCylinder(*(a+L*np.array(s)),*delta,r)))
            print('CAD primitives',len(solids),flush=True)
            fused,_=occ.fuse(solids[:1],solids[1:]);box=(3,occ.addBox(0.,0.,0.,L,L,L))
            clipped,_=occ.intersect(fused,[box]);occ.synchronize()
        cad_volume=sum(occ.getMass(3,tag) for dim,tag in clipped if dim==3)
        print('CAD volume',cad_volume,'porosity',1-cad_volume/L**3,'expected',c['requested_initial_porosity'],flush=True)
        (directory/'cad-volume.json').write_text(json.dumps(dict(cad_volume_mm3=cad_volume,cad_porosity=1-cad_volume/L**3,requested=c['requested_initial_porosity']),indent=2))
        if abs(1-cad_volume/L**3-c['requested_initial_porosity'])>c.get('cad_porosity_tolerance',1e-7):raise ValueError('CAD union differs from the independently integrated neck/sphere volume')
        gmsh.write(str(directory/'geometry.brep'))
        surfaces=gmsh.model.getEntities(2);paired=[];tol=L*1e-6
        for axis in range(3):
            lower=[];upper=[]
            for _,tag in surfaces:
                box=np.array(gmsh.model.getBoundingBox(2,tag))
                if max(abs(box[axis]),abs(box[axis+3]))<tol:lower.append((tag,box))
                if max(abs(box[axis]-L),abs(box[axis+3]-L))<tol:upper.append((tag,box))
            if len(lower)!=len(upper) or not lower:raise ValueError('Unmatched periodic CAD faces')
            shift=np.zeros(6);shift[axis]=shift[axis+3]=L
            for slave,box in upper:
                matches=[master for master,other in lower if np.max(abs(box-shift-other))<tol]
                if len(matches)!=1:raise ValueError('Ambiguous periodic face pairing')
                transform=np.eye(4);transform[axis,3]=L
                gmsh.model.mesh.setPeriodic(2,[slave],matches,transform.ravel().tolist())
                paired.append((slave,matches[0],axis))
        h=c.get('mesh_size_mm',.6);hn=c.get('neck_mesh_mm',.3)
        gmsh.option.setNumber('Mesh.MeshSizeMin',hn*.5);gmsh.option.setNumber('Mesh.MeshSizeMax',h)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',c.get('curvature_points',24))
        cylinders=[tag for _,tag in surfaces if gmsh.model.getType(2,tag)=='Cylinder']
        if not cylinders:raise ValueError('No neck cylinder surfaces in CAD')
        distance=gmsh.model.mesh.field.add('Distance');gmsh.model.mesh.field.setNumbers(distance,'FacesList',cylinders)
        gmsh.model.mesh.field.setNumber(distance,'Sampling',100)
        threshold=gmsh.model.mesh.field.add('Threshold')
        for name,value in dict(InField=distance,SizeMin=hn,SizeMax=h,DistMin=.3*r,DistMax=1.5*r).items():gmsh.model.mesh.field.setNumber(threshold,name,value)
        background=threshold
        if 'surface_mesh_mm' in c:
            # Accurate curved boundary, coarser grain interiors: no volume
            # correction is applied to the faceted mesh.
            gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
            gmsh.option.setNumber('Mesh.MeshSizeFromPoints',0)
            skin=gmsh.model.mesh.field.add('Distance');gmsh.model.mesh.field.setNumbers(skin,'FacesList',[tag for _,tag in surfaces])
            gmsh.model.mesh.field.setNumber(skin,'Sampling',100)
            shell=gmsh.model.mesh.field.add('Threshold')
            for name,value in dict(InField=skin,SizeMin=c['surface_mesh_mm'],SizeMax=h,DistMin=0.,DistMax=1.5).items():gmsh.model.mesh.field.setNumber(shell,name,value)
            background=gmsh.model.mesh.field.add('Min');gmsh.model.mesh.field.setNumbers(background,'FieldsList',[threshold,shell])
        gmsh.model.mesh.field.setAsBackgroundMesh(background)
        if 'surface_mesh_mm' in c:
            hs=c['surface_mesh_mm']
            gmsh.model.mesh.setSizeCallback(lambda dim,tag,x,y,z,lc:min(lc,hs) if dim<=2 else lc)
        gmsh.model.mesh.generate(3);gmsh.model.mesh.optimize('Netgen')
        tags0,x0,_=gmsh.model.mesh.getNodes();x0=np.asarray(x0).reshape(-1,3);index0={int(tag):i for i,tag in enumerate(tags0)};before=0.
        for slave,master,axis in paired:
            _,sn,mn,A=gmsh.model.mesh.getPeriodicNodes(2,slave);A=np.asarray(A).reshape(4,4)
            before=max(before,float(np.max(abs(x0[[index0[int(tag)] for tag in sn]]-(x0[[index0[int(tag)] for tag in mn]]@A[:3,:3].T+A[:3,3])))))
        if before>1e-6:raise ValueError('Periodic coordinate mismatch exceeds numerical correction budget')
        gmsh.write(str(directory/'mesh-before-periodic-reload.msh'))
        # Gmsh reconstructs slave coordinates from its affine constraints on read.
        if 'surface_mesh_mm' in c:gmsh.model.mesh.removeSizeCallback()
        gmsh.clear();gmsh.open(str(directory/'mesh-before-periodic-reload.msh'))
        gmsh.write(str(directory/'mesh.msh'))
        tags,coordinates,_=gmsh.model.mesh.getNodes();coordinates=np.asarray(coordinates).reshape(-1,3)
        types,_,nodes=gmsh.model.mesh.getElements(3)
        if list(types)!=[4]:raise ValueError('Only linear tetrahedra are supported by this backend')
        raw=np.asarray(nodes[0]).reshape(-1,4);used=np.unique(raw);lookup={int(tag):i for i,tag in enumerate(tags)}
        X=coordinates[[lookup[int(tag)] for tag in used]];indices={int(tag):i for i,tag in enumerate(used)}
        t=oriented(X,np.array([[indices[int(tag)] for tag in row] for row in raw],int))
        periodic_error=0.
        for slave,master,axis in paired:
            found,sn,mn,_=gmsh.model.mesh.getPeriodicNodes(2,slave)
            if found!=master or not len(sn):raise ValueError('No periodic mesh-node correspondence')
            shift=np.zeros(3);shift[axis]=L
            periodic_error=max(periodic_error,max(float(np.max(abs(coordinates[lookup[int(s)]]-coordinates[lookup[int(m)]]-shift))) for s,m in zip(sn,mn)))
        if periodic_error>L*1e-10:raise ValueError('Periodic node tolerance exceeded')
        volume=float(geometry(X,t)[0].sum());components,rank=topology(X,t,L)
        metadata=dict(mesh_kind='random_shape_conforming_cad',cell_length_mm=L,particle_count=len(centers),
                      packing_seed=c['packing_seed'],nodes=len(X),tetrahedra=len(t),initial_porosity=1-volume/L**3,
                      cad_porosity=1-cad_volume/L**3,mesh_solid_volume_mm3=volume,cad_solid_volume_mm3=cad_volume,
                      geometry_volume_relative_error=abs(volume-cad_volume)/cad_volume,original_periodic_error_mm=before,numerical_coordinate_budget_mm=1e-6,periodic_components=components,
                      winding_rank=rank,periodic_node_error_mm=periodic_error,periodic_surface_pairs=len(paired),
                      minimum_quality=float(quality(X,t).min()),cells_across_neck=c['neck_diameter_mm']/hn,
                      nominal_neck_mesh_mm=hn,mesh_size_mm=h,diameter_mm=c['diameter_mm'],neck_diameter_mm=c['neck_diameter_mm'],
                      statistical_isotropy_verified=False,mesh_convergence_verified=False)
        if components!=1 or rank!=3:raise ValueError('FEM mesh lost periodic connectivity')
        np.savez_compressed(directory/'mesh.npz',points=X,tets=t)
        (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        (directory/'config.json').write_text(json.dumps(c,indent=2)+'\n')
        return metadata
    finally:
        (directory/'gmsh.log').write_text('\n'.join(gmsh.logger.get())+'\n');gmsh.logger.stop();gmsh.finalize()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--packing',required=True);p.add_argument('--out',required=True);p.add_argument('--geometry');a=p.parse_args()
    c=json.loads(Path(a.config).read_text());packing=dict(np.load(a.packing));print(json.dumps(build(c,packing,a.out,a.geometry)),flush=True)
