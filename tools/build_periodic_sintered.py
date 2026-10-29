"""Shape-conforming linear tetrahedra of the SAME periodic BCC sphere/tube union.

Meshing only: no installation, constitutive changes or correction of measured
solid volume. CAD and faceted-mesh volumes are reported independently.
"""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from packing_fem.packing_rve import continuum_porosity
from packing_fem.explicit_material import geometry,oriented
from packing_fem.adaptive_tet import FACES,quality
from packing_fem.explicit_periodic import Periodic


def topology(X,t,L):
    p=Periodic(X,L);faces=t[:,FACES].reshape(-1,3)
    _,labels,count=np.unique(np.sort(p.classes[faces],axis=1),axis=0,return_inverse=True,return_counts=True)
    if np.any(count>2):raise ValueError('Nonmanifold periodic face matching')
    pairs={};neighbors=[[] for _ in t]
    for i,label in enumerate(labels):
        if count[label]==2:pairs.setdefault(int(label),[]).append(i)
    for a,b in pairs.values():
        u,v=a//4,b//4;delta=(X[faces[b]].mean(0)-X[faces[a]].mean(0))/L
        shift=np.rint(delta).astype(int)
        if np.max(abs(delta-shift))>1e-9:raise ValueError('Periodic face has an inconsistent lattice shift')
        neighbors[u].append((v,shift));neighbors[v].append((u,-shift))
    offsets={};cycles=[];components=0
    for root in range(len(t)):
        if root in offsets:continue
        components+=1;offsets[root]=np.zeros(3,int);queue=[root]
        while queue:
            u=queue.pop()
            for v,shift in neighbors[u]:
                proposed=offsets[u]+shift
                if v not in offsets:offsets[v]=proposed;queue.append(v)
                else:
                    winding=proposed-offsets[v]
                    if np.any(winding):cycles.append(winding)
    return components,int(np.linalg.matrix_rank(np.array(cycles,float))) if cycles else 0


def build(c,directory):
    import gmsh
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    L=c['length_mm'];R=c['diameter_mm']/2;r=c['neck_diameter_mm']/2
    expected=continuum_porosity(L,c['diameter_mm'],c['neck_diameter_mm'])
    gmsh.initialize();gmsh.logger.start()
    try:
        gmsh.option.setNumber('General.Terminal',0);gmsh.model.add('periodic-sintered-bcc')
        occ=gmsh.model.occ;center=np.full(3,L/2);corners=np.array(list(itertools.product((0.,L),repeat=3)))
        solids=[(3,occ.addSphere(*point,R)) for point in np.r_[center[None],corners]]
        for corner in corners:
            delta=corner-center;solids.append((3,occ.addCylinder(*center,*delta,r)))
        fused,_=occ.fuse(solids[:1],solids[1:]);box=(3,occ.addBox(0.,0.,0.,L,L,L))
        clipped,_=occ.intersect(fused,[box]);occ.synchronize()
        if len([v for v in clipped if v[0]==3])!=1:raise ValueError('Sintered CAD is not one connected volume')
        cad_volume=sum(occ.getMass(3,tag) for dim,tag in clipped if dim==3)
        if abs((1-cad_volume/L**3)-expected)>1e-9:raise ValueError('CAD union volume differs from the independent analytic union')
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
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.model.mesh.generate(3);gmsh.model.mesh.optimize('Netgen')
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
        metadata=dict(mesh_kind='shape_conforming_cad',cell_length_mm=L,nodes=len(X),tetrahedra=len(t),
                      initial_porosity=1-volume/L**3,analytic_porosity=expected,cad_porosity=1-cad_volume/L**3,
                      mesh_solid_volume_mm3=volume,cad_solid_volume_mm3=cad_volume,
                      geometry_volume_relative_error=abs(volume-cad_volume)/cad_volume,
                      periodic_components=components,winding_rank=rank,periodic_node_error_mm=periodic_error,
                      periodic_surface_pairs=len(paired),minimum_quality=float(quality(X,t).min()),
                      cells_across_neck=c['neck_diameter_mm']/hn,nominal_neck_mesh_mm=hn,mesh_size_mm=h,
                      diameter_mm=c['diameter_mm'],neck_diameter_mm=c['neck_diameter_mm'],gmsh_version=gmsh.__version__,
                      statistical_isotropy_verified=False,mesh_convergence_verified=False)
        np.savez_compressed(directory/'mesh.npz',points=X,tets=t)
        (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        return X,t,np.empty((0,0,0),bool),metadata
    finally:
        (directory/'gmsh.log').write_text('\n'.join(gmsh.logger.get())+'\n')
        gmsh.logger.stop();gmsh.finalize()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();_,_,_,m=build(json.loads(a.config.read_text()),a.output);print(json.dumps(m),flush=True)


if __name__=='__main__':main()
