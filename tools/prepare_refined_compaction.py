"""Prepare a reproducible T4 input refinement without changing its surface shape."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from packing_fem.uniform_tet import refine
from packing_fem.explicit_material import geometry,oriented
from packing_fem.adaptive_tet import boundary,quality
from packing_fem.explicit_periodic import Periodic


def area(X,t):
    f=boundary(t);return float(np.linalg.norm(np.cross(X[f[:,1]]-X[f[:,0]],X[f[:,2]]-X[f[:,0]]),axis=1).sum()/2)


def prepare(mesh,config,out,levels=1):
    if not isinstance(levels,int) or levels<1:raise ValueError('Refinement levels must be positive integers')
    out=Path(out);out.mkdir(exist_ok=False,parents=True);c=json.loads(Path(config).read_text())
    with np.load(mesh) as f:X=f['points'];t=oriented(X,f['tets'])
    initial=dict(nodes=len(X),tetrahedra=len(t),volume_mm3=float(geometry(X,t)[0].sum()),boundary_area_mm2=area(X,t))
    for _ in range(levels):X,t,_=refine(X,t)
    volume=float(geometry(X,t)[0].sum());surface=area(X,t);p=Periodic(X,c['length_mm'])
    if abs(volume/initial['volume_mm3']-1)>1e-12 or abs(surface/initial['boundary_area_mm2']-1)>1e-12:
        raise ValueError('Refinement changed volume or faceted surface area')
    np.savez_compressed(out/'mesh.npz',points=X,tets=t)
    digest=hashlib.sha256((out/'mesh.npz').read_bytes()).hexdigest()
    report=dict(operation='uniform_conforming_eight_way_T4_refinement',levels=levels,initial=initial,
        nodes=len(X),tetrahedra=len(t),volume_mm3=volume,boundary_area_mm2=surface,
        minimum_quality=float(quality(X,t).min()),periodic_classes=int(p.classes.max()+1),
        source_mesh_sha256=hashlib.sha256(Path(mesh).read_bytes()).hexdigest(),prepared_mesh_sha256=digest,
        geometry_unchanged=True,mesh_convergence_verified=False)
    c.update(required_mesh_sha256=digest,require_prepared_mesh=True,implicit_require_stress_free_initial_contact=True,
        usage_scope='Prepared same-faceted-surface refinement for finite-strain solid FEM. Contact activation requires independent sensitivity; this refinement alone does not certify mesh convergence.')
    (out/'config.json').write_text(json.dumps(c,indent=2)+'\n');(out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mesh',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--levels',type=int,default=1)
    a=p.parse_args();prepare(a.mesh,a.config,a.out,a.levels)
