"""Audit identical-CAD mesh refinements; geometry checks are not load convergence."""
import argparse,gc,json
from pathlib import Path
import numpy as np
import ipctk as ipc
from packing_fem.explicit_material import geometry,oriented
from packing_fem.rigid_inclusion import make_periodic,make_contact,porosities


def audit(config,meshes,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    c=json.loads(Path(config).read_text());rows=[]
    for directory in map(Path,meshes):
        metadata=json.loads((directory/'metadata.json').read_text())
        with np.load(directory/'mesh.npz') as data:X=data['points'];t=oriented(X,data['tets'])
        p=make_periodic(X,c['length_mm'],c);contact=make_contact(X,t,p,c);q=np.zeros(p.ndof)
        volume=geometry(X,t)[0];phi,total=porosities(volume.sum(),p.L**3,c)
        intersections=bool(ipc.has_intersections(contact.mesh,contact.positions(q)))
        row=dict(mesh=str(directory),nodes=len(X),tetrahedra=len(t),matrix_porosity=phi,total_porosity=total,cad_solid_volume_mm3=metadata['cad_solid_volume_mm3'],geometry_volume_relative_error=metadata['geometry_volume_relative_error'],minimum_tetrahedron_volume_mm3=float(volume.min()),periodic_node_error_mm=metadata['periodic_node_error_mm'],periodic_components=metadata['periodic_components'],winding_rank=metadata['winding_rank'],surface_intersections=intersections)
        row['passed']=bool(not intersections and volume.min()>0 and row['periodic_node_error_mm']<p.L*1e-10 and row['periodic_components']==1 and row['winding_rank']==3 and abs(phi-c['requested_matrix_porosity'])<c['initial_porosity_tolerance'])
        rows.append(row);print(json.dumps(row),flush=True);del contact;gc.collect()
    same_cad=bool(rows and np.ptp([r['cad_solid_volume_mm3'] for r in rows])<1e-7)
    errors=[r['geometry_volume_relative_error'] for r in rows]
    decreasing=bool(len(errors)>=3 and all(a>b for a,b in zip(errors,errors[1:])))
    report=dict(passed=bool(rows and all(r['passed'] for r in rows) and same_cad),same_cad_volume=same_cad,three_level_geometry_error_decreases=decreasing,loaded_mesh_convergence_verified=False,interpretation='Initial geometry and periodic connectivity only; no claim of converged compression response.',meshes=rows)
    out.write_text(json.dumps(report,indent=2)+'\n');return report['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--meshes',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if not audit(a.config,a.meshes,a.out):raise SystemExit(1)
