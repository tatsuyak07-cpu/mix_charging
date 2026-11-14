"""Independent material/rigid balance, geometry and Coulomb replay of accepted states."""
import argparse,csv,gc,json,hashlib
from pathlib import Path
import numpy as np
import ipctk as ipc
from scipy.spatial.distance import pdist
from packing_fem.rigid_inclusion import make_periodic,make_contact,cylinder_surface,rigid_volume,porosities
from packing_fem.explicit_periodic import deformation
from packing_fem.explicit_material import geometry,response
from packing_fem.implicit_material import evaluate
from packing_fem.friction_audit import friction_report


def load(path):
    with np.load(path) as f:return dict(f)


def predecessor(run,step,current,old,local_only=False):
    fields=['X','t','V0','parent']
    if all(np.array_equal(old[k],current[k]) for k in fields):return old,str(run/f'state-{step-1:04d}.npz')
    sources=[run]
    if not local_only and (run/'provenance.json').exists():sources.extend(Path(r['run']) for r in json.loads((run/'provenance.json').read_text())['segments'])
    for source in sources:
        for path in sorted(source.glob('remesh-*.npz')):
            d=load(path)
            if all(np.array_equal(d[k],current[k]) for k in fields) and np.max(abs(d['q'][-6:]-old['q'][-6:]))<1e-12:return d,str(path)
    raise ValueError('Missing actual post-remesh material/rigid predecessor')


def audit(run,out,physics_config=None,local_predecessors_only=False):
    run=Path(run);out=Path(out)
    if out.exists():raise FileExistsError(out)
    c=json.loads((run/'config.json').read_text())
    if physics_config is not None:
        original=json.loads(Path(physics_config).read_text())
        physical=['length_mm','diameter_mm','neck_diameter_mm','young_mpa','poisson','yield_mpa','hardening_mpa','friction','transverse_boundary','rigid_inclusion','rigid_diameter_mm','rigid_height_mm','rigid_segments','rigid_center_mm','ipc_dhat_mm','ipc_stiffness_mpa','ipc_slip_regularization_mm','implicit_force_tolerance_N','implicit_macro_tolerance']
        differences=[key for key in physical if original.get(key)!=c.get(key)]
        if differences:raise ValueError('Original physical parameters or acceptance criteria changed: '+str(differences))
        c=original
    initial=load(run/'state-0000.npz');reference=float(initial['V0'].sum());local,faces=cylinder_surface(c);distances=pdist(local);vr=rigid_volume(c);records=[]
    with (run/'history.csv').open() as f:history={int(row['step']):row for row in csv.DictReader(f)}
    p=None;contact=None;cached_t=None;mesh_builds=0
    for path in sorted(run.glob('state-*.npz')):
        step=int(path.stem.split('-')[1]);d=load(path)
        if p is None or not np.array_equal(p.X,d['X']) or not np.array_equal(cached_t,d['t']):
            contact=None;gc.collect();p=make_periodic(d['X'],c['length_mm'],c);cached_t=d['t'].copy();contact=make_contact(d['X'],d['t'],p,c);mesh_builds+=1
        H=deformation(d['q'][-6:]);volume=geometry(d['x'],d['t'])[0];cell=p.L**3*np.linalg.det(H);phi,total=porosities(volume.sum(),cell,c)
        center,(R,_,_)=p.pose(d['q']);world=local@R.T;actual_v=abs(np.einsum('ij,ij->i',world[faces[:,0]],np.cross(world[faces[:,1]],world[faces[:,2]])).sum()/6)
        J=response(d['b'],d['ep'],c)[2];material=np.bincount(d['parent'],weights=d['V0']*J,minlength=len(d['t']))
        row=dict(step=step,state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),matrix_region_porosity=phi,total_porosity=total,compression=float(-d['q'][p.prescribed]),solid_volume_mm3=float(volume.sum()),reference_volume_mm3=float(d['V0'].sum()),reference_volume_relative_error=abs(float(d['V0'].sum()/reference)-1),minimum_element_volume_mm3=float(volume.min()),quadrature_volume_error=float(np.max(abs(material-volume)/volume)),fixed_lattice_error=float(np.max(abs(H-np.diag([1,1,H[2,2]])))),saved_position_error_mm=float(abs(d['x']-p.positions(d['q'])).max()),rigid_volume_mm3=actual_v,rigid_volume_relative_error=abs(actual_v/vr-1),rigid_pair_distance_error_mm=float(abs(pdist(world)-distances).max()),rigid_center_mm=center.tolist(),rigid_rotation_matrix=R.tolist())
        passed=(volume.min()>0 and row['reference_volume_relative_error']<1e-7 and row['quadrature_volume_error']<1e-7 and row['fixed_lattice_error']<1e-12 and row['saved_position_error_mm']<1e-8 and row['rigid_volume_relative_error']<1e-10 and row['rigid_pair_distance_error_mm']<1e-9 and 0<=total<=phi<=1)
        row['surface_intersections']=bool(ipc.has_intersections(contact.mesh,contact.positions(d['q'])))
        passed=passed and not row['surface_intersections']
        if step in history:
            saved=history[step];old,origin=predecessor(run,step,d,load(run/f'state-{step-1:04d}.npz'),local_predecessors_only)
            inverse=np.linalg.inv(geometry(old['x'],old['t'])[2]);_,gm,_,st=evaluate(d['x'],d['t'],old['b'],old['ep'],inverse,old['V0'],c,parent=old['parent'])
            _,gcforce,_,count=contact.evaluate_coupled(d['q'],old['q']);g=np.asarray(p.A.T@gm.ravel()).ravel()+gcforce
            force,torque=contact.rigid_resultant(d['q']);res=float(abs(g[:p.nmicro]).max());pressure=float(-g[p.prescribed]/p.L**3);gauge=-g[:p.body_nmicro].reshape(-1,3).sum(axis=0)-g[p.rigid_dofs[:3]]
            row.update(predecessor=origin,free_residual_N=res,free_residual_l2_N=float(np.linalg.norm(g[:p.nmicro])),pressure_MPa=pressure,pressure_replay_error_MPa=abs(pressure-float(saved['nominal_pressure_MPa'])),b_replay_error=float(abs(st['b']-d['b']).max()),ep_replay_error=float(abs(st['ep']-d['ep']).max()),rigid_force_N=force.tolist(),rigid_torque_Nmm=torque.tolist(),rigid_force_max_N=float(abs(force).max()),rigid_torque_over_radius_max_N=float(abs(torque).max()/p.rigid_radius),rigid_contacts=contact.rigid_contact_count(),gauge_reaction_N=gauge.tolist(),csv_porosity_error=abs(phi-float(saved['porosity'])))
            row['coulomb']=friction_report(contact,d['q'],old['q'],c['friction'])
            passed=passed and res<c['implicit_force_tolerance_N'] and row['rigid_force_max_N']<c['implicit_force_tolerance_N'] and row['rigid_torque_over_radius_max_N']<c['implicit_force_tolerance_N'] and abs(res-float(saved['free_residual_N']))<1e-8 and row['pressure_replay_error_MPa']<1e-8 and row['b_replay_error']<1e-7 and row['ep_replay_error']<1e-7 and row['csv_porosity_error']<1e-10 and row['coulomb']['passed']
        else:row['initial_geometry_only']=True
        row['passed']=bool(passed);records.append(row);print(json.dumps({k:row[k] for k in ['step','matrix_region_porosity','passed']}),flush=True)
    del contact;gc.collect()
    report=dict(passed=bool(records and all(r['passed'] for r in records)),collision_mesh_builds=mesh_builds,local_predecessors_only=bool(local_predecessors_only),force_tolerance_N=c['implicit_force_tolerance_N'],torque_scale_radius_mm=c['rigid_diameter_mm']/2,original_physics_config=None if physics_config is None else str(physics_config),original_physics_config_sha256=None if physics_config is None else hashlib.sha256(Path(physics_config).read_bytes()).hexdigest(),initial_state_is_geometry_not_equilibrium=True,mesh_increment_convergence_verified=False,states=records)
    out.write_text(json.dumps(report,indent=2)+'\n');return report['passed']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--physics-config');p.add_argument('--local-predecessors-only',action='store_true');a=p.parse_args()
    if not audit(a.run,a.out,a.physics_config,a.local_predecessors_only):raise SystemExit(1)
