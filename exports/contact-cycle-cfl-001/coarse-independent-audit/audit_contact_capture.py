"""Audit recorded active contact reactions by independent central differences.

The recorded active set is used; this does not independently validate search,
mesh convergence or a general restart of the explicit integrator.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from packing_fem.contact_audit import reassemble,potential
from packing_fem.explicit_material import response
from packing_fem.explicit_periodic import Periodic,deformation
from tools.plastic_cycle_audit import face_internal_force


def reconstructed_spring(record):
    z=np.asarray(record['z']);edge=z[4]-z[0];edge/=np.linalg.norm(edge);aux=np.cross(edge,z[8]-z[0]);aux/=np.linalg.norm(aux);d=np.asarray(record['director']);normal=d[0]*edge+d[1]*np.cross(aux,edge)+d[2]*aux
    tangent=z[4]-z[0];tangent-=normal*(tangent@normal);tangent/=np.linalg.norm(tangent);offset=z[0]-np.asarray(record['bary'])@z[5:8]
    return np.array([tangent@offset,np.cross(normal,tangent)@offset])-record['plastic']


def audit(root,output):
    output.mkdir(parents=True,exist_ok=False);run=root/'run' if (root/'run').is_dir() else root
    data=json.loads((run/'contact-audit.json').read_text());c=json.loads((run/'config.json').read_text());r=json.loads((run/'summary.json').read_text())
    with np.load(run/'last-state.npz') as z:state={k:z[k] for k in z.files}
    if data['schema']!=1:raise ValueError('Unsupported contact audit schema')
    if c.get('mass_frame','pinned')!='pinned':raise ValueError('This endpoint audit currently verifies the pinned mass-frame case')
    L=c['length_mm'];data['reference_length_mm']=L;accepted=data['accepted'];x=state['points'];records=accepted['records'];F=deformation(state['generalized_position'][-6:]);periodic=Periodic(state['reference_points'],L,c.get('mass_frame','pinned'))
    target=np.asarray(data['solver_internal_generalized_force']);external=np.asarray(data['external_generalized_force']);force_scale=c['pressure_mpa']*L**2
    material=face_internal_force(x,state['tets'],response(state['elastic_b'],state['peeq'],c)[0]);geometry_exact=np.array_equal(np.asarray(accepted['points']),x) and np.array_equal(np.asarray(accepted['cell']),F*L)
    trace_force=np.zeros_like(x);trace_extra=np.zeros(6);trace_energy=0.;spring_error=0.;coulomb=0.;force_sum=np.zeros(3);moment_sum=np.zeros(3)
    for rec in records:
        local=np.asarray(rec['local_force']);np.add.at(trace_force,rec['support'],local);R=np.asarray(rec['images'])*L;fx,fy,fz=local.T;rx,ry,rz=R.T;trace_extra+=np.array([fx@rx,fy@ry,fz@rz,fx@ry+fy@rx,fx@rz+fz@rx,fy@rz+fz@ry]);trace_energy+=potential(np.asarray(rec['z']),rec)
        np.testing.assert_allclose(np.asarray(rec['z']),x[rec['support']]+np.asarray(rec['images'])@np.asarray(accepted['cell']).T,atol=1e-13,rtol=0.)
        spring=reconstructed_spring(rec) if rec['mu'] else np.zeros(2);spring_error=max(spring_error,float(np.max(abs(spring-rec['spring']))));coulomb=max(coulomb,float(rec['kt']*np.linalg.norm(spring)/rec['normal_force']) if rec['mu'] else 0.);force_sum+=local.sum(0);moment_sum+=np.cross(np.asarray(rec['z']),local).sum(0)
    savedforce=np.asarray(accepted['force']);savedextra=np.asarray(accepted['extra']);trace_error=float(max(np.max(abs(trace_force-savedforce)),np.max(abs(trace_extra-savedextra))/L))
    tests=[]
    for h in (L*1e-7,L*5e-8):
        fc,extra=reassemble(data,h);reduced=periodic.force(material+fc,extra);delta=reduced-target;delta[-6:]/=L;residual=reduced+external;residual[-6:]/=L
        force_error=float(max(np.max(abs(fc-savedforce)),np.max(abs(extra-savedextra))/L));norm=max(float(np.linalg.norm(savedforce)),float(np.linalg.norm(savedextra)/L),1e-12)
        tests.append(dict(step_mm=h,contact_force_max_error_N=force_error,contact_force_relative_error=force_error/norm,total_generalized_force_max_error_N=float(np.max(abs(delta))),free_static_residual_N=float(np.linalg.norm(residual)),free_static_residual_over_peak=float(np.linalg.norm(residual)/force_scale)))
    h=np.atleast_1d(np.genfromtxt(run/'history.csv',delimiter=',',names=True));energy_error=abs(trace_energy-float(h['normal_contact_Nmm'][-1]+h['tangent_contact_Nmm'][-1]));baseline=Path('exports/plastic-contact-unloading-001/connected-plastic-unload8-001/run');before=np.loadtxt(baseline/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1);after=np.loadtxt(run/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1);same_shape=before.shape==after.shape;path_bits=same_shape and before.tobytes()==after.tobytes();path_error=float(np.max(abs(before-after))) if same_shape else None
    with np.load(baseline/'last-state.npz') as z:
        state_differences={key:float(np.max(abs(state[key]-z[key]))) for key in z.files}
        state_identical=all(state[key].shape==z[key].shape and state[key].dtype==z[key].dtype and state[key].tobytes()==z[key].tobytes() for key in z.files)
    contact_scale=max(float(np.linalg.norm(savedforce)),1e-12)
    checks=dict(contact_resultant_small=float(np.linalg.norm(force_sum))<contact_scale*1e-8,contact_moment_small=float(np.linalg.norm(moment_sum))<L*contact_scale*1e-8,baseline_state_bitwise_identical=state_identical,geometry_exact=geometry_exact,record_sum_matches_saved_force=trace_error<1e-12,committed_history_preserved=len(data['committed_history'])==len(records),friction_coordinate_matches=spring_error<1e-12,constitutive_Coulomb_bound=coulomb<=c['friction']+1e-10,potential_energy_matches=energy_error<1e-11,contact_force_finite_difference=all(t['contact_force_relative_error']<2e-5 for t in tests),full_force_reassembly=all(t['total_generalized_force_max_error_N']<force_scale*1e-7 for t in tests),unloaded_equilibrium=all(t['free_static_residual_over_peak']<.02 for t in tests),original_solver_passed=all(r['checks'].values()),unchanged_full_audit=path_bits)
    summary=dict(scope='independent energy-gradient and force assembly on recorded active features, not independent collision search or general restart',checks={k:bool(v) for k,v in checks.items()},all_checks_passed=bool(all(checks.values())),records=len(records),committed_history_entries=len(data['committed_history']),saved_force_sum_error_N=trace_error,potential_energy_error_Nmm=energy_error,constitutive_Coulomb_ratio=coulomb,total_contact_resultant_N=force_sum.tolist(),total_unwrapped_contact_moment_Nmm=moment_sum.tolist(),finite_difference_tests=tests,original_static_residual_ratio=r['final_static_residual_ratio'],original_energy_relative_error=r['energy_relative_error'],baseline_state_max_absolute_differences=state_differences,full_step_baseline_max_absolute_difference=path_error,full_step_baseline_bitwise_identical=bool(path_bits),residual_compression=r['residual_compression_strain'],final_porosity=r['final_porosity'],history_is_restart_checkpoint=False,contact_cycle_CFL_sensitivity_verified=False,mesh_convergence_verified=False,particle_compaction_verified=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(output/'audit_contact_capture.py').write_bytes(Path(__file__).read_bytes());(output/'input-sha256.json').write_text(json.dumps({str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'contact-audit.json',run/'last-state.npz',run/'summary.json')},indent=2)+'\n');return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.input,a.output);print(json.dumps(r,indent=2));raise SystemExit(0 if r['all_checks_passed'] else 2)

if __name__=='__main__':main()
