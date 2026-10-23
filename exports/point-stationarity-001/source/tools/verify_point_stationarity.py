"""Diagnose PT source versus target stationarity; falsify naive source pruning.

Prescribed geometry/contact probes only. PT diagnostics never authorize removal
of integration nodes; no material/contact force or solver gate is changed.
"""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
from packing_fem.area_contact import AreaContact,SurfaceResolutionError
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.explicit_material import geometry
from packing_fem.surface_stationarity import SurfaceStationarity
from packing_fem.contact_audit import plain
from packing_fem.adaptive_tet import boundary
from tools.verify_plane_contact import fixture


def wave(amplitude,phase):
    X,t,x,H,upper=fixture(4,.015,0.);interface=abs(X[:,2]-.6)<1e-12
    x[interface,2]+=amplitude*np.cos(2*np.pi*X[interface,0]+phase)
    p=Periodic(X,1.);c=AreaContact(X,t,p,20.,.3,.1,thickness=.02,edge_stationarity_filter=True);c.capture_audit=True
    return X,t,x,H,upper,interface,p,c


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan=dict(scope=__doc__,stopped_reference='exports/edge-stationarity-002/preflight',wave_cases=[dict(amplitude_mm=.001,phase_rad=.23),dict(amplitude_mm=.002,phase_rad=.23),dict(amplitude_mm=.002,phase_rad=.37)],symmetric_unresolved_control=dict(amplitude_mm=.002,phase_rad=0.),mean_wave_gap_mm=.015,penalty_mpa_mm=20.,friction=.3,barrier_thickness_mm=.02,geometric_cosine_tolerance=1e-10,normal_force_relative_tolerance=1e-10,force_law_changed=False,pressure_integration=False)
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');stopped=[];controls=[];node_rows=[];source=Path(plan['stopped_reference']);pre=json.loads((source/'summary.json').read_text())
    with np.load(source/'shape-source/last-state.npz') as z:H=4*deformation(z['generalized_position'][-6:])
    for row in pre['levels'][3:]:
        n=row['refinement']
        with np.load(source/f'mesh-{n}.npz') as z:X=z['reference_points'];t=z['tets'];x=z['points']
        p=Periodic(X,4.);c=AreaContact(X,t,p,20.,.3,.1,thickness=.02);s=SurfaceStationarity(X,c.faces,p.classes,4.);hit=row['mapped']['failure_feature'];r=s.classify_point(np.array(hit['ids']),np.array(hit['shift']),x,H)
        sensitivities=[s.classify_point(np.array(hit['ids']),np.array(hit['shift']),x,H,tolerance=v) for v in (1e-12,1e-10,1e-8)]
        stopped.append(dict(refinement=n,feature=hit,diagnostic=r,sensitivity=[dict(tolerance=v,source_descent=d['source']['has_descent'],target_descent=d['target']['has_descent']) for v,d in zip((1e-12,1e-10,1e-8),sensitivities)]))
        np.savez_compressed(a.output/f'stopped-mesh-{n}.npz',reference_points=X,points=x,tets=t,cell=H)
    meshes=[]
    for case_index,case in enumerate(plan['wave_cases']):
        X,t,x,H,upper,interface,p,c=wave(case['amplitude_mm'],case['phase_rad']);f,e,_,report=c.evaluate(x,x*0,H,H*0,0.);s=SurfaceStationarity(X,c.faces,p.classes,1.);normals=[];suppressed=[];errors=[];records=[]
        faces=boundary(t);faces=faces[np.all(abs(X[faces,2]-.6)<1e-12,axis=1)];area=np.zeros(len(X))
        for face in faces:
            triangle=X[face];A=np.linalg.norm(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]))/2;np.add.at(area,face,A/3)
        for record in c.trial_audit['records']:
            ids=np.array(record['support'][:4]);node=int(ids[0])
            if not interface[node]:continue
            value=s.classify_point(ids,np.array(record['images'][1],int),x,H);gap=float(x[node,2]-.4);kn=.5*20*area[node];u=(gap/.02)**2;expected=-kn*gap/4*(2*(1-u)*np.log(u)-(1-u)**2/u)
            error=abs(record['normal_force']-expected)/expected;errors.append(error);normals.append(record['normal_force'])
            source_only=value['source']['has_descent'] and not value['target']['has_descent']
            if source_only:suppressed.append(record['normal_force'])
            item=dict(case=case_index,node=node,gap_mm=gap,normal_force_N=record['normal_force'],manual_area_traction_N=float(expected),relative_error=float(error),source_only_descent=source_only,diagnostic=value);records.append(item)
            node_rows.append([case_index,node,gap,record['normal_force'],expected,value['source']['has_descent'],value['target']['has_descent'],value['candidate_exclusion_authorized']])
        second=c.evaluate(x,x*0,H,H*0,0.);micro=x-X@H.T;periodic_error=0.
        for cl in np.unique(p.classes):
            group=micro[p.classes==cl];periodic_error=max(periodic_error,float(np.max(abs(group-group[0]))))
        upper_total=float(sum(normals));discarded=float(sum(suppressed));controls.append(dict(case=case_index,**case,reference_upper_area_mm2=float(area.sum()),upper_source_nodes=len(normals),source_only_descent_nodes=len(suppressed),upper_source_normal_force_sum_N=upper_total,hypothetical_discarded_upper_source_force_N=discarded,hypothetical_discarded_fraction=discarded/upper_total,total_upper_reaction_N=float(f[upper,2].sum()),maximum_manual_force_relative_error=float(max(errors)),diagnostic_leaves_force_unchanged=bool(np.array_equal(f,second[0]) and np.array_equal(e,second[1])),minimum_tet_volume_mm3=float(geometry(x,t)[0].min()),periodic_displacement_error_mm=periodic_error,records=records))
        np.savez_compressed(a.output/f'wave-case-{case_index}.npz',reference_points=X,points=x,tets=t,cell=H,contact_force=f,generalized_image_force=e);meshes.append((X,t,x,H,interface))
    X,t,x,H,_,_,_,c=wave(.002,0.);symmetric=dict(resolved=False,scope='separate symmetric test; unresolved geometry not replaced by asymmetric results')
    try:c.evaluate(x,x*0,H,H*0,0.);symmetric['resolved']=True
    except SurfaceResolutionError as error:symmetric.update(error_type=type(error).__name__,error=str(error))
    np.savez_compressed(a.output/'symmetric-unresolved.npz',reference_points=X,points=x,tets=t,cell=H)
    checks=dict(stopped_source_and_target_distinguished=all(v['diagnostic']['source']['has_descent'] and not v['diagnostic']['target']['has_descent'] for v in stopped),no_diagnostic_authorizes_exclusion=all(not v['diagnostic']['candidate_exclusion_authorized'] for v in stopped) and all(not node['diagnostic']['candidate_exclusion_authorized'] for v in controls for node in v['records']),valid_counterexample_in_every_wave_case=all(v['source_only_descent_nodes']>0 and v['hypothetical_discarded_upper_source_force_N']>0 for v in controls),manual_area_traction=all(v['maximum_manual_force_relative_error']<1e-10 and abs(v['reference_upper_area_mm2']-1.)<1e-12 for v in controls),positive_volume_and_periodicity=all(v['minimum_tet_volume_mm3']>0 and v['periodic_displacement_error_mm']<1e-12 for v in controls),diagnostic_does_not_change_force=all(v['diagnostic_leaves_force_unchanged'] for v in controls),symmetric_limitation_retained=not symmetric['resolved'] and 'Multiple distinct closest' in symmetric.get('error',''))
    result=dict(scope=__doc__,checks=checks,diagnostic_checks_passed=all(checks.values()),stopped_candidates=stopped,wave_controls=controls,symmetric_control=symmetric,source_only_pruning_supported=False,pressure_cycle_mesh_convergence_verified=False,particle_compaction_verified=False,test_result=(a.evidence/'point-stationarity-full-tests-001.log').read_text().strip().splitlines()[-1])
    (a.output/'summary.json').write_text(json.dumps(plain(result),indent=2)+'\n')
    with (a.output/'wave-quadrature.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['case','source_node','gap_mm','normal_force_N','manual_force_N','source_descent','target_descent','exclusion_authorized']);w.writerows(node_rows)
    with (a.output/'wave-summary.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['case','amplitude_mm','phase_rad','upper_source_nodes','source_only_descent_nodes','upper_source_force_N','hypothetical_discarded_N','hypothetical_discarded_fraction'])
        for v in controls:w.writerow([v['case'],v['amplitude_mm'],v['phase_rad'],v['upper_source_nodes'],v['source_only_descent_nodes'],v['upper_source_normal_force_sum_N'],v['hypothetical_discarded_upper_source_force_N'],v['hypothetical_discarded_fraction']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,9));ax=axs[0,0]
    for v in stopped:ax.scatter(v['refinement'],v['diagnostic']['source']['maximum_descent_cosine'],marker='o',color='tab:blue');ax.scatter(v['refinement'],v['diagnostic']['target']['maximum_descent_cosine'],marker='x',color='tab:orange')
    ax.plot([],[],'o',color='tab:blue',label='Source integration node');ax.plot([],[],'x',color='tab:orange',label='Target surface');ax.set(xlabel='Stopped mesh refinement',ylabel='Maximum descent cosine',title='Stopped PT candidates: two sides differ');ax.legend(fontsize=8);ax.grid(alpha=.3)
    X,t,x,H,interface=meshes[1];ids=np.flatnonzero(interface&np.isclose(X[:,1],.5));ids=ids[np.argsort(X[ids,0])];ax=axs[0,1];ax.plot(X[ids,0],(x[ids,2]-.4)*1000,'o-',label='Valid prescribed gap');ax.set(xlabel='Reference x (mm)',ylabel='Gap (micrometres)',title='Counterexample: periodic wavy interface');ax.grid(alpha=.3)
    ax=axs[1,0];v=controls[1];node_ids=[r['node'] for r in v['records']];colors=['red' if r['source_only_descent'] else 'gray' for r in v['records']];ax.scatter([X[i,0] for i in node_ids],[r['normal_force_N'] for r in v['records']],c=colors);ax.set(xlabel='Reference x (mm)',ylabel='Positive normal quadrature force (N)',title='Red: source descent, but valid pressure point');ax.grid(alpha=.3)
    ax=axs[1,1];ax.bar([str(v['case']) for v in controls],[100*v['hypothetical_discarded_fraction'] for v in controls]);ax.set(xlabel='Wave control case',ylabel='Upper-source force discarded (%)',title='Loss if source-only pruning were applied',ylim=(0,100));ax.grid(axis='y',alpha=.3)
    fig.suptitle('PT diagnostic verification: source motion is not an exclusion rule\nNo refined pressure-cycle or particle compaction solution');fig.tight_layout();fig.savefig(a.output/'point-stationarity.png',dpi=160);plt.close(fig)
    shutil.copytree(a.evidence,a.output/'evidence')
    for path in [Path(__file__),Path('tests/test_point_stationarity.py'),Path('tools/verify_plane_contact.py'),*Path('packing_fem').glob('*.py')]:
        target=a.output/'source'/path.resolve().relative_to(Path.cwd());target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    (a.output/'README.md').write_text('PT source/target diagnostics only; no candidate exclusion authorized. Valid wavy quadrature points falsify source-only pruning. Symmetric closest-patch limitation retained. No refined pressure-cycle or packing solution. Units mm, N, MPa. See plan.json and summary.json.\n')
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*') if f.is_file()};(a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in a.output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    with zipfile.ZipFile(a.output.with_suffix('.zip')) as z:
        if z.testzip():raise ValueError('ZIP CRC failure')
    print(json.dumps(plain(dict(checks=checks,diagnostic_checks_passed=result['diagnostic_checks_passed'])),indent=2));raise SystemExit(0 if result['diagnostic_checks_passed'] else 2)
if __name__=='__main__':main()
