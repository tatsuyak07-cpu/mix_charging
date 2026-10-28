"""Verify opt-in surface stationarity screening; retain failed mapped preflight."""
import argparse,csv,hashlib,json,shutil,zipfile
from itertools import product
from pathlib import Path
import numpy as np
from packing_fem.area_contact import AreaContact,SurfaceResolutionError
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.explicit_material import oriented
from packing_fem.barrier_contact import feature
from packing_fem.material_neighborhood import segment_covered,MaterialNeighborhood
from packing_fem.surface_stationarity import SurfaceStationarity
from packing_fem.contact_audit import plain
from tools.run_barrier_compaction import fixture as pore_fixture
from tools.verify_plane_contact import fixture


def crossed(connected=False):
    X=np.array([[-1.,0,0],[1,0,0],[-1,-.1,-.1],[-1,.1,-.1],[0,-1,.1],[0,1,.1],[-.1,-1,.2],[.1,-1,.2]])+2
    t=np.array([[0,1,2,3],[4,5,6,7]])
    if connected:t=np.r_[t,[[0,2,3,7],[0,2,6,7],[0,4,6,7]]]
    t=oriented(X,t);x=X.copy();x[4:,2]-=.085
    return X,t,x


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('preflight','evidence','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan=dict(scope=__doc__,opt_in_setting='edge_stationarity_filter=true; production default remains false',descent_cosine_tolerance=1e-10,tolerance_sensitivity=[1e-12,1e-10,1e-8],acceptance=dict(corner_direction_descent=True,true_edge_guard_retained=True,plane_force_relative=1e-10,energy_gradient_relative=1e-5),mesh_convergence_claim=False)
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    pre=json.loads((a.preflight/'summary.json').read_text());old=json.loads(Path('exports/local-material-neighborhood-001/preflight/summary.json').read_text());corners=[];controls=[]
    for row in old['levels'][3:]:
        n=row['refinement'];X,t=pore_fixture(n);p=Periodic(X,4.);c=AreaContact(X,t,p,20.,.3,.1,thickness=.02);hit=row['initial']['failure_feature'];s=SurfaceStationarity(X,c.faces,p.classes,4.);ids=np.array(hit['ids']);shift=np.array(hit['shift']);r=s.classify(ids,shift,X,np.eye(3)*4)
        sensitivity=[s.classify(ids,shift,X,np.eye(3)*4,tolerance=v)['dominated'] for v in plan['tolerance_sensitivity']]
        witness=r['witness'];toward=np.array(witness['toward']);v=np.array(witness['direction']);derivative=-2*float(toward@v);fd=[]
        for h in (1e-5,1e-6):
            measured=float((np.linalg.norm(toward-h*v)**2-np.linalg.norm(toward)**2)/h)
            fd.append(dict(step_fraction=h,derivative_mm2=measured,relative_error=abs(measured-derivative)/abs(derivative)))
        corners.append(dict(refinement=n,classification=r,sensitivity=sensitivity,exact_one_sided_derivative_mm2=derivative,finite_differences=fd,feature=hit))
    for connected in (False,True):
        X,t,x=crossed(connected);p=Periodic(X,5.);c=AreaContact(X,t,p,20.,.3,.2,thickness=.02,edge_stationarity_filter=True);r=c.surface_stationarity.classify(np.array([0,1,4,5]),np.zeros(3,int),x,np.eye(3)*5)
        guard=None
        if not connected:
            try:c.evaluate(x,x*0,np.eye(3)*5,np.zeros((3,3)),0.)
            except SurfaceResolutionError as e:guard=str(e)
        controls.append(dict(face_connected_solid=connected,classification=r,full_evaluation_guard=guard,scope='specific critical pair retained; connected full evaluation not claimed' if connected else 'full evaluation must stop at edge guard'))
    planes=[];gap=.015;kn=20.;u=(gap/.02)**2;traction=-kn*gap/4*(2*(1-u)*np.log(u)-(1-u)**2/u)
    for n,phase in ((1,0.),(2,.25),(4,.5)):
        X,t,x,H,upper=fixture(n,gap,phase);values=[]
        for enabled in (False,True):
            c=AreaContact(X,t,Periodic(X,1.),kn,.3,.1,thickness=.02,edge_stationarity_filter=enabled);values.append(c.evaluate(x,x*0,H,H*0,0.))
        f,extra,_,r=values[1];eps=1e-7;direction=np.zeros_like(x);direction[upper,2]=1.;dH=np.zeros((3,3));dH[2,2]=1.
        def E(y,G):
            report=c.evaluate(y,y*0,G,G*0,0.)[-1];return report['normal_energy']+report['tangent_energy']
        derivative=-(E(x+eps*direction,H+eps*dH)-E(x-eps*direction,H-eps*dH))/(2*eps)
        force=float(f[upper,2].sum());planes.append(dict(refinement=n,phase=phase,force_N=force,analytic_uniform_force_N=float(traction),force_relative_error=abs(force-traction)/traction,force_arrays_identical=bool(np.array_equal(values[0][0],f) and np.array_equal(values[0][1],extra)),energy_identical=values[0][-1]['normal_energy']==r['normal_energy'],minimum_separation_identical=values[0][-1]['minimum_separation_mm']==r['minimum_separation_mm'],energy_gradient_relative=abs(derivative-force-extra[2])/force,screened_edge_pairs=r['nonstationary_edge_pairs']))
    mapped=[]
    with np.load(a.preflight/'shape-source/last-state.npz') as z:H=4*deformation(z['generalized_position'][-6:])
    for row in pre['levels'][3:]:
        hit=row['mapped']['failure_feature'];points=np.array(hit['points']);_,w=feature(points,hit['kind']);left,right=MaterialNeighborhood.endpoints(points,w)
        with np.load(a.preflight/f"mesh-{row['refinement']}.npz") as z:x=z['points'];t=z['tets']
        fraction=x@np.linalg.inv(H).T;endpoints=np.array([left,right])@np.linalg.inv(H).T
        lo=np.ceil(endpoints.min(0)-fraction.max(0)-1e-10).astype(int);hi=np.floor(endpoints.max(0)-fraction.min(0)+1e-10).astype(int);images=list(product(*(range(i,j+1) for i,j in zip(lo,hi))))
        vertices=np.concatenate([x[t]+H@np.array(image) for image in images])
        mapped.append(dict(refinement=row['refinement'],gap_mm=float(np.linalg.norm(left-right)),full_current_material_coverage=segment_covered(left,right,vertices),complete_image_list=[list(v) for v in images],scope='complete current-material check of the stopped segment; not a pressure solution'))
    checks=dict(corner_witness_lowers_distance=all(r['exact_one_sided_derivative_mm2']<0 and all(v['derivative_mm2']<0 and v['relative_error']<1e-5 for v in r['finite_differences']) for r in corners),tolerance_insensitivity=all(all(r['sensitivity']) for r in corners),true_edge_pairs_retained=all(r['classification']['certified'] and not r['classification']['dominated'] for r in controls),true_crossed_edge_guard_retained='Edge-only' in (controls[0]['full_evaluation_guard'] or ''),plane_force_and_energy_unchanged=all(r['force_arrays_identical'] and r['energy_identical'] and r['minimum_separation_identical'] for r in planes),plane_analytic_force=all(r['force_relative_error']<1e-10 for r in planes),plane_energy_gradient=all(r['energy_gradient_relative']<1e-5 for r in planes),all_initial_mesh_levels_zero_contact=all(r['initial']['resolved'] and r['initial']['report']['contacts']==0 and r['initial']['report']['normal_energy']==0 for r in pre['levels']),mapped_failure_retained=not pre['checks']['all_mapped_geometries_resolved'])
    result=dict(scope=__doc__,checks=checks,screening_checks_passed=all(checks.values()),production_default_enabled=False,corners=corners,critical_edge_controls=controls,periodic_planes=planes,mapped_full_material_check=mapped,refined_preflight_passed=pre['preflight_passed'],pressure_cycle_mesh_convergence_verified=False,particle_compaction_verified=False,test_result=(a.evidence/'edge-stationarity-full-tests-001.log').read_text().strip().splitlines()[-1])
    result=plain(result)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (a.output/'mesh-status.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['refinement','nodes','tetrahedra','initial_resolved','initial_contacts','screened_initial_EE','mapped_resolved','mapped_failure'])
        for row in pre['levels']:writer.writerow([row['refinement'],row['nodes'],row['tetrahedra'],row['initial']['resolved'],row['initial']['report']['contacts'],row['initial']['report']['nonstationary_edge_pairs'],row['mapped']['resolved'],row['mapped'].get('error','')])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,9));ax=fig.add_subplot(2,2,1,projection='3d');r=corners[0];points=np.array(r['feature']['points']);wit=r['classification']['witness'];q=np.array(wit['point']);v=np.array(wit['direction'])
    for edge,color in ((points[:2],'orange'),(points[2:],'red')):ax.plot(*edge.T,'o-',color=color)
    ax.quiver(*q,*(.5*v),color='green');ax.set(title='Corner pair: surface motion decreases gap',xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)')
    ax=fig.add_subplot(2,2,2,projection='3d');X,t,x=crossed()
    for edge,color in ((x[:2],'orange'),(x[4:6],'red')):ax.plot(*edge.T,'o-',color=color)
    ax.set(title='True critical crossed edges: guard retained',xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)')
    ax=fig.add_subplot(2,2,3);levels=[r['refinement'] for r in pre['levels']];ax.plot(levels,[int(r['initial']['resolved']) for r in pre['levels']],'o-',label='Initial geometry');ax.plot(levels,[int(r['mapped']['resolved']) for r in pre['levels']],'s--',label='Mapped geometry');ax.set(xlabel='Mesh refinement',yticks=[0,1],yticklabels=['Stopped','Evaluated'],ylim=(-.1,1.1));ax.legend();ax.grid(alpha=.3)
    ax=fig.add_subplot(2,2,4);ax.bar(['Corner n4','Corner n6','True edge'],[r['classification']['maximum_descent_cosine'] for r in corners]+[controls[0]['classification']['maximum_descent_cosine']]);ax.axhline(0,color='black',lw=.7);ax.set(ylabel='Maximum distance-decreasing direction cosine',title='Positive: dominated primitive pair')
    fig.suptitle('Experimental surface-stationarity screening (default OFF)\nInitial mesh gates pass; mapped geometry still FAILED');fig.tight_layout();fig.savefig(a.output/'edge-stationarity.png',dpi=160);plt.close(fig)
    shutil.copytree(a.preflight,a.output/'preflight');shutil.copytree(a.evidence,a.output/'evidence')
    for path in (Path(__file__),Path('tests/test_surface_stationarity.py')):
        target=a.output/'source'/path.resolve().relative_to(Path.cwd());target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    (a.output/'README.md').write_text('Experimental EE stationarity screening, explicitly opt-in. All initial mesh levels evaluated with zero contact; mapped n4/n6 still FAIL. No refined pressure-cycle or packing solution. See summary.json, mesh-status.csv and preflight/. Units mm, N, MPa.\n')
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*') if f.is_file()};(a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in a.output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    with zipfile.ZipFile(a.output.with_suffix('.zip')) as z:
        if z.testzip():raise ValueError('ZIP CRC failure')
    print(json.dumps(dict(checks=checks,screening_checks_passed=result['screening_checks_passed'],refined_preflight_passed=result['refined_preflight_passed']),indent=2));raise SystemExit(0 if result['screening_checks_passed'] else 2)
if __name__=='__main__':main()
