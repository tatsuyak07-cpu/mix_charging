"""Periodic solid-side and known-patch friction verification; no compaction."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from packing_fem import sintered
from packing_fem.explicit_material import hex_to_tet
from packing_fem.adaptive_tet import boundary
from packing_fem.contact_geometry import closed_probe
from packing_fem.contact_probe import PeriodicSolidProbe,friction_return


def save_csv(path,header,rows):
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);periodic=[];checks={};errors=[]
    for refinement in (1,2):
        for axis in (0,1,2):
            n=refinement;mask=np.zeros((3*n,3*n,3*n),bool)
            mask[:n,n:2*n,n:2*n]=True;mask[2*n:,n:2*n,n:2*n]=True
            mask=np.moveaxis(mask,0,axis);X,h,_=sintered.mesh(mask,1/n);t=hex_to_tet(X,h)
            probe=PeriodicSolidProbe(X,t,3.,100.,.2);other=(axis+1)%3
            np.savez_compressed(a.output/f'periodic-mesh-r{n}-axis{axis}.npz',points=X,tets=t,physical_faces=probe.faces)
            for coordinate in np.linspace(-.08,.08,41):
                point=np.ones(3)*1.5;point[axis]=coordinate;point[other]=1.04
                r=probe.evaluate(point,X,np.eye(3)*3);expected=np.zeros(3);expected[other]=-4.
                fe=float(np.max(abs(r['force']-expected)));ee=abs(r['energy']-.08)
                errors.append((fe,ee,r['inside']))
                periodic.append([n,axis,coordinate,r['inside'],r['depth_mm'],r['energy'],*r['force'],fe,ee])
    checks['periodic_cut_path']=dict(samples=len(periodic),max_force_error_N=max(e[0] for e in errors),
        max_energy_error_Nmm=max(e[1] for e in errors),passed=all(e[0]<1e-10 and e[1]<1e-10 and e[2] for e in errors))
    save_csv(a.output/'periodic-path.csv',['refinement','axis','coordinate_mm','inside','depth_mm','energy_Nmm','fx_N','fy_N','fz_N','force_error_N','energy_error_Nmm'],periodic)
    # Friction on a plane while the closest triangle changes across a diagonal.
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.);faces=boundary(hex_to_tet(X,h));rows=[]
    for samples in (201,401):
        spring=np.zeros(3);normal=np.array([0.,0.,-1.]);previous=.3
        work=friction=algorithmic=0.;seen=set();residual=0.;cap=0.
        for u in np.linspace(.3,.6,samples):
            r=closed_probe(np.array([u,.43,.02]),X,faces,100.);fn=np.linalg.norm(r['force']);n=r['force']/fn
            f=friction_return(spring,normal,n,[u-previous,0,0],fn,50.,.3)
            spring=f['spring'];normal=n;previous=u;seen.add(r['face'])
            work+=f['work'];friction+=f['friction_loss'];algorithmic+=f['algorithmic_loss']
            residual=max(residual,abs(f['balance_residual']));cap=max(cap,np.linalg.norm(f['force'])-.3*fn)
            rows.append([samples,u,r['face'],*f['force'],f['energy'],friction,algorithmic,work])
        ledger=work-f['energy']-friction-algorithmic
        checks[f'friction_seam_{samples}']=dict(triangle_count=len(seen),final_force_N=float(np.linalg.norm(f['force'])),
            elastic_energy_Nmm=f['energy'],friction_loss_Nmm=friction,algorithmic_loss_Nmm=algorithmic,
            backward_euler_work_Nmm=work,analytic_continuum_work_Nmm=.1764,
            discrete_work_residual_Nmm=ledger,max_step_residual_Nmm=residual,
            passed=bool(len(seen)>1 and cap<1e-12 and abs(ledger)<1e-12 and abs(friction-.1728)<1e-10 and abs(f['energy']-.0036)<1e-10))
    checks['friction_step_refinement']=dict(passed=bool(checks['friction_seam_401']['algorithmic_loss_Nmm']<.51*checks['friction_seam_201']['algorithmic_loss_Nmm']))
    save_csv(a.output/'friction-seam.csv',['samples','coordinate_mm','triangle','fx_N','fy_N','fz_N','elastic_Nmm','friction_Nmm','algorithmic_Nmm','work_Nmm'],rows)
    # Rotate the normal around a concave edge, with prescribed tangential slip.
    normal=np.array([1.,0.,0.]);spring=np.array([0.,.001,0.]);rows=[];maxres=0.;cap=0.;minimumloss=0.
    oldtheta=0.;initialenergy=.5*50*(spring@spring);work=friction=algorithmic=0.
    for theta in np.linspace(0,np.pi/2,201):
        n=np.array([np.cos(theta),np.sin(theta),0.]);tangent=np.array([-np.sin(theta),np.cos(theta),0.])
        f=friction_return(spring,normal,n,.1*(theta-oldtheta)*tangent,10.,50.,.3)
        spring=f['spring'];normal=n;oldtheta=theta
        maxres=max(maxres,abs(f['balance_residual']));minimumloss=min(minimumloss,f['friction_loss'])
        cap=max(cap,np.linalg.norm(f['force'])-3.);work+=f['work'];friction+=f['friction_loss'];algorithmic+=f['algorithmic_loss']
        rows.append([theta,*n,*f['force'],f['energy'],friction,algorithmic,work])
    ledger=work-(f['energy']-initialenergy)-friction-algorithmic
    checks['normal_rotation_sliding']=dict(max_step_residual_Nmm=maxres,discrete_work_residual_Nmm=ledger,
        friction_loss_Nmm=friction,passed=bool(maxres<1e-12 and abs(ledger)<1e-12 and cap<1e-12 and minimumloss>-1e-12))
    save_csv(a.output/'friction-rotation.csv',['theta_rad','nx','ny','nz','fx_N','fy_N','fz_N','elastic_Nmm','friction_Nmm','algorithmic_Nmm','work_Nmm'],rows)
    summary=dict(purpose='periodic independent-probe and prescribed known-patch friction verification',
        checks=checks,all_checks_passed=all(c['passed'] for c in checks.values()),
        integrated_with_compression_solver=False,automatic_contact_patch_tracking=False,
        edge_edge_contact=False,quasistatic_compaction_validated=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    if not summary['all_checks_passed']:raise RuntimeError('Verification failed')

if __name__=='__main__':main()
