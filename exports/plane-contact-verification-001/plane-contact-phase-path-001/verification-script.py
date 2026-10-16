"""Same periodic flat interface, different triangulations and mesh phases.

Prescribed contact-only verification, not a loaded packing or material solve.
No calibration or change to the production contact law is made here.
"""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from packing_fem import sintered
from packing_fem.explicit_material import hex_to_tet
from packing_fem.explicit_periodic import Periodic,affine_basis
from packing_fem.barrier_contact import BarrierContact,barrier,constraint


def fixture(n,gap,phase):
    if n<1 or not 0<gap<.2:raise ValueError('Invalid plane verification geometry')
    mask=np.ones((n,n,3),bool);mask[:,:,1]=False
    X,h,_=sintered.mesh(mask,1/n)
    X[:,2]=np.array([0.,.4,.6,1.])[np.rint(X[:,2]*n).astype(int)]
    t=hex_to_tet(X,h);upper=X[:,2]>.5;x=X.copy()
    shift=np.array([phase/n,.37*phase/n])
    x[upper,:2]+=shift;x[upper,2]-=.2-gap
    H=np.eye(3);H[:2,2]=shift;H[2,2]=.8+gap
    return X,t,x,H,upper


def evaluate(X,t,x,H,kn,thickness,skin=.1):
    p=Periodic(X,1.);contact=BarrierContact(X,t,p,kn,.3,skin,thickness=thickness,max_candidates=2000000)
    f,extra,_,report=contact.evaluate(x,np.zeros_like(x),H,H*0,0.)
    return f,extra,report,contact,p


def measure(n,gap,phase,kn=20.,thickness=.02,skin=.1):
    start=time.perf_counter();X,t,x,H,upper=fixture(n,gap,phase)
    f,extra,r,contact,p=evaluate(X,t,x,H,kn,thickness,skin)
    # Independent area-spring reference: integrate B(gap) over exactly 1 mm².
    # This is a physical uniform-interface diagnostic, NOT the implemented sum.
    E,traction,_=barrier(gap,kn,thickness);force=float(f[upper,2].sum())
    parts={'pt':0.,'ee':0.}
    for key,ids,kind,shift,weight,threshold in contact.neighbors:
        points=x[ids].copy();points[1 if kind=='pt' else 2:]+=H@shift
        value=constraint(points,kind,weight,thickness,.3,threshold=threshold)
        if value is not None:parts[kind]+=float(value['force'][upper[ids],2].sum())
    if abs(sum(parts.values())-force)>1e-10*max(abs(force),1.):raise RuntimeError('Component force decomposition failed')
    eps=1e-7;direction=np.zeros_like(x);direction[upper,2]=1.
    def energy(xnew,Hnew):
        value=contact.evaluate(xnew,np.zeros_like(xnew),Hnew,Hnew*0,0.)[-1]
        return value['normal_energy']+value['tangent_energy']
    dH=np.zeros((3,3));dH[2,2]=1.
    fd=-(energy(x+eps*direction,H+eps*dH)-energy(x-eps*direction,H-eps*dH))/(2*eps)
    basis=affine_basis(X);generalized=p.force(f,extra);work_errors=[]
    for k in (2,4):
        B=np.zeros(6);B[k]=1.;macro=np.einsum('aij,j->ai',affine_basis(np.eye(3)),B).T
        # H = Fbar * L; L=1 mm here.
        derivative=-(energy(x+eps*basis[:,:,k],H+eps*macro)-energy(x-eps*basis[:,:,k],H-eps*macro))/(2*eps)
        work_errors.append(abs(derivative-generalized[-6+k])/max(abs(force),1e-12))
    return dict(refinement=n,phase_fraction=phase,gap_mm=gap,thickness_mm=thickness,penalty_MPa_mm=kn,skin_mm=skin,
        nodes=len(X),tetrahedra=len(t),contacts=r['contacts'],edge_contacts=r['edge_contacts'],
        point_triangle_force_N=parts['pt'],edge_edge_force_N=parts['ee'],tangent_resultant_N=float(np.linalg.norm(f[upper,:2].sum(axis=0))),
        initial_tangent_energy_Nmm=r['tangent_energy'],initial_friction_increment_Nmm=r['friction_increment'],
        force_N=force,energy_Nmm=r['normal_energy'],area_spring_force_N=traction,area_spring_energy_Nmm=E,
        force_gain=force/traction if traction else None,energy_gain=r['normal_energy']/E if E else None,
        force_balance_relative=float(np.linalg.norm(f.sum(axis=0)))/max(abs(force),1e-12),
        gap_gradient_relative=abs(fd-force-extra[2])/max(abs(force),1e-12),
        macro_work_relative=max(work_errors),elapsed_s=time.perf_counter()-start)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',type=Path,required=True);a.add_argument('--refinements',type=int,nargs='+',default=[1,2,4]);a.add_argument('--phases',type=float,nargs='+',default=[0.,.25,.5]);a.add_argument('--gaps',type=float,nargs='+',default=[.015,.01]);args=a.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);rows=[]
    (args.output/'verification-script.py').write_bytes(Path(__file__).read_bytes())
    for n in args.refinements:
        for gap in args.gaps:
            for phase in args.phases:
                row=measure(n,gap,phase);rows.append(row)
                with (args.output/'plane-contact.csv').open('w') as stream:
                    w=csv.DictWriter(stream,fieldnames=list(row));w.writeheader();w.writerows(rows)
                print(json.dumps(row),flush=True)
    groups=[]
    for n in args.refinements:
        for gap in args.gaps:
            values=[r['force_N'] for r in rows if r['refinement']==n and r['gap_mm']==gap]
            groups.append(dict(refinement=n,gap_mm=gap,phase_force_spread_relative=(max(values)-min(values))/max(np.mean(values),1e-12)))
    tolerances=dict(balance=1e-10,gradient=1e-5,macro_work=1e-5,phase_spread=.02)
    checks=dict(force_balance=all(r['force_balance_relative']<tolerances['balance'] for r in rows),
        energy_gradient=all(r['gap_gradient_relative']<tolerances['gradient'] for r in rows),
        periodic_macro_work=all(r['macro_work_relative']<tolerances['macro_work'] for r in rows),
        flat_interface_phase_invariance=all(r['phase_force_spread_relative']<tolerances['phase_spread'] for r in groups))
    report=dict(scope='contact_only_same_flat_interface_not_packing_compaction',tolerances=tolerances,checks=checks,phase_groups=groups,
        status='screening_passed_not_general_convergence' if all(checks.values()) else 'verification_failed_do_not_accept_packing_predictions',
        source_sha256=hashlib.sha256(Path('packing_fem/barrier_contact.py').read_bytes()).hexdigest(),
        settings=dict(refinements=args.refinements,phases=args.phases,gaps=args.gaps,friction=.3,skin_mm=.1,thickness_mm=.02,penalty_MPa_mm=20.))
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));raise SystemExit(0 if all(checks.values()) else 2)
if __name__=='__main__':main()
