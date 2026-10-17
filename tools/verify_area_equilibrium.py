"""Manufactured coupled solid/contact equilibrium; no time integration claim."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from tools.verify_plane_contact import fixture
from packing_fem.explicit_material import internal_force,geometry
from packing_fem.explicit_periodic import Periodic
from packing_fem.area_contact import AreaContact
from packing_fem.barrier_contact import barrier


def check(n,P):
    c=json.loads(Path('configs/explicit-area-pressure.json').read_text());E,y,H,nu=c['young_mpa'],c['yield_mpa'],c['hardening_mpa'],c['poisson']
    elastic=brentq(lambda v:E*np.log(v)+P*v,.1,1.)
    lam=brentq(lambda v:P*v-y-H*(-np.log(v)-P*v/E),.1,1.) if P*elastic>y else elastic
    ep=max(0.,-np.log(lam)-P*lam/E);lateral=np.exp(nu*P*lam/E+.5*ep)
    gap=brentq(lambda g:barrier(g,20.,.02)[1]-P,1e-8,.02)
    X,t,_,_,upper=fixture(n,.015,0.);x=X.copy();x[:,:2]*=lateral;x[:,2]*=lam;x[upper,2]+=-.2*lam+gap
    cell=np.diag([lateral,lateral,.8*lam+gap]);b=np.diag(np.exp([2*nu*P*lam/E,2*nu*P*lam/E,-2*P*lam/E]))
    fint,volume,energy,J,_,_=internal_force(x,t,np.repeat(b[None],len(t),axis=0),np.full(len(t),ep),c)
    p=Periodic(X,1.);contact=AreaContact(X,t,p,20.,.3,.1,thickness=.02);fc,extra,_,r=contact.evaluate(x,x*0,cell,cell*0,0.)
    total=p.force(fint+fc,extra);total[p.prescribed]-=P
    V0=geometry(X,t)[0];phi=1-volume.sum()/np.linalg.det(cell)
    return dict(refinement=n,nominal_pressure_MPa=P,gap_mm=gap,material_axial_stretch=lam,peeq=ep,porosity=float(phi),analytic_porosity=gap/(.8*lam+gap),
        force_residual_relative=float(np.linalg.norm(total))/P,material_volume_relative=float(np.max(abs(volume-V0*J)/V0)),
        contact_force_relative=abs(float(fc[upper,2].sum())-P)/P)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[check(n,P) for n in (1,2,4) for P in (.02,.1)]
    with (a.output/'equilibrium.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    checks=dict(force_balance=all(r['force_residual_relative']<1e-8 for r in rows),material_volume=all(r['material_volume_relative']<1e-10 for r in rows),contact_force=all(r['contact_force_relative']<1e-8 for r in rows),porosity=all(abs(r['porosity']-r['analytic_porosity'])<1e-12 for r in rows))
    (a.output/'summary.json').write_text(json.dumps(dict(scope='manufactured_analytic_state_FEM_residual_not_explicit_solution',checks=checks),indent=2)+'\n');print(json.dumps(rows,indent=2));raise SystemExit(0 if all(checks.values()) else 2)
if __name__=='__main__':main()
