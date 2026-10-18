"""Independently detect the initially unstiffened pore-closing mode.

Rigidly translate one slab and change the periodic cell height. Before the
barrier is reached, all solid strains and contact forces must remain zero.
This is a kinematic/static diagnostic, not a time-integrated solution.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.explicit_material import geometry,advance,internal_force
from packing_fem.area_contact import AreaContact
from packing_fem.macro_loading import uniaxial_load
from tools.verify_plane_contact import fixture


def check(closure,c):
    X,t,_,_,upper=fixture(1,.015,0.);x=X.copy();x[upper,2]-=closure
    periodic=Periodic(X,1.);q=np.linalg.lstsq(periodic.A.toarray(),(x-X).ravel(),rcond=None)[0];F=deformation(q[-6:])
    D=geometry(X,t)[2];increment=geometry(x,t)[2]@np.linalg.inv(D)
    b,ep=advance(np.repeat(np.eye(3)[None],len(t),axis=0),np.zeros(len(t)),increment,c)
    fi,_,energy,_,_,_=internal_force(x,t,b,ep,c)
    contact=AreaContact(X,t,periodic,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],thickness=c['barrier_thickness_mm'])
    fc,extra,_,r=contact.evaluate(x,x*0,F,F*0,0.)
    force=periodic.force(fi+fc,extra);load=uniaxial_load(F,c['pressure_mpa'],1.)
    direction=np.linalg.lstsq(periodic.A.toarray(),np.where(upper[:,None],np.array([0.,0.,-1.]),0.).ravel(),rcond=None)[0]
    return dict(closure_mm=closure,gap_mm=.2-closure,periodic_representation_error_mm=float(np.max(abs(periodic.positions(q)-x))),cell_F_error=float(np.max(abs(F-np.diag([1.,1.,1-closure])))),
                solid_deformation_error=float(np.max(abs(increment-np.eye(3)))),internal_energy_Nmm=float(geometry(X,t)[0]@energy),contact_energy_Nmm=r['normal_energy']+r['tangent_energy'],contacts=r['contacts'],
                generalized_internal_force_norm_Nmm=float(np.linalg.norm(force)),constant_pressure_virtual_work_Nmm=float(load@q[-6:]),external_closing_direction_force_N=float(load@direction[-6:]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads(Path('configs/explicit-area-cauchy-pressure.json').read_text());c.update(length_mm=1.,pressure_mpa=.02)
    rows=[check(d,c) for d in (0.,.05,.1,.17)]
    checks=dict(periodic_kinematics=all(r['periodic_representation_error_mm']<1e-12 and r['cell_F_error']<1e-12 for r in rows),rigid_solid_motion=all(r['solid_deformation_error']<1e-12 for r in rows),zero_internal_energy=all(abs(r['internal_energy_Nmm'])+abs(r['contact_energy_Nmm'])<1e-20 for r in rows),zero_resisting_force=all(r['generalized_internal_force_norm_Nmm']<1e-10 for r in rows),positive_external_closing_force=all(abs(r['external_closing_direction_force_N']-.02)<1e-12 for r in rows))
    with (a.output/'mechanism.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.output/'summary.json').write_text(json.dumps(dict(scope='kinematic_zero_stiffness_mechanism_not_a_transient_solution',checks=checks,conclusion='No static balance at positive axial pressure while gap exceeds barrier thickness: the unbonded layered fixture has a pore-closing mechanism. This does not establish instability of a connected sintered particle network.'),indent=2)+'\n')
    (a.output/'verification-script.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(rows=rows,checks=checks),indent=2));raise SystemExit(0 if all(checks.values()) else 2)
if __name__=='__main__':main()
