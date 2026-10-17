"""Pressure integration on a periodic layered solid with an analytic endpoint.

The reference cube has solid volume 0.8 mm3 and a 0.2 mm plane pore.
This is a 1 mm verification cell, not the requested particle packing.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from tools.verify_plane_contact import fixture
from packing_fem.explicit_solver import solve
from packing_fem.barrier_contact import barrier


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--refinement',type=int,default=1);p.add_argument('--cfl',type=float,default=.2);p.add_argument('--pressure',type=float,default=.1);p.add_argument('--ramp-seconds',type=float,default=.016);a=p.parse_args()
    X,t,_,_,_=fixture(a.refinement,.015,0.)
    c=json.loads(Path('configs/explicit-area-pressure.json').read_text());c.update(length_mm=1.,ramp_s=a.ramp_seconds,pressure_mpa=a.pressure,cfl=a.cfl,max_steps=250000,output_every=100,verification_refinement=a.refinement,usage_scope='1mm_layered_plane_pore_20percent_not_particle_packing')
    source_snapshot=Path('packing_fem/area_contact.py').read_bytes()
    r=solve(a.output,X,t,c)
    (a.output/'area-contact-source.py').write_bytes(source_snapshot)
    P=c['pressure_mpa'];E=c['young_mpa'];y=c['yield_mpa'];H=c['hardening_mpa']
    # Independent scalar equilibrium for a homogeneous slab under reference
    # nominal pressure: |Kirchhoff zz| = P * material axial stretch.
    elastic=brentq(lambda lam:E*np.log(lam)+P*lam,.1,1.)
    stretch=brentq(lambda lam:P*lam-y-H*(-np.log(lam)-P*lam/E),.1,1.) if P*elastic>y else elastic
    gap=brentq(lambda d:barrier(d,c['contact_penalty_mpa_mm'],c['barrier_thickness_mm'])[1]-P,1e-8,c['barrier_thickness_mm'])
    height=.8*stretch+gap;phi=gap/height
    checks=dict(axial_endpoint=abs(r['final_axial_compression']-(1-height))<1e-4,porosity_endpoint=abs(r['final_porosity']-phi)<1e-5)
    with np.load(a.output/'last-state.npz') as state:macro_shear=state['generalized_position'][-3:].tolist()
    (a.output/'analytic-comparison.json').write_text(json.dumps(dict(scope=c['usage_scope'],reference_assumptions='homogeneous_diagonal_macro_stretch_branch',observed_macro_shear_xy_xz_yz=macro_shear,material_stretch=stretch,gap_mm=gap,axial_compression=1-height,porosity=phi,checks=checks,solver_checks=r['checks'],solver_status=r['status']),indent=2)+'\n')
    print(json.dumps(r,indent=2));raise SystemExit(0 if all(checks.values()) and all(r['checks'].values()) else 2)
if __name__=='__main__':main()
