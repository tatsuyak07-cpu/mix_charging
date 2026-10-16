"""Dynamic contact/remesh integration fixture, NOT a quasi-static benchmark.

Equal/opposite interior-face initial velocities close a slit. The purpose is
to exercise nonzero self-contact and remeshing in the same explicit run.
"""
import argparse,json
from pathlib import Path
import numpy as np
from verify_explicit_contact import fixture
from packing_fem.explicit_solver import solve


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    X,t=fixture();c=json.loads(Path('configs/explicit-prototype.json').read_text())
    velocity=np.zeros_like(X);patch=(X[:,0]>1)&(X[:,0]<4)&(X[:,1]>1)&(X[:,1]<4)
    velocity[patch&np.isclose(X[:,2],2.45),2]=6000.
    velocity[patch&np.isclose(X[:,2],2.55),2]=-6000.
    c.update(length_mm=5.,compression_fraction=.001,ramp_s=.00005,hold_s=.00005,initial_remesh=True,remesh_every=100,remesh_quality=.36,cfl=.15,max_steps=20000,output_every=20,initial_velocity_mm_s=velocity.tolist())
    r=solve(a.output,X,t,c);h=np.genfromtxt(a.output/'history.csv',names=True,delimiter=',')
    check={'purpose':'dynamic contact integration only, NOT quasi-static validation','active_contact_observed':bool(np.max(h['contacts'])>0),'accepted_remesh_observed':any(e['accepted'] for e in r['remesh_events']),'max_contacts':int(np.max(h['contacts'])),'max_penetration_mm':float(np.max(h['penetration_mm'])),'status':r['status'],'checks':r['checks']}
    (a.output/'integration-check.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check,indent=2),flush=True)

if __name__=='__main__':main()
