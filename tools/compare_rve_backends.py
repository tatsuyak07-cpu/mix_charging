"""NumPy/native agreement on a stopped plastic RVE path, not a physical solution."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--numpy',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sa=json.loads((a.numpy/'summary.json').read_text());sb=json.loads((a.native/'summary.json').read_text());A=np.loadtxt(a.numpy/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1);B=np.loadtxt(a.native/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1)
    limits=dict(points=1e-7,reference_points=1e-12,tets=0.,elastic_b=1e-8,peeq=1e-8,reference_volume=1e-12,generalized_position=1e-8,generalized_velocity=1e-5);errors={};checks={}
    with np.load(a.numpy/'last-state.npz') as x,np.load(a.native/'last-state.npz') as y:
        for key,limit in limits.items():
            errors[key]=float(np.max(abs(x[key]-y[key])));checks[key]=errors[key]<=limit
    same_shape=A.shape==B.shape
    checks.update(same_step_count=sa['steps']==sb['steps'],same_audit_shape=same_shape,same_stop_reason=sa['failure']==sb['failure'])
    if same_shape:
        diff=np.max(abs(A-B),axis=0);checks.update(audit_times=bool(diff[0]<1e-12),audit_compression=bool(diff[1]<1e-8),audit_porosity=bool(diff[2]<1e-8),audit_mechanical_energies=bool(np.max(diff[4:10])<1e-7*max(abs(A[:,5]).max(),1e-12)));np.savetxt(a.output/'maximum-audit-difference.csv',diff[None],delimiter=',',header='time,compression,porosity,command_pressure,kinetic,internal,work,energy_residual,normal_contact,tangent_contact,contacts',comments='')
    result=dict(scope=__doc__,checks=checks,passed=all(checks.values()),absolute_state_differences=errors,predeclared_state_tolerances=limits,steps=sa['steps'],numpy_status=sa['status'],native_status=sb['status'],physical_equilibrium_verified=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'compare_rve_backends.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
