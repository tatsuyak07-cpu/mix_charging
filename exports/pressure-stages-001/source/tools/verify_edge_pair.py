"""Manufactured edge contact force/work verification, not packing mechanics."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from packing_fem.edge_contact import start_pair,response


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    before=np.array([[-1.,0,0],[1.,0,0],[0,-1,.01],[0,1,.01]])
    after=before.copy();after[2:,2]=-.01;initial=start_pair(before,after,.5)
    origin=before.copy();origin[2:,2]=0.;rows=[];checks={}
    for intervals in (100,200):
        state=initial;oldx=origin.copy();oldforce=np.zeros_like(origin)
        work=friction=correction=0.;maxmoment=maxbalance=maxratio=0.
        for step in range(1,intervals+1):
            s=step/intervals;x=origin.copy();x[2:,2]=-.01*s;x[2:,0]=.02*s
            r=response(x,state,10.,5.,.3);state=r['state']
            work-=float(np.sum(.5*(oldforce+r['force'])*(x-oldx)))
            friction+=r['friction_increment'];correction+=r['algorithmic_increment']
            energy=r['normal_energy']+r['tangent_energy'];residual=work-energy-friction-correction
            maxmoment=max(maxmoment,float(np.linalg.norm(np.cross(x,r['force']).sum(axis=0))))
            maxbalance=max(maxbalance,float(np.linalg.norm(r['force'].sum(axis=0))))
            maxratio=max(maxratio,r['coulomb_ratio'])
            rows.append([intervals,s,r['depth_mm'],r['normal_force_N'],r['tangent_energy'],energy,work,friction,correction,residual,maxmoment,maxbalance])
            oldx=x;oldforce=r['force']
        checks[str(intervals)]=dict(work_Nmm=work,analytic_work_Nmm=.0008,energy_Nmm=energy,
            friction_Nmm=friction,correction_Nmm=correction,work_residual_Nmm=residual,
            force_balance_N=maxbalance,moment_Nmm=maxmoment,coulomb_ratio=maxratio,
            passed=bool(abs(work-.0008)<1e-12 and abs(energy-.00059)<1e-12 and abs(residual)<1e-12 and maxbalance<1e-12 and maxmoment<1e-12 and maxratio<=.3+1e-12))
    with (a.output/'pair-path.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['intervals','path_fraction','depth_mm','normal_force_N','tangent_Nmm','energy_Nmm','work_Nmm','friction_Nmm','correction_Nmm','residual_Nmm','moment_Nmm','force_balance_N']);w.writerows(rows)
    summary=dict(purpose=__doc__,normal_stiffness_N_mm=10.,tangent_stiffness_N_mm=5.,friction=.3,checks=checks,all_checks_passed=all(c['passed'] for c in checks.values()))
    (a.output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    if not summary['all_checks_passed']:raise RuntimeError('Edge pair verification failed')

if __name__=='__main__':main()
