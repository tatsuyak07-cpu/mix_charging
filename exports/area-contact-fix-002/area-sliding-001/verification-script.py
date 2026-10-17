"""Finite sliding over periodic triangulated planes, with an analytic Coulomb limit."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from tools.verify_plane_contact import fixture
from packing_fem.area_contact import AreaContact
from packing_fem.explicit_periodic import Periodic
from packing_fem.barrier_contact import barrier


def run():
    X,t,x0,H0,upper=fixture(1,.015,.25);c=AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02)
    r=c.evaluate(x0,x0*0,H0,H0*0,0.);c.history=r[2];c.accept_geometry(x0,H0)
    rows=[];diss=0.;switches=0;expected_force=.3*barrier(.015,20.,.02)[1];yield_slip=expected_force/10.
    for slip in np.linspace(0,.8,513)[1:]:
        x=x0.copy();H=H0.copy();x[upper,0]+=slip;H[0,2]+=slip
        f,_,history,r=c.evaluate(x,x*0,H,H*0,1/512);anchor_error=0.
        for key,v in history.items():
            old=c.history[key];switches+=int(old['face']!=v['face'] or np.any(old['shift']!=v['shift']))
            delta=x[key[1]]-v['bary']@(x[v['anchor_ids']]+v['anchor_shift']@H.T)
            anchor_error=max(anchor_error,float(np.linalg.norm(delta[:2])))
        c.history=history;c.accept_geometry(x,H);diss+=r['friction_increment']
        rows.append(dict(slip_mm=float(slip),tangent_force_N=float(f[upper,0].sum()),normal_force_N=float(f[upper,2].sum()),coulomb_ratio=r['max_coulomb_ratio'],anchor_tangent_offset_mm=anchor_error,friction_dissipation_Nmm=diss))
    exact=expected_force*(.8-yield_slip)
    checks=dict(coulomb_cap=all(r['coulomb_ratio']<=.3+1e-10 for r in rows),sliding_force=abs(rows[-1]['tangent_force_N']+expected_force)<1e-10,friction_dissipation=abs(diss-exact)<1e-10,sliding_anchor_follows=all(r['anchor_tangent_offset_mm']<1e-10 for r in rows[2:]),facet_switch_observed=switches>0)
    return rows,dict(scope='prescribed_periodic_plane_sliding_not_packing',checks=checks,facet_image_switches=switches,analytic_friction_dissipation_Nmm=float(exact))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows,s=run()
    with (a.output/'sliding.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.output/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2));raise SystemExit(0 if all(s['checks'].values()) else 2)
if __name__=='__main__':main()
