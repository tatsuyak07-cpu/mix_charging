"""Independent arc-slip check of lagged friction and energy-preserving transport."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from packing_fem.area_contact import AreaContact
from packing_fem.explicit_material import oriented
from packing_fem.explicit_periodic import Periodic
from packing_fem.barrier_contact import barrier


def run(count):
    radius=.015;start=.1;end=1.2
    target=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,-1]])
    def point(theta):return np.array([.3,-radius*np.sin(theta),radius*np.cos(theta)])
    source=point(start)+np.array([[0,0,0],[.4,0,0],[-.2,0,.3],[0,-.3,.3]])
    X=np.r_[target,source]+2;t=oriented(X,np.array([[0,1,2,3],[4,5,6,7]]));H=np.eye(3)*5
    c=AreaContact(X,t,Periodic(X,5.),20.,.3,.1,thickness=.02)
    x=X.copy();value=c.evaluate(x,x*0,H,H*0,0.);c.history=value[2];c.accept_geometry(x,H)
    initial_nodes=[k[1] for k in c.history]
    expected_diss=0.;expected_elastic=0.;slip=radius*(end-start)
    for node in initial_nodes:
        kn=.5*20*c.node_area0[node];kt=.5*kn;cap=.3*barrier(radius,kn,.02)[1];yield_slip=cap/kt
        expected_diss+=cap*max(slip-yield_slip,0.);expected_elastic+=.5*kt*min(slip,yield_slip)**2
    diss=correction=0.;peak=0.
    for theta in np.linspace(start,end,count+1)[1:]:
        x=X.copy();x[4:]+=point(theta)-point(start)
        value=c.evaluate(x,x*0,H,H*0,1/count);c.history=value[2];c.accept_geometry(x,H)
        diss+=value[-1]['friction_increment'];correction+=abs(value[-1]['algorithmic_increment']);peak=max(peak,value[-1]['max_coulomb_ratio'])
    return dict(intervals=count,slip_mm=slip,friction_dissipation_Nmm=diss,analytic_dissipation_Nmm=expected_diss,
        relative_dissipation_error=abs(diss-expected_diss)/expected_diss,elastic_energy_Nmm=value[-1]['tangent_energy'],analytic_elastic_energy_Nmm=expected_elastic,coulomb_ratio=peak,algorithmic_variation_Nmm=correction)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for count in (8,16,32,64):
        rows.append(run(count));print(rows[-1],flush=True)
        with (a.output/'arc-slip.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    errors=[r['relative_dissipation_error'] for r in rows]
    checks=dict(coulomb_cap=all(r['coulomb_ratio']<=.3+1e-10 for r in rows),dissipation_refines=all(a>b for a,b in zip(errors,errors[1:])),finest_relative_error=errors[-1]<1e-3,elastic_energy=all(abs(r['elastic_energy_Nmm']-r['analytic_elastic_energy_Nmm'])<1e-10 for r in rows))
    checks={k:bool(v) for k,v in checks.items()}
    (a.output/'summary.json').write_text(json.dumps(dict(scope='prescribed_arc_slip_not_packing',checks=checks),indent=2)+'\n');raise SystemExit(0 if all(checks.values()) else 2)
if __name__=='__main__':main()
