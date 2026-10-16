"""Sequential gates through guarded experimental compression coupling.

Edge-edge RESPONSE is absent: a detected/ambiguous crossing must STOP a run.
The resulting red gate is retained, not classified as a validated compaction.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from packing_fem import sintered
from packing_fem.explicit_material import oriented,hex_to_tet,geometry
from packing_fem.explicit_periodic import Periodic
from packing_fem.tracked_contact import TrackedContact
from packing_fem.adaptive_tet import propose
from packing_fem.explicit_solver import solve
from verify_explicit_remesh import seed_bipyramid


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);c=json.loads(Path('configs/explicit-prototype.json').read_text())
    # A manufactured nonzero-history contact; not an equilibrium solution.
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.);X[:,2]-=.6;t=hex_to_tet(X,h)
    X,t=seed_bipyramid(X,t,0,.04)
    extra=np.array([[.4,.43,.6],[.3,.33,.8],[.5,.33,.8],[.4,.53,.8]])
    ids=np.arange(len(X),len(X)+4);X=np.r_[X,extra];t=oriented(X,np.r_[t,ids[None]])
    con=TrackedContact(X,t,Periodic(X,3.),100.,.3,.1);x=X.copy();x[-4:,2]-=.21
    rows=[];transfers=0
    for u in np.linspace(.4,.47,71):
        current=x.copy();current[-4:,0]+=u-.4
        f,extra_force,hist,ci=con.evaluate(current,np.zeros_like(x),np.eye(3)*3,np.zeros((3,3)),.001)
        con.history=hist;transfers+=ci['transferred_pairs']
        rows.append([u,ci['contacts'],ci['transferred_pairs'],ci['normal_energy'],ci['tangent_energy'],ci['friction_increment'],ci['algorithmic_increment'],ci['max_coulomb_ratio']])
    with (a.output/'tracking.csv').open('w',newline='') as file:
        w=csv.writer(file);w.writerow(['coordinate_mm','contacts','transferred_pairs','normal_Nmm','tangent_Nmm','friction_increment_Nmm','algorithmic_increment_Nmm','coulomb_ratio']);w.writerows(rows)
    # Internal flip while normal force AND tangential history are nonzero.
    b=np.repeat(np.eye(3)[None],len(t),axis=0);ep=np.full(len(t),.2);V0=geometry(X,t)[0]
    proposal=propose(current,t,b,ep,V0,c,1.,.001)
    if proposal is None:raise RuntimeError('Manufactured active-contact flip unavailable')
    t2,b2,ep2,V02,event=proposal
    old_faces=con.faces.copy();con.update_topology(t2)
    f2,ex2,hist2,ci2=con.evaluate(current,np.zeros_like(x),np.eye(3)*3,np.zeros((3,3)),0.)
    force_error=float(np.max(abs(f2-f)));energy_error=abs(ci2['tangent_energy']-ci['tangent_energy'])
    flip=dict(event,contacts=ci['contacts'],tangent_energy_before_Nmm=ci['tangent_energy'],force_change_N=force_error,
        tangent_energy_change_Nmm=energy_error,history_keys_preserved=set(hist)==set(hist2),
        volume_error_mm3=float(abs(V02.sum()-V0.sum())),peeq_error=float(np.max(abs(ep2-.2))))
    np.savez_compressed(a.output/'active-contact-remesh.npz',reference_points=X,points=current,tets_before=t,tets_after=t2,peeq_after=ep2,reference_volume_after=V02)
    gates={'tracking':dict(passed=bool(transfers>0 and ci['tangent_energy']>0),transferred_pairs=transfers),
        'active_contact_internal_flip':dict(passed=bool(force_error<1e-10 and energy_error<1e-12 and flip['history_keys_preserved'] and flip['peeq_error']<1e-12),details=flip),
        'edge_edge_response':dict(passed=False,reason='No response law: swept crossings/ambiguous coplanar sweeps stop the run')}
    published=Path('exports/explicit-fem-prototype-002');runs={}
    for name,source,scale in [('solid','solid',1.),('void','void_time1_cfl0.2',1.),('impact','dynamic_impact',1.),('impact_half_dt','dynamic_impact',.5)]:
        original=published/source;config=json.loads((original/'config.json').read_text())
        with np.load(original/'initial-mesh.npz') as m:points,tets=m['points'],m['tets']
        config.update(contact_backend='tracked_guarded',cfl=config['cfl']*scale,output_every=20)
        runs[name]=solve(a.output/name,points,tets,config)
        print(json.dumps(dict(case=name,status=runs[name]['status'],failure=runs[name]['failure'],steps=runs[name]['steps'])),flush=True)
        (a.output/'partial-results.json').write_text(json.dumps(runs,indent=2)+'\n')
    gates['solver_noncontact_regression']=dict(passed=all(all(runs[n]['checks'].values()) for n in ('solid','void')))
    gates['physical_contact_compaction']=dict(passed=False,reason='Requires edge response, complete path, contact/time/mesh convergence and quasi-static checks')
    summary=dict(gates=gates,runs=runs,software_coupling_implemented=True,physical_integration_validated=False,
        original_material_geometry_friction_unchanged=True)
    (a.output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(gates,indent=2))

if __name__=='__main__':main()
