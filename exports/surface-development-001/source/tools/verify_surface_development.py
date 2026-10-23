"""Staged contact/surface-bisection validation, not particle compaction."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from packing_fem import sintered
from packing_fem.area_contact import AreaContact
from packing_fem.explicit_material import geometry,hex_to_tet
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.explicit_solver import solve
from packing_fem.surface_remesh import bisect,transfer_contact,audit
from packing_fem.contact_audit import plain
from tools.verify_plane_contact import fixture


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--evidence',type=Path,required=True);args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    c=json.loads(Path('configs/explicit-prototype.json').read_text());c.update(length_mm=1.,contact_backend='area_guarded',loading_mode='nominal_pressure',pressure_mpa=.01,yield_mpa=1e6,ramp_s=.0006,hold_s=.0003,damping_per_s=20000.,certified_local_filter=True,remesh_every=0)
    plan=dict(scope=__doc__,pressure_fixture='1 mm fully dense cube, elastic analytical control; NOT the requested porous packing',contact_fixture='periodic planar slabs, prescribed sticking/sliding',coupon_fixture='archived same mapped endpoint, not independent mesh-pressure solutions',pressure_cfl=[.2,.1],surface_refinement='conforming periodic edge bisection, exact parent material-state subdivision',remesh_tolerance=c['remesh_energy_tolerance'],production_defaults_changed=False,limitations=['No arbitrary surface regeneration/coarsening','No general edge-edge force integration','No 40 to 10 percent packing result','General split/merge and connected equal-minimum contact remain fail-stop'])
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    X,h,_=sintered.mesh(np.ones((2,2,2),bool),.5);t=hex_to_tet(X,h);runs={}
    for name,cfl,steps in [('baseline',.2,[]),('refined-cfl02',.2,[10]),('refined-cfl01',.1,[10])]:
        runs[name]=solve(out/name,X,t,dict(c,cfl=cfl,surface_remesh_steps=steps,surface_remesh_max_events=1));print(name,runs[name]['status'],flush=True)
    expected=brentq(lambda lam:-c['young_mpa']*np.log(lam)/lam-.01,.99,1.)
    # Contact history transfer with finite elastic slip, followed by Coulomb slip.
    X,t,x,H,upper=fixture(2,.015,0.);V0=geometry(X,t)[0];b=np.repeat(np.eye(3)[None],len(t),axis=0);ep=np.linspace(0,.2,len(t));contact=AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02)
    contact.history=contact.evaluate(x,x*0,H,H*0,0.)[2];x=x.copy();H=H.copy();x[upper,0]+=.0001;H[0,2]+=.0001
    before=contact.evaluate(x,x*0,H,H*0,.001);contact.history=before[2]
    edges=np.unique(np.sort(np.concatenate([t[:,[i,j]] for i in range(4) for j in range(i+1,4)]),axis=1),axis=0);edge=next(e for e in edges if np.all(np.isclose(X[e,2],.6)) and np.linalg.norm(X[e[1]]-X[e[0]])==.5)
    p=bisect(X,x,t,b,ep,V0,x*0,edge,1.);new=AreaContact(p['X'],p['t'],Periodic(p['X'],1.),20.,.3,.1,thickness=.02)
    after=transfer_contact(contact,new,p,H);cc=dict(c,yield_mpa=.05)
    transfer=audit((X,x,t,b,ep,V0,x*0),p,cc,before[-1],after[-1],1e-7,old_response=before,new_response=after)
    np.savez_compressed(out/'contact-before.npz',reference_points=X,points=x,tets=t,force=before[0],elastic_b=b,peeq=ep,reference_volume=V0,cell=H)
    np.savez_compressed(out/'contact-after.npz',reference_points=p['X'],points=p['x'],tets=p['t'],force=after[0],elastic_b=p['b'],peeq=p['ep'],reference_volume=p['V0'],supports=p['supports'],parents=p['parents'],cell=H)
    # Preserve full in-memory friction states in the downloadable JSON audit.
    histories=lambda h:[dict(key=list(k),state=plain(v)) for k,v in h.items()]
    (out/'transferred-history.json').write_text(json.dumps(dict(before=histories(contact.history),after=histories(new.history)),indent=2)+'\n')
    y=p['x'].copy();G=H.copy();mask=p['X'][:,2]>.5;y[mask,0]+=.01;G[0,2]+=.01
    sliding=new.evaluate(y,y*0,G,G*0,.001)
    # The same archived mapped geometry at five refinements is only preflight.
    rows=[];root=Path('exports/edge-stationarity-002/preflight')
    with np.load(root/'shape-source/last-state.npz') as z:cell=4*deformation(z['generalized_position'][-6:])
    for n in (1,2,3,4,6):
        with np.load(root/f'mesh-{n}.npz') as z:Y=z['reference_points'];T=z['tets'];mapped=z['points']
        obj=AreaContact(Y,T,Periodic(Y,4.),20.,.3,.1,thickness=.02,edge_stationarity_filter=True,certified_local_filter=True)
        force,extra,hist,report=obj.evaluate(mapped,mapped*0,cell,cell*0,0.)
        rows.append(dict(refinement=n,nodes=len(Y),elements=len(T),contacts=report['contacts'],normal_energy_Nmm=report['normal_energy'],certificate_margin=report['local_filter_geometry_margin'],force_resultant_N=float(np.linalg.norm(force.sum(0)))))
        print('mapped preflight',n,report['contacts'],flush=True)
    with (out/'mapped-preflight.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    checks=dict(pressure_native_checks=all(all(r['checks'].values()) for r in runs.values()),refinement_committed=all(any(e['accepted'] and e['operation']=='periodic_edge_bisection' for e in runs[k]['remesh_events']) for k in ('refined-cfl02','refined-cfl01')),elastic_analytic_limit=all(abs(1-r['final_axial_compression']-expected)<1e-7 for r in runs.values()),one_timestep_halving=abs(runs['refined-cfl02']['final_axial_compression']-runs['refined-cfl01']['final_axial_compression'])<1e-7,contact_transfer_accepted=transfer['accepted'],uniform_reaction_preserved=bool(np.max(abs(before[0][upper].sum(0)-after[0][mask].sum(0)))<1e-12),sliding_coulomb_bound=sliding[-1]['max_coulomb_ratio']<=.3+1e-12,sliding_work_positive=sliding[-1]['friction_increment']>0,mapped_preflight_all_levels=all(r['contacts']>0 and r['certificate_margin']>0 for r in rows))
    result=plain(dict(checks=checks,passed=all(checks.values()),analytic_stretch=expected,pressure_runs={k:dict(status=r['status'],compression=r['final_axial_compression'],remesh_events=r['remesh_events']) for k,r in runs.items()},contact_transfer=transfer,sliding_report=sliding[-1],mapped_preflight=rows,packing_compaction_verified=False,general_surface_remeshing_verified=False,limitations=plan['limitations']))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    from packing_fem.adaptive_tet import boundary
    for ax,z,mesh,title in ((axes[0,0],x,t,'Before surface bisection'),(axes[0,1],p['x'],p['t'],'After periodic surface bisection')):
        faces=boundary(mesh);segments=[]
        for face in faces:
            if np.all(np.isclose(z[face,1],0.)):
                segments.extend(z[face[[i,j]]][:,[0,2]] for i,j in ((0,1),(1,2),(2,0)))
        ax.add_collection(LineCollection(segments,linewidths=.7));ax.autoscale();ax.set_aspect('equal');ax.set(title=title,xlabel='x [mm]',ylabel='z [mm]')
    for name in runs:
        data=np.loadtxt(out/name/'history.csv',delimiter=',',skiprows=1);axes[1,0].plot(data[:,0]*1e3,data[:,1],label=name)
    axes[1,0].axhline(1-expected,color='k',ls='--',lw=.7);axes[1,0].set(xlabel='Time [ms]',ylabel='Axial compression',title='Dense elastic pressure integration control');axes[1,0].legend(fontsize=8)
    axes[1,1].plot([r['refinement'] for r in rows],[r['normal_energy_Nmm'] for r in rows],'o-');axes[1,1].set(xlabel='Refinement',ylabel='Normal energy [N mm]',title='Mapped coupon: NOT mesh convergence')
    fig.tight_layout();fig.savefig(out/'surface-development.png',dpi=170);plt.close(fig)
    shutil.copytree(args.evidence,out/'evidence')
    for folder in ('packing_fem','tests','tools'):
        dest=out/'source'/folder;dest.mkdir(parents=True)
        names=list(Path(folder).glob('*.py')) if folder=='packing_fem' else [Path('tests/test_surface_development.py')] if folder=='tests' else [Path(__file__).resolve()]
        for src in names:shutil.copy2(src,dest/src.name)
    hashes={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob('*')) if f.is_file()}
    (out/'SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    archive=out.with_suffix('.zip')
    if archive.exists():raise FileExistsError(archive)
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,str(f.relative_to(out.parent)))
    print(json.dumps(result['checks']),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
