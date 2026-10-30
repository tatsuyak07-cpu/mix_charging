"""Prescribed planar contact verification, not loaded packing compaction."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from packing_fem.embedded_contact import EmbeddedContact
from packing_fem.embedded_remesh import refine_rules,transfer,coordinates
from packing_fem.explicit_material import geometry
from packing_fem.explicit_periodic import Periodic
from packing_fem.surface_remesh import bisect,audit
from packing_fem.triangle_quadrature import RULES
from tools.verify_plane_contact import fixture


def case(fraction=.5,slip=.0001,n=2):
    X,t,_,_,upper=fixture(n,.015,0.)
    b=np.repeat(np.eye(3)[None],len(t),0);ep=np.full(len(t),.2);V=geometry(X,t)[0]
    edges=np.unique(np.sort(np.concatenate([t[:,[i,j]] for i in range(4) for j in range(i+1,4)]),axis=1),axis=0)
    edge=next(e for e in edges if np.all(np.isclose(X[e,2],.6)) and (np.linalg.norm(X[e[1]]-X[e[0]])>1. if n==1 else np.linalg.norm(X[e[1]]-X[e[0]])==.5))
    base=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,error_budget='normalized')
    p=bisect(X,X,t,b,ep,V,X*0,edge,1.,fraction=fraction)
    child=EmbeddedContact(p['X'],p['t'],Periodic(p['X'],1.),20.,.3,.1,thickness=.02,error_budget='normalized')
    # Prepare integration-only rules on the anticipated nested mesh BEFORE
    # creating any production history; replay the prescribed path if needed.
    from packing_fem.area_contact import SurfaceResolutionError
    child_rules={};B,w=RULES[8]
    for j,ids in enumerate(child.faces):
        tri=p['X'][ids];A=np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2
        child_rules[str(j)]=dict(bary=B.tolist(),weights=(A*w).tolist())
    path=[(float(g),0.) for g in np.linspace(.025,.015,5)]+[(.015,float(d)) for d in np.linspace(0,slip,5)[1:]]
    path +=[(.015,float(slip+d)) for d in np.linspace(-.0001,-.005,21)]
    for replay in range(12):
        prepared=EmbeddedContact(p['X'],p['t'],Periodic(p['X'],1.),20.,0.,.1,thickness=.02,error_budget='normalized',quadrature_rules=child_rules)
        try:
            for gap,d in path:
                _,_,z,G,_=fixture(n,gap,0.);z[upper,0]+=d;G[0,2]+=d
                shape=p['weights'][:,:,None];y=(shape*z[p['supports']]).sum(1)
                response=prepared.evaluate(y,y*0,G,G*0,.0001)
                prepared.history=response[2];prepared.accept_geometry(y,G)
            break
        except SurfaceResolutionError as exc:
            if not getattr(exc,'feature',{}).get('rules'):raise
            child_rules.update(exc.feature['rules'])
    else:raise AssertionError('Precontact integration preparation exhausted')
    rules={}
    for i,face in enumerate(base.faces):
        pieces=[];weights=[]
        for j,ids in enumerate(child.faces):
            C=coordinates(X[face],p['X'][ids],1e-10)
            if C is None:continue
            rule=child_rules[str(j)];pieces.extend(np.asarray(rule['bary'])@C);weights.extend(rule['weights'])
        rules[str(i)]=dict(bary=np.asarray(pieces).tolist(),weights=weights)
    old=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,error_budget='normalized',quadrature_rules=rules)
    for gap in np.linspace(.025,.015,5):
        _,_,x,H,_=fixture(n,float(gap),0.)
        r=old.evaluate(x,x*0,H,H*0,.0001);old.history=r[2];old.accept_geometry(x,H)
    for d in np.linspace(0,slip,5)[1:]:
        _,_,x,H,_=fixture(n,.015,0.);x[upper,0]+=d;H[0,2]+=d
        r=old.evaluate(x,x*0,H,H*0,.0001);old.history=r[2];old.accept_geometry(x,H)
    p=bisect(X,x,t,b,ep,V,x*0,edge,1.,fraction=fraction)
    rules,mapping,parent=refine_rules(old,child.faces,p)
    new=EmbeddedContact(p['X'],p['t'],Periodic(p['X'],1.),20.,.3,.1,thickness=.02,error_budget='normalized',quadrature_rules=rules)
    return old,new,p,H,r,mapping,parent,(X,x,t,b,ep,V,x*0)



def verify(fraction,slip,n,output):
    from scipy.sparse import csr_matrix
    from packing_fem.barrier_contact import barrier
    old,new,p,H,before,mapping,parent,state=case(fraction,slip,n)
    after,report=transfer(old,new,p,H,mapping,parent,before,1e-8)
    c=json.loads(Path('configs/explicit-prototype.json').read_text());c.update(length_mm=1.,friction=.3)
    audited=audit(state,p,c,before[-1],after[-1],1e-8,old_response=before,new_response=after)
    report.update(audited)
    new.accept_geometry(p['x'],H)
    N=len(p['X']);R=csr_matrix((p['weights'].ravel(),(np.repeat(np.arange(N),2),p['supports'].ravel())),shape=(N,len(old.X)))
    comparisons=[]
    # Follow both models through reverse sliding and separation, with no reset.
    for displacement in np.linspace(-.0001,-.005,21):
        x=state[1].copy();G=H.copy();upper=old.X[:,2]>.5;x[upper,0]+=displacement;G[0,2]+=displacement
        y=np.asarray(R@x)
        a=old.evaluate(x,x*0,G,G*0,.0001);b=new.evaluate(y,y*0,G,G*0,.0001)
        f0=old.periodic.force(a[0],a[1]);f1=old.periodic.force(np.asarray(R.T@b[0]),b[1])
        force_error=float(np.linalg.norm(f1-f0)/max(np.linalg.norm(f0),1e-16))
        energy_error=max(abs(a[-1][k]-b[-1][k])/max(abs(a[-1][k]),1e-16) for k in ('normal_energy','tangent_energy'))
        diss_error=abs(a[-1]['friction_increment']-b[-1]['friction_increment'])/max(abs(a[-1]['friction_increment']),abs(b[-1]['friction_increment']),1e-16)
        comparisons.append(dict(displacement=displacement,force_error=force_error,energy_error=energy_error,dissipation_error=diss_error,coulomb=b[-1]['max_coulomb_ratio']))
        for contact,r,z in ((old,a,x),(new,b,y)):
            contact.history=r[2];contact.accept_geometry(z,G)
    # Open the gap incrementally; accumulate released history energy rather
    # than reading only the already-separated final zero-history step.
    released=[0.,0.]
    for opening in np.linspace(.0015,.015,10):
        x=state[1].copy();G=H.copy();upper=old.X[:,2]>.5
        x[upper,0]-=.005;G[0,2]-=.005;x[upper,2]+=opening;G[2,2]+=opening;y=np.asarray(R@x)
        a=old.evaluate(x,x*0,G,G*0,.0001);b=new.evaluate(y,y*0,G,G*0,.0001)
        for i,(contact,r,z) in enumerate(((old,a,x),(new,b,y))):
            released[i]+=r[-1]['released_history_energy'];contact.history=r[2];contact.accept_geometry(z,G)
    separation=dict(no_active_histories=not a[2] and not b[2],release_energy_difference_Nmm=abs(released[0]-released[1]))
    E,F,_=barrier(.015,20.,.02);spring=min(slip,.3*F/10.)
    analytic=dict(normal_energy_error=abs(before[-1]['normal_energy']-E)/E,tangent_energy_error=abs(before[-1]['tangent_energy']-5*spring**2)/max(5*spring**2,1e-16))
    checks=dict(instantaneous_audit=report['accepted'],virtual_work=report['periodic_virtual_work_relative_error']<=1e-8,
                continued_path=all(max(v['force_error'],v['energy_error'],v['dissipation_error'])<=1e-7 and v['coulomb']<=.3+1e-12 for v in comparisons),
                separated=separation['no_active_histories'] and separation['release_energy_difference_Nmm']<=1e-12,
                analytic=max(analytic.values())<=.002)
    checks={name:bool(value) for name,value in checks.items()}
    result=dict(fraction=fraction,slip_mm=slip,passed=all(checks.values()),checks=checks,transfer=report,continuation=comparisons,separation=separation,analytic=analytic)
    np.savez_compressed(output/'mesh-and-samples.npz',reference_before=state[0],points_before=state[1],tets_before=state[2],reference_after=p['X'],points_after=p['x'],tets_after=p['t'],samples_before=old.P@state[1],samples_after=new.P@p['x'],force_before=before[0],force_after=after[0],weights_before=old.inner.integration_area0,weights_after=new.inner.integration_area0)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result,old,new,p,state


def main():
    import argparse,csv,hashlib,shutil,zipfile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--n',type=int,default=1);args=parser.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=False);results=[]
    source=root/'source';source.mkdir()
    for path in Path('packing_fem').glob('*.py'):shutil.copy2(path,source/path.name)
    shutil.copy2(__file__,source/Path(__file__).name)
    for fraction in (.2,.5,.8):
        for slip in (.0001,.01):
            target=root/f'fraction-{fraction}-slip-{slip}';target.mkdir()
            try:
                result,old,new,p,state=verify(fraction,slip,args.n,target)
                if fraction==.5 and slip==.01:
                    fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
                    for ax,contact,z,title in ((axes[0],old,state[1],'Before'),(axes[1],new,p['x'],'After nested bisection')):
                        for ids in contact.faces:
                            if not np.all(np.isclose(contact.X[ids,2],.6)):continue
                            tri=z[np.r_[ids,ids[0]]];ax.plot(tri[:,0],tri[:,1],color='black',lw=1)
                        nodes=np.flatnonzero((contact.inner.integration_area0>0)&np.isclose(contact.inner.X[:,2],.6));samples=contact.P@z
                        ax.scatter(samples[nodes,0],samples[nodes,1],s=10,color='#267ba6');ax.set(title=title,xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
                    fig.suptitle('Prescribed plane contact: same material samples and friction histories\nNot a packing compaction or quality-repair result');fig.savefig(root/'retained-samples.png',dpi=170);plt.close(fig)
            except Exception as exc:
                import traceback
                result=dict(fraction=fraction,slip_mm=slip,passed=False,error=repr(exc),traceback=traceback.format_exc());(target/'failure.json').write_text(json.dumps(result,indent=2)+'\n')
            results.append(result);print(json.dumps(result),flush=True)
    summary=dict(passed=all(r['passed'] for r in results),scope=__doc__,cases=results,limits=['Partition-aligned retained-node refinement only.','No arbitrary sample insertion, pose-atlas split, coarsening, quality recovery or packing 40-to-10% solution.'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (root/'comparison.csv').open('w',newline='') as stream:
        fields=['fraction','slip_mm','passed','periodic_virtual_work_error','normal_energy_error','tangent_energy_error'];w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
        for r in results:w.writerow(dict(fraction=r['fraction'],slip_mm=r['slip_mm'],passed=r['passed'],periodic_virtual_work_error=r.get('transfer',{}).get('periodic_virtual_work_relative_error'),**r.get('analytic',{})))
    if not summary['passed']:raise SystemExit(2)


if __name__=='__main__':main()
