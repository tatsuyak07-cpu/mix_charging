"""Quasi-static full-path finite-strain packing experiment (not certified).

All accepted states satisfy the reduced force residual. Failed iterates are
saved separately and are never used as the next material history.
"""
import argparse,csv,json,time,traceback,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy.sparse import diags
from packing_fem.implicit_linear import solve as block_solve
from packing_fem.explicit_material import geometry,oriented
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.implicit_material import evaluate
from packing_fem.ipc_periodic import PeriodicIPC
from packing_fem.adaptive_tet import quality
from packing_fem.fixed_point import anderson_candidate,affine_predictor


def active_dofs(p,c,prescribed):
    """Fixed transverse lattice vectors: only Fzz is a macro unknown.

    Periodic microscopic fluctuations remain free. Transverse reactions are
    constraint reactions, not equilibrium errors. Unloading keeps the same box.
    """
    mode=c.get('transverse_boundary','stress_free')
    if mode=='stress_free':active=np.arange(p.ndof)
    elif mode=='fixed_cell':active=np.r_[np.arange(p.nmicro),p.prescribed]
    else:raise ValueError('Unknown transverse_boundary')
    physical=active[active!=p.prescribed] if prescribed is not None else active
    return active,physical


def equilibrium(p,t,q,b,ep,inverse,V0,c,contact,prescribed,external,log,parent=None,initial_guess=None):
    if c.get('implicit_optimizer','newton')=='lbfgs':
        from packing_fem.quasistatic_lbfgs import equilibrium as lbfgs_equilibrium
        return lbfgs_equilibrium(p,t,q,b,ep,inverse,V0,c,contact,prescribed,external,log,parent,initial_guess)
    if c.get('implicit_optimizer')=='coupled_newton':
        from packing_fem.quasistatic_coupled import equilibrium as coupled_equilibrium
        return coupled_equilibrium(p,t,q,b,ep,inverse,V0,c,contact,prescribed,external,log,parent,initial_guess)
    oldq=q.copy();trial=q.copy() if initial_guess is None else initial_guess.copy()
    use_initial_guess=initial_guess is not None
    if prescribed is not None and c.get('implicit_affine_predictor',False):
        trial=affine_predictor(p,trial,prescribed);use_initial_guess=True
        log(dict(affine_predictor=True,compression=float(-trial[p.prescribed])))
    actuator=c.get('implicit_actuator_Nmm',1e5) if prescribed is not None else 0.
    multiplier=0.
    free,physical_free=active_dofs(p,c,prescribed)
    fixed=np.setdiff1d(np.arange(p.ndof),free)
    if np.any(abs(q[fixed])>1e-12):raise ValueError('Fixed-cell restart must have the initial transverse lattice; start from the unloaded initial mesh')
    scale=np.ones(p.ndof);scale[-6:]=1/p.L
    contact.set_lag(oldq,oldq)
    depth=c.get('implicit_anderson_depth',0);outputs=[];residuals=[];lag_input=trial.copy()
    displacement_scale=np.ones(p.ndof);displacement_scale[-6:]=p.L
    try:
        if prescribed is not None:
            _,old_g,old_K,_=evaluate(p.positions(oldq),t,b,ep,inverse,V0,c,True,parent)
            _,contact_g,contact_K,_=contact.evaluate(oldq,True)
            multiplier=-float((p.A.T@old_g.ravel()+contact_g)[p.prescribed])
            column=p.A[:,p.prescribed]
            affine_tangent=float((column.T@old_K@column).toarray()[0,0]+contact_K[p.prescribed,p.prescribed])
            actuator=max(actuator,c.get('implicit_actuator_tangent_multiplier',10.)*abs(affine_tangent))
        if use_initial_guess:
            if np.any(abs(trial[fixed])>1e-12) or np.linalg.det(deformation(trial[-6:]))<=0 or np.min(geometry(p.positions(trial),t)[0])<=0:raise ValueError('Inadmissible initial iterate')
            contact.set_lag(trial,oldq)
            if prescribed is not None:
                _,wg,_,_=evaluate(p.positions(trial),t,b,ep,inverse,V0,c,parent=parent)
                _,wcg,_,_=contact.evaluate(trial)
                multiplier=-float((p.A.T@wg.ravel()+wcg)[p.prescribed])
        for lag in range(c['implicit_max_lag']):
            converged=False
            for it in range(c['implicit_max_newton']):
                if np.linalg.det(deformation(trial[-6:]))<=0:raise ValueError('Nonpositive periodic cell volume')
                em,gm,Km,state=evaluate(p.positions(trial),t,b,ep,inverse,V0,c,True,parent)
                ec,gc,Kc,n=contact.evaluate(trial,True)
                internal=np.asarray(p.A.T@gm.ravel()).ravel()+gc
                g=internal-external;energy=em+ec-float(external@trial)
                error=0. if prescribed is None else trial[p.prescribed]-prescribed
                g[p.prescribed]+=multiplier+actuator*error
                energy+=multiplier*error+.5*actuator*error**2
                norm=float(np.max(abs(g[free]*scale[free])))
                log(dict(lag=lag,newton=it,compression=float(-trial[p.prescribed]),residual_N=norm,contacts=n,actuator_Nmm=actuator))
                if norm<c['implicit_force_tolerance_N']:converged=True;break
                projected=c.get('implicit_project_material_tangent',True) and c.get('ipc_project_tangent',True)
                if not projected and norm>c.get('implicit_exact_tangent_max_residual_N',float('inf')):
                    _,_,Km,_=evaluate(p.positions(trial),t,b,ep,inverse,V0,dict(c,implicit_project_material_tangent=True),True,parent)
                    if not c.get('ipc_project_tangent',True):_,_,Kc,_=contact.evaluate(trial,True,project_hessian=True)
                    projected=True
                for project in ([True] if projected else [False,True]):
                    if project and not projected:
                        _,_,Km,_=evaluate(p.positions(trial),t,b,ep,inverse,V0,dict(c,implicit_project_material_tangent=True),True,parent)
                        if not c.get('ipc_project_tangent',True):_,_,Kc,_=contact.evaluate(trial,True,project_hessian=True)
                        log(dict(projected_tangent_fallback=True,lag=lag,newton=it))
                    K=(p.A.T@Km@p.A+Kc).tocsr()
                    add=np.zeros(p.ndof);add[p.prescribed]=actuator;K=K+diags(add);Kf=K[free][:,free]
                    diagonal=np.maximum(abs(Kf.diagonal()),1e-6)
                    try:
                        d=np.zeros(p.ndof);d[free]=block_solve(Kf+diags(diagonal*c['implicit_regularization']),-g[free],p.nmicro)
                        if not np.all(np.isfinite(d)) or g@d>=0:raise ValueError('Newton direction is not a descent direction')
                        break
                    except (ValueError,RuntimeError):
                        if project:raise
                bound=contact.ccd(trial,d);alpha=1. if bound>=1 else .9*bound
                accepted=False
                roundoff=32*np.finfo(float).eps*max(c['young_mpa']*float(V0.sum()),abs(energy),1e-30)
                for ls in range(35):
                    candidate=trial+alpha*d
                    try:
                        if np.linalg.det(deformation(candidate[-6:]))<=0:raise ValueError('Nonpositive periodic cell volume')
                        ve,vg,_,_=evaluate(p.positions(candidate),t,b,ep,inverse,V0,c,parent=parent)
                        ce,vcg,_,_=contact.evaluate(candidate)
                        candidate_error=0. if prescribed is None else candidate[p.prescribed]-prescribed
                        candidate_energy=ve+ce-float(external@candidate)+multiplier*candidate_error+.5*actuator*candidate_error**2
                        if candidate_energy<=energy+1e-4*alpha*(g@d):accepted=True;break
                        if candidate_energy<=energy+roundoff:
                            candidate_g=np.asarray(p.A.T@vg.ravel()).ravel()+vcg-external
                            candidate_g[p.prescribed]+=multiplier+actuator*candidate_error
                            candidate_norm=float(np.max(abs(candidate_g[free]*scale[free])))
                            if candidate_norm<norm*(1-1e-4*alpha):
                                log(dict(roundoff_residual_guard=True,residual_before_N=norm,residual_after_N=candidate_norm,energy_window_Nmm=roundoff))
                                accepted=True;break
                    except (ValueError,RuntimeError):pass
                    alpha*=.5
                if not accepted:raise ValueError('Line search failed')
                trial=candidate
            if not converged:raise ValueError('Newton iteration budget exhausted')
            contact.set_lag(trial,oldq)
            _,gm,_,state=evaluate(p.positions(trial),t,b,ep,inverse,V0,c,parent=parent)
            _,gc,_,n=contact.evaluate(trial)
            internal=np.asarray(p.A.T@gm.ravel()).ravel()+gc
            norm=float(np.max(abs((internal-external)[physical_free]*scale[physical_free])))
            error=0. if prescribed is None else trial[p.prescribed]-prescribed
            multiplier+=actuator*error
            if norm<c['implicit_force_tolerance_N'] and abs(error)<c.get('implicit_macro_tolerance',1e-7):
                return trial,state,internal,n,it,norm
            if depth>1:
                outputs.append(trial.copy());residuals.append(trial-lag_input)
                outputs=outputs[-depth:];residuals=residuals[-depth:]
                candidate=anderson_candidate(outputs,residuals,displacement_scale)
                if candidate is not None:
                    candidate[fixed]=0.;direction=candidate-trial
                    bound=contact.ccd(trial,direction);fraction=1. if bound>=1 else .9*bound
                    accepted_acceleration=False
                    for safeguard in range(20):
                        candidate=trial+fraction*direction
                        candidate_error=0. if prescribed is None else candidate[p.prescribed]-prescribed
                        if abs(candidate_error)>max(abs(error),c.get('implicit_macro_tolerance',1e-7)):
                            fraction*=.5;continue
                        try:
                            if np.linalg.det(deformation(candidate[-6:]))<=0 or np.min(geometry(p.positions(candidate),t)[0])<=0:raise ValueError('Inadmissible Anderson proposal')
                            _,cg,_,cs=evaluate(p.positions(candidate),t,b,ep,inverse,V0,c,parent=parent)
                            contact.set_lag(candidate,oldq)
                            _,ccg,_,cn=contact.evaluate(candidate)
                            ci=np.asarray(p.A.T@cg.ravel()).ravel()+ccg
                            candidate_norm=float(np.max(abs((ci-external)[physical_free]*scale[physical_free])))
                            if candidate_norm<norm:
                                log(dict(anderson_lag=lag,residual_before_N=norm,residual_after_N=candidate_norm,fraction=fraction,history=len(outputs)))
                                trial=candidate;accepted_acceleration=True
                                if candidate_norm<c['implicit_force_tolerance_N'] and abs(candidate_error)<c.get('implicit_macro_tolerance',1e-7):
                                    return trial,cs,ci,cn,it,candidate_norm
                                break
                        except (ValueError,RuntimeError):pass
                        # A non-improving proposal restores the established map.
                        break
                    if not accepted_acceleration:contact.set_lag(trial,oldq)
            lag_input=trial.copy()
        raise ValueError('Friction lag iterations did not converge')
    except (ValueError,RuntimeError) as exc:
        return dict(reason=str(exc),iterate=trial)


def run(config,mesh,out,restart=None,initial_iterate=None):
    c=json.loads(Path(config).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(c,indent=2))
    if c.get('implicit_optimizer','newton') not in ('newton','lbfgs','coupled_newton'):raise ValueError('Unknown implicit optimizer')
    if c.get('implicit_optimizer') in ('lbfgs','coupled_newton') and (c.get('transverse_boundary')!='fixed_cell' or not c.get('loading_only',False)):raise ValueError('Direct-displacement development backends require fixed transverse vectors and compression-only scope')
    if c.get('implicit_iterate_checkpoint_every',0)>0:
        trace=out/'iterate-trace';trace.mkdir();c['_iterate_checkpoint_directory']=str(trace)
    sources=out/'source';sources.mkdir()
    manifest={}
    for name in ['tools/run_implicit_compaction.py','packing_fem/fixed_point.py','packing_fem/line_search.py','packing_fem/quasistatic_lbfgs.py','packing_fem/quasistatic_coupled.py','packing_fem/implicit_material.py','packing_fem/ipc_periodic.py','packing_fem/implicit_remesh.py','packing_fem/implicit_linear.py','packing_fem/explicit_material.py','packing_fem/explicit_periodic.py']:
        source=Path(name);target=sources/source.name;shutil.copy2(source,target);manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    (out/'source-sha256.json').write_text(json.dumps(manifest,indent=2))
    (out/'inputs.json').write_text(json.dumps(dict(mesh=str(mesh),mesh_sha256=hashlib.sha256(Path(mesh).read_bytes()).hexdigest(),restart=None if restart is None else str(restart),initial_iterate=None if initial_iterate is None else str(initial_iterate),initial_iterate_sha256=None if initial_iterate is None else hashlib.sha256(Path(initial_iterate).read_bytes()).hexdigest(),iterate_is_initial_guess_only=True),indent=2))
    data=np.load(mesh);X=data['points'];t=oriented(X,data['tets'])
    if restart:
        r=np.load(restart);X=r['X'];t=oriented(X,r['t'])
    p=Periodic(X,c['length_mm']);q=np.zeros(p.ndof);b=np.tile(np.eye(3),(len(t),1,1));ep=np.zeros(len(t))
    V0,_,D=geometry(X,t);inverse=np.linalg.inv(D);parent=np.arange(len(t))
    if restart:
        r=np.load(restart)
        if abs(r['V0'].sum()/geometry(data['points'],oriented(data['points'],data['tets']))[0].sum()-1)>1e-6:raise ValueError('Restart material volume mismatch')
        q=r['q'];b=r['b'];ep=r['ep'];V0=r['V0'];parent=r['parent'] if 'parent' in r else np.arange(len(t));inverse=np.linalg.inv(geometry(p.positions(q),t)[2])
    initial_mesh_porosity=float(1-geometry(data['points'],oriented(data['points'],data['tets']))[0].sum()/p.L**3)
    if 'initial_porosity_tolerance' in c and abs(initial_mesh_porosity-c['requested_initial_porosity'])>c['initial_porosity_tolerance']:raise ValueError('Measured mesh porosity outside the configured initial geometry tolerance')
    contact=PeriodicIPC(X,t,p,c)
    warm_guess=None;warm_X=None;warm_t=None
    if initial_iterate:
        if not restart:raise ValueError('An initial iterate requires an accepted restart state')
        with np.load(initial_iterate) as guess:
            for name,expected in [('X',X),('t',t),('oldq',q),('b',b),('ep',ep)]:
                if not np.array_equal(guess[name],expected):raise ValueError('Initial iterate does not belong to this accepted material history')
            warm_guess=guess['q'].copy();warm_X=X.copy();warm_t=t.copy()
        import ipctk as ipc
        if ipc.has_intersections(contact.mesh,contact.positions(warm_guess)):raise ValueError('Initial iterate has surface intersections')
    print('mesh',len(X),len(t),'ghost',len(contact.X),'ndof',p.ndof,flush=True)
    initial=1-geometry(data['points'],oriented(data['points'],data['tets']))[0].sum()/p.L**3
    rows=[];start=time.monotonic();increment=c.get('implicit_initial_strain_increment',c['implicit_strain_increment']);index=0;failed=0
    if not 0<increment<=c['implicit_strain_increment']:raise ValueError('Initial load increment must be positive and no larger than its ceiling')
    remesh_events=0;remesh_attempted_at=-1
    phase=c.get('initial_phase','compression')
    if phase not in ('compression','hold'):raise ValueError('initial_phase must be compression or hold')
    zero=np.zeros(p.ndof);unload_fraction=0.;unload_reaction=None;loaded_porosity=None
    def log(v):print(json.dumps(v),flush=True)
    def phi_now():return float(1-geometry(p.positions(q),t)[0].sum()/(p.L**3*np.linalg.det(deformation(q[-6:]))))
    already_loaded=phase=='compression' and phi_now()<=c['target_final_porosity']+c['porosity_tolerance']
    if already_loaded and not c.get('loading_only',False):phase='hold'
    if phase=='hold':loaded_porosity=phi_now()
    np.savez_compressed(out/'state-0000.npz',X=X,x=p.positions(q),t=t,q=q,b=b,ep=ep,V0=V0,parent=parent)
    if already_loaded and c.get('loading_only',False):
        (out/'status.json').write_text(json.dumps(dict(status='loaded_target_reached',initial_porosity=float(initial),loaded_porosity=phi_now(),target_porosity=c['target_final_porosity'],target_reached=True,validated=False),indent=2));return
    while index<c['implicit_max_stages']:
        if (phase!='unload' and c.get('implicit_remesh',False) and index!=remesh_attempted_at
                and quality(p.positions(q),t).min()<c['implicit_remesh_quality']):
            from packing_fem.implicit_remesh import refine_deformed_edges,adaptive_remesh
            remesh_attempted_at=index
            try:
                try:
                    nx,nt,nb,ne,nv,nparent,report=refine_deformed_edges(p.positions(q),t,b,ep,V0,c,parent,X,p.L)
                except ValueError:
                    nx,nt,nb,ne,nv,nparent,report=adaptive_remesh(p.positions(q),t,b,ep,V0,c,parent=parent)
                macro=q[-6:].copy();newX=report.pop('_reference_points',None)
                if newX is None:
                    newX=nx@np.linalg.inv(deformation(macro)).T;report['reference_chart_reset']=True
                else:report['reference_chart_reset']=False
                np_new=Periodic(newX,p.L)
                if report['reference_chart_reset']:
                    nq=np.zeros(np_new.ndof);nq[-6:]=macro
                else:nq=np_new.remap_positions(nx,macro)
                np.testing.assert_allclose(np_new.positions(nq),nx,atol=1e-9)
                new_inverse=np.linalg.inv(geometry(nx,nt)[2])
                new_contact=PeriodicIPC(newX,nt,np_new,c)
                # Commit only after the transferred state has equilibrated.
                relax=equilibrium(np_new,nt,nq,nb,ne,new_inverse,nv,c,new_contact,nq[np_new.prescribed],np.zeros(np_new.ndof),log,nparent)
                if isinstance(relax,dict):raise ValueError('Post-remesh equilibrium failed: '+relax['reason'])
                nq,rs,_,_,_,_=relax
                p=np_new;X=newX;t=nt;q=nq;b=rs['b'];ep=rs['ep'];V0=nv;parent=nparent
                inverse=np.linalg.inv(rs['D']);zero=np.zeros(p.ndof);contact=new_contact;remesh_events+=1
                (out/f'remesh-{remesh_events:03d}.json').write_text(json.dumps(report,indent=2))
                log(dict(remesh=report))
                np.savez_compressed(out/f'remesh-{remesh_events:03d}.npz',X=X,x=p.positions(q),t=t,q=q,b=b,ep=ep,V0=V0,parent=parent)
            except ValueError as exc:
                log(dict(remesh_rejected=str(exc)))
                (out/f'remesh-rejected-{index:04d}.txt').write_text(str(exc))
        if phase=='compression':
            remaining=c['maximum_axial_compression']+q[p.prescribed]
            if remaining<=c.get('implicit_macro_tolerance',1e-6):
                (out/'status.json').write_text(json.dumps(dict(status='compression_limit_reached_target_not_reached',initial_porosity=initial,last_converged_porosity=phi_now(),target_porosity=c['target_final_porosity'],target_reached=False,validated=False),indent=2));return
            increment=min(increment,remaining)
            if len(rows)>1 and phi_now()<.2:
                slope=(rows[-2]['porosity']-rows[-1]['porosity'])/max(rows[-1]['compression']-rows[-2]['compression'],1e-10)
                if slope>0:increment=min(increment,max(c['implicit_minimum_increment'],.9*(phi_now()-c['target_final_porosity'])/slope))
            prescribed=q[p.prescribed]-increment;external=zero.copy()
        elif phase=='hold':prescribed=q[p.prescribed];external=zero.copy()
        else:
            prescribed=None;external=zero.copy();external[p.prescribed]=unload_reaction*(1-min(1.,unload_fraction+increment))
        if warm_guess is not None and (not np.array_equal(X,warm_X) or not np.array_equal(t,warm_t)):
            log(dict(initial_iterate_discarded='Accepted remesh changed its topology'));warm_guess=None
        solution=equilibrium(p,t,q,b,ep,inverse,V0,c,contact,prescribed,external,log,parent,initial_guess=warm_guess)
        warm_guess=None
        if isinstance(solution,dict):
            failed+=1;np.savez_compressed(out/f'failed-{failed:04d}.npz',q=solution['iterate'],X=X,t=t,oldq=q,b=b,ep=ep)
            log(dict(rejected_stage=index+1,reason=solution['reason'],increment=increment));increment*=.5
            if increment<c['implicit_minimum_increment']:raise RuntimeError('Minimum increment reached; rejected iterates are NOT solutions')
            continue
        q,state,g,n,it,norm=solution;b=state['b'];ep=state['ep'];inverse=np.linalg.inv(state['D']);index+=1
        phi=phi_now();H=deformation(q[-6:]);area=p.L**2*np.linalg.norm(np.cross(H[:,0],H[:,1]))
        _,free_mask=active_dofs(p,c,prescribed)
        scaled=g-external;scaled[-6:]/=p.L
        l2=float(np.linalg.norm(scaled[free_mask]))
        macro_error=0. if prescribed is None else abs(q[p.prescribed]-prescribed)
        row=dict(transverse_boundary=c.get('transverse_boundary','stress_free'),lateral_length_x_mm=float(p.L*np.linalg.norm(H[:,0])),lateral_length_y_mm=float(p.L*np.linalg.norm(H[:,1])),transverse_area_ratio=float(area/p.L**2),lateral_reaction_x_N=float(g[p.nmicro]/p.L),lateral_reaction_y_N=float(g[p.nmicro+1]/p.L),macro_target_error=macro_error,step=index,phase=phase,porosity=phi,compression=float(-q[p.prescribed]),axial_reaction_N=float(g[p.prescribed]/p.L),nominal_pressure_MPa=float(-g[p.prescribed]/p.L**3),free_residual_N=norm,free_residual_l2_N=l2,residual_l2_to_axial_reaction=l2/max(abs(g[p.prescribed])/p.L,1e-12),history_points=len(parent),contacts=n,minimum_quality=float(quality(p.positions(q),t).min()),maximum_peeq=float(ep.max()),iterations=it,wall_s=time.monotonic()-start)
        rows.append(row)
        with (out/'history.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=row.keys());w.writeheader();w.writerows(rows)
        np.savez_compressed(out/f'state-{index:04d}.npz',X=X,x=p.positions(q),t=t,q=q,b=b,ep=ep,V0=V0,parent=parent)
        log(row)
        (out/'status.json').write_text(json.dumps(dict(status='running',phase=phase,initial_porosity=initial,last_converged_porosity=phi,validated=False),indent=2))
        if phase=='compression' and phi<=c['target_final_porosity']+c['porosity_tolerance']:
            if c.get('loading_only',False):
                (out/'status.json').write_text(json.dumps(dict(status='loaded_target_reached',initial_porosity=float(initial),loaded_porosity=phi,target_porosity=c['target_final_porosity'],target_reached=True,validated=False),indent=2));return
            phase='hold';loaded_porosity=phi
        elif phase=='hold':
            phase='unload';loaded_porosity=phi;unload_reaction=float(g[p.prescribed]);increment=.05
        elif phase=='unload':
            unload_fraction=min(1.,unload_fraction+increment)
            if unload_fraction>=1:
                (out/'status.json').write_text(json.dumps(dict(status='cycle_completed' if loaded_porosity<=c['target_final_porosity']+c['porosity_tolerance'] else 'diagnostic_cycle_completed_target_not_reached',initial_porosity=initial,loaded_porosity=loaded_porosity,recovered_porosity=phi,recovery_boundary=c.get('transverse_boundary','stress_free'),recovery_load='zero_axial_load',target_porosity=c['target_final_porosity'],target_reached=bool(loaded_porosity<=c['target_final_porosity']+c['porosity_tolerance']),validated=False),indent=2));return
        if phase=='compression' and -q[p.prescribed]>=c['maximum_axial_compression']-c.get('implicit_macro_tolerance',1e-6):
            (out/'status.json').write_text(json.dumps(dict(status='compression_limit_reached_target_not_reached',initial_porosity=initial,last_converged_porosity=phi,target_porosity=c['target_final_porosity'],target_reached=False,validated=False),indent=2));return
        increment=min(.1 if phase=='unload' else c['implicit_strain_increment'],increment*1.25)
    raise RuntimeError('Stage budget exhausted')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--mesh',required=True);a.add_argument('--out',required=True);a.add_argument('--restart');a.add_argument('--initial-iterate');args=a.parse_args()
    try:run(args.config,args.mesh,args.out,args.restart,args.initial_iterate)
    except FileExistsError:
        raise
    except BaseException:
        if Path(args.out).is_dir():
            with Path(args.out,'failure.txt').open('x') as stream:stream.write(traceback.format_exc())
        raise
