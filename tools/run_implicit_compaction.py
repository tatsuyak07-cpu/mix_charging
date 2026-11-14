"""Quasi-static full-path finite-strain packing experiment (not certified).

All accepted states satisfy the reduced force residual. Failed iterates are
saved separately and are never used as the next material history.
"""
from packing_fem.porosity_control import limited_increment
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



def periodic_model(X,c):
    if c.get('rigid_inclusion',False):
        from packing_fem.rigid_inclusion import make_periodic
        return make_periodic(X,c['length_mm'],c)
    return Periodic(X,c['length_mm'])


def contact_model(X,t,p,c):
    if c.get('rigid_inclusion',False):
        from packing_fem.rigid_inclusion import make_contact
        return make_contact(X,t,p,c)
    return PeriodicIPC(X,t,p,c)


def measured_porosity(volume,cell,c):
    if c.get('rigid_inclusion',False):
        from packing_fem.rigid_inclusion import porosities
        return porosities(volume,cell,c)[0]
    return float(1-volume/cell)


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
    if c.get('implicit_optimizer')=='lagged_coupled':
        from packing_fem.quasistatic_hybrid import equilibrium as hybrid_equilibrium
        return hybrid_equilibrium(p,t,q,b,ep,inverse,V0,c,contact,prescribed,external,log,parent,initial_guess)
    if c.get('implicit_optimizer','newton') in ('lbfgs','lagged_lbfgs'):
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
                        d=np.zeros(p.ndof);d[free]=block_solve(Kf+diags(diagonal*c['implicit_regularization']),-g[free],p.nmicro,c)
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


def run(config,mesh,out,restart=None,initial_iterate=None,rejected_geometry=None):
    c=json.loads(Path(config).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(c,indent=2))
    from packing_fem.sparse_factor import solver_metadata
    (out/'linear-solver.json').write_text(json.dumps(solver_metadata(c),indent=2)+'\n')
    if c.get('implicit_optimizer','newton') not in ('newton','lbfgs','coupled_newton','lagged_lbfgs','lagged_coupled'):raise ValueError('Unknown implicit optimizer')
    if c.get('rigid_inclusion',False) and c.get('implicit_optimizer') not in ('coupled_newton','lagged_lbfgs','lagged_coupled'):raise ValueError('Floating rigid inclusion requires a nonlinear rigid-capable backend')
    if c.get('implicit_optimizer') in ('lbfgs','coupled_newton','lagged_lbfgs','lagged_coupled') and (c.get('transverse_boundary')!='fixed_cell' or not c.get('loading_only',False)):raise ValueError('Direct-displacement development backends require fixed transverse vectors and compression-only scope')
    if c.get('implicit_iterate_checkpoint_every',0)>0:
        trace=out/'iterate-trace';trace.mkdir();c['_iterate_checkpoint_directory']=str(trace)
    sources=out/'source';sources.mkdir();source_root=Path(__file__).resolve().parents[1]
    manifest={}
    for name in ['tools/run_implicit_compaction.py','packing_fem/fixed_point.py','packing_fem/rigid_inclusion.py','packing_fem/load_predictor.py','packing_fem/line_search.py','packing_fem/quasistatic_lbfgs.py','packing_fem/quasistatic_coupled.py','packing_fem/quasistatic_hybrid.py','packing_fem/implicit_material.py','packing_fem/ipc_periodic.py','packing_fem/implicit_remesh.py','packing_fem/implicit_linear.py','packing_fem/sparse_factor.py','packing_fem/porosity_control.py','packing_fem/explicit_material.py','packing_fem/explicit_periodic.py']:
        source=source_root/name;target=sources/source.name;shutil.copy2(source,target);manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    (out/'source-sha256.json').write_text(json.dumps(manifest,indent=2))
    (out/'inputs.json').write_text(json.dumps(dict(mesh=str(mesh),mesh_sha256=hashlib.sha256(Path(mesh).read_bytes()).hexdigest(),restart=None if restart is None else str(restart),initial_iterate=None if initial_iterate is None else str(initial_iterate),initial_iterate_sha256=None if initial_iterate is None else hashlib.sha256(Path(initial_iterate).read_bytes()).hexdigest(),iterate_is_initial_guess_only=True,rejected_geometry=None if rejected_geometry is None else str(rejected_geometry),rejected_geometry_sha256=None if rejected_geometry is None else hashlib.sha256(Path(rejected_geometry).read_bytes()).hexdigest(),rejected_geometry_material_history_used=False),indent=2))
    data=np.load(mesh);X=data['points'];t=oriented(X,data['tets'])
    if restart:
        r=np.load(restart);X=r['X'];t=oriented(X,r['t'])
    p=periodic_model(X,c);q=np.zeros(p.ndof);b=np.tile(np.eye(3),(len(t),1,1));ep=np.zeros(len(t))
    V0,_,D=geometry(X,t);inverse=np.linalg.inv(D);parent=np.arange(len(t))
    if restart:
        r=np.load(restart)
        if abs(r['V0'].sum()/geometry(data['points'],oriented(data['points'],data['tets']))[0].sum()-1)>1e-6:raise ValueError('Restart material volume mismatch')
        q=r['q'];b=r['b'];ep=r['ep'];V0=r['V0'];parent=r['parent'] if 'parent' in r else np.arange(len(t));inverse=np.linalg.inv(geometry(p.positions(q),t)[2])
    previous_path=c.get('implicit_previous_accepted_state')
    if c.get('implicit_slip_predictor',False) and previous_path:
        with np.load(previous_path) as previous:
            c['_previous_q']=previous['q'].copy();c['_previous_X']=previous['X'].copy();c['_previous_t']=previous['t'].copy()
        (out/'slip-predictor-input.json').write_text(json.dumps(dict(source=previous_path,sha256=hashlib.sha256(Path(previous_path).read_bytes()).hexdigest(),numerical_initial_guess_only=True),indent=2)+'\n')
    initial_mesh_porosity=measured_porosity(geometry(data['points'],oriented(data['points'],data['tets']))[0].sum(),p.L**3,c)
    if 'initial_porosity_tolerance' in c and abs(initial_mesh_porosity-c.get('requested_matrix_porosity',c['requested_initial_porosity']))>c['initial_porosity_tolerance']:raise ValueError('Measured mesh porosity outside the configured initial geometry tolerance')
    contact=contact_model(X,t,p,c)
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
    initial=initial_mesh_porosity
    rows=[];start=time.monotonic();increment=c.get('implicit_initial_strain_increment',c['implicit_strain_increment']);index=0;failed=0
    if not 0<increment<=c['implicit_strain_increment']:raise ValueError('Initial load increment must be positive and no larger than its ceiling')
    last_porosity_slope=None
    remesh_events=0;remesh_attempted_at=-1;predictor_index=0;pending_geometry=None
    if rejected_geometry:
        if not c.get('implicit_bisection_predictor',False):raise ValueError('Rejected geometry requires the bisection predictor')
        with np.load(rejected_geometry) as g:
            if not np.array_equal(g['X'],X) or not np.array_equal(g['t'],t) or g['q'].shape!=q.shape or not np.all(np.isfinite(g['q'])):raise ValueError('Rejected geometry has a different periodic chart')
            pending_geometry=dict(q=g['q'].copy(),X=X.copy(),t=t.copy(),source=str(rejected_geometry))
    phase=c.get('initial_phase','compression')
    if phase not in ('compression','hold'):raise ValueError('initial_phase must be compression or hold')
    zero=np.zeros(p.ndof);unload_fraction=0.;unload_reaction=None;loaded_porosity=None
    def log(v):print(json.dumps(v),flush=True)
    def phi_now():return measured_porosity(geometry(p.positions(q),t)[0].sum(),p.L**3*np.linalg.det(deformation(q[-6:])),c)
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
                rigid_pose=q[p.rigid_dofs].copy() if hasattr(p,'rigid_dofs') else None
                macro=q[-6:].copy();newX=report.pop('_reference_points',None)
                if newX is None:
                    newX=nx@np.linalg.inv(deformation(macro)).T;report['reference_chart_reset']=True
                else:report['reference_chart_reset']=False
                np_new=periodic_model(newX,c)
                if report['reference_chart_reset']:
                    nq=np.zeros(np_new.ndof);nq[-6:]=macro
                else:nq=np_new.remap_positions(nx,macro)
                if rigid_pose is not None:
                    nq[np_new.rigid_dofs]=rigid_pose
                    old_center,(old_rotation,_,_)=p.pose(q);new_center,(new_rotation,_,_)=np_new.pose(nq)
                    np.testing.assert_allclose(old_center,new_center,atol=1e-10);np.testing.assert_allclose(old_rotation,new_rotation,atol=1e-12)
                    report['rigid_pose_preserved']=True
                np.testing.assert_allclose(np_new.positions(nq),nx,atol=1e-9)
                new_inverse=np.linalg.inv(geometry(nx,nt)[2])
                new_contact=contact_model(newX,nt,np_new,c)
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
            if pending_geometry is not None:
                if not np.array_equal(X,pending_geometry['X']) or not np.array_equal(t,pending_geometry['t']):
                    log(dict(rejected_geometry_discarded='Accepted remesh changed the chart'));pending_geometry=None
                else:
                    gap=q[p.prescribed]-pending_geometry['q'][p.prescribed]
                    if gap<=c.get('implicit_macro_tolerance',1e-6):pending_geometry=None
                    # The rejected load is not a required output station; keep
                    # the original adaptive schedule even when advancing beyond it.
            remaining=c['maximum_axial_compression']+q[p.prescribed]
            if remaining<=c.get('implicit_macro_tolerance',1e-6):
                (out/'status.json').write_text(json.dumps(dict(status='compression_limit_reached_target_not_reached',initial_porosity=initial,last_converged_porosity=phi_now(),target_porosity=c['target_final_porosity'],target_reached=False,validated=False),indent=2));return
            increment=min(increment,remaining)
            increment=limited_increment(increment,phi_now(),-q[p.prescribed],c['target_final_porosity'],c['implicit_minimum_increment'],last_porosity_slope)
            prescribed=q[p.prescribed]-increment;external=zero.copy()
        elif phase=='hold':prescribed=q[p.prescribed];external=zero.copy()
        else:
            prescribed=None;external=zero.copy();external[p.prescribed]=unload_reaction*(1-min(1.,unload_fraction+increment))
        if warm_guess is not None and (not np.array_equal(X,warm_X) or not np.array_equal(t,warm_t)):
            log(dict(initial_iterate_discarded='Accepted remesh changed its topology'));warm_guess=None
        if warm_guess is None and phase=='compression' and pending_geometry is not None:
            try:
                from packing_fem.load_predictor import bisected_initial_guess
                warm_guess,report=bisected_initial_guess(p,t,q,pending_geometry['q'],prescribed,b,ep,inverse,V0,parent,c,contact)
                warm_X=X.copy();warm_t=t.copy();predictor_index+=1
                np.savez_compressed(out/f'predictor-{predictor_index:04d}.npz',q=warm_guess,X=X,t=t,oldq=q,b=b,ep=ep)
                report.update(source_geometry=pending_geometry['source'],accepted_predecessor_step=index,target_compression=float(-prescribed))
                (out/f'predictor-{predictor_index:04d}.json').write_text(json.dumps(report,indent=2)+'\n')
                log(dict(load_bisection_predictor=report))
            except (ValueError,RuntimeError) as exc:log(dict(load_bisection_predictor_rejected=str(exc)));warm_guess=None
        old_phi=phi_now();old_compression=float(-q[p.prescribed])
        solution=equilibrium(p,t,q,b,ep,inverse,V0,c,contact,prescribed,external,log,parent,initial_guess=warm_guess)
        warm_guess=None
        if isinstance(solution,dict):
            failed+=1;np.savez_compressed(out/f'failed-{failed:04d}.npz',q=solution['iterate'],X=X,t=t,oldq=q,b=b,ep=ep)
            log(dict(rejected_stage=index+1,reason=solution['reason'],increment=increment));increment*=.5
            if increment<c['implicit_minimum_increment']:raise RuntimeError('Minimum increment reached; rejected iterates are NOT solutions')
            if phase=='compression' and c.get('implicit_bisection_predictor',False):
                pending_geometry=dict(q=solution['iterate'].copy(),X=X.copy(),t=t.copy(),source=str(out/f'failed-{failed:04d}.npz'))
            continue
        if c.get('implicit_slip_predictor',False):
            c['_previous_q']=q.copy();c['_previous_X']=X.copy();c['_previous_t']=t.copy()
            np.savez_compressed(out/f'slip-predictor-{index+1:04d}.npz',q=q,X=X,t=t)
        q,state,g,n,it,norm=solution;b=state['b'];ep=state['ep'];inverse=np.linalg.inv(state['D']);index+=1
        phi=phi_now()
        if phase=='compression':last_porosity_slope=(old_phi-phi)/max(-q[p.prescribed]-old_compression,1e-10)
        H=deformation(q[-6:]);area=p.L**2*np.linalg.norm(np.cross(H[:,0],H[:,1]))
        _,free_mask=active_dofs(p,c,prescribed)
        scaled=g-external;scaled[-6:]/=p.L
        l2=float(np.linalg.norm(scaled[free_mask]))
        macro_error=0. if prescribed is None else abs(q[p.prescribed]-prescribed)
        row=dict(transverse_boundary=c.get('transverse_boundary','stress_free'),lateral_length_x_mm=float(p.L*np.linalg.norm(H[:,0])),lateral_length_y_mm=float(p.L*np.linalg.norm(H[:,1])),transverse_area_ratio=float(area/p.L**2),lateral_reaction_x_N=float(g[p.nmicro]/p.L),lateral_reaction_y_N=float(g[p.nmicro+1]/p.L),macro_target_error=macro_error,step=index,phase=phase,porosity=phi,compression=float(-q[p.prescribed]),axial_reaction_N=float(g[p.prescribed]/p.L),nominal_pressure_MPa=float(-g[p.prescribed]/p.L**3),free_residual_N=norm,free_residual_l2_N=l2,residual_l2_to_axial_reaction=l2/max(abs(g[p.prescribed])/p.L,1e-12),history_points=len(parent),contacts=n,minimum_quality=float(quality(p.positions(q),t).min()),maximum_peeq=float(ep.max()),iterations=it,wall_s=time.monotonic()-start)
        if hasattr(p,'rigid_dofs'):
            from packing_fem.rigid_inclusion import porosities,rigid_volume
            center,(rotation,_,_)=p.pose(q);force_rigid,torque_rigid=contact.rigid_resultant(q)
            _,total=porosities(geometry(p.positions(q),t)[0].sum(),p.L**3*np.linalg.det(H),c)
            row.update(total_porosity=total,rigid_contacts=contact.rigid_contact_count(),rigid_volume_mm3=rigid_volume(c),rigid_center_x_mm=float(center[0]),rigid_center_y_mm=float(center[1]),rigid_center_z_mm=float(center[2]),rigid_force_N=float(np.linalg.norm(force_rigid)),rigid_torque_Nmm=float(np.linalg.norm(torque_rigid)),rigid_rotation_rad=float(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))))
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
        growth=1.25
        if phase=='compression' and c.get('implicit_target_iterations') is not None:
            if c['implicit_target_iterations']<=0:raise ValueError('Target nonlinear iteration count must be positive')
            growth=float(np.clip(np.sqrt(c['implicit_target_iterations']/(it+1)),.5,1.25))
        increment=min(.1 if phase=='unload' else c['implicit_strain_increment'],max(c['implicit_minimum_increment'],increment*growth))
    raise RuntimeError('Stage budget exhausted')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--mesh',required=True);a.add_argument('--out',required=True);a.add_argument('--restart');a.add_argument('--initial-iterate');a.add_argument('--rejected-geometry');args=a.parse_args()
    try:run(args.config,args.mesh,args.out,args.restart,args.initial_iterate,args.rejected_geometry)
    except FileExistsError:
        raise
    except BaseException as exc:
        if Path(args.out).is_dir():
            with Path(args.out,'failure.txt').open('x') as stream:stream.write(traceback.format_exc())
            status_path=Path(args.out,'status.json')
            status=json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update(status='failed',failure_reason=str(exc),target_reached=False,validated=False)
            status_path.write_text(json.dumps(status,indent=2)+'\n')
        raise
