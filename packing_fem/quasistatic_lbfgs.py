"""Fixed-cell solid FEM minimization; exact axial Dirichlet control, unchanged laws."""
from packing_fem.sparse_factor import factorize
import numpy as np
import time
from pathlib import Path
from scipy.sparse import diags
from .implicit_material import evaluate
from .fixed_point import affine_predictor,anderson_candidate
from .line_search import bounded_wolfe


def slip_predictor(p,t,oldq,affine,b,ep,inverse,V0,c,contact,parent,external,log):
    """Use previous accepted slip only as a feasible numerical initial guess."""
    previous=c.get('_previous_q');previous_X=c.get('_previous_X');previous_t=c.get('_previous_t')
    if previous is None:return affine
    if previous.shape!=oldq.shape or not np.array_equal(previous_X,p.X) or not np.array_equal(previous_t,t):
        log(dict(slip_predictor_discarded='Reference chart or topology changed'));return affine
    correction=oldq-previous;correction[-6:]=0.;best=None;records=[]
    for blend in [0.,1.,.5]:
        direction=blend*correction
        try:
            bound=1. if blend==0 else contact.ccd(affine,direction);alpha=1. if bound>=1 else .9*bound
            candidate=affine+alpha*direction
            _,gm,_,_=evaluate(p.positions(candidate),t,b,ep,inverse,V0,c,parent=parent)
            _,gc,_,_=contact.evaluate_coupled(candidate,oldq)
            gradient=(np.asarray(p.A.T@gm.ravel()).ravel()+gc-external)[:p.nmicro];norm=float(np.linalg.norm(gradient))
            records.append(dict(blend=blend,ccd_fraction=alpha,residual_l2_N=norm,maximum_N=float(abs(gradient).max())))
            if np.isfinite(norm) and (best is None or norm<best[0]):best=(norm,candidate.copy(),blend)
        except (ValueError,RuntimeError) as exc:records.append(dict(blend=blend,rejected=str(exc)))
    if best is None:raise ValueError('No admissible slip-history initial guess')
    log(dict(slip_predictor=True,selected_blend=best[2],accepted_material_history_unchanged=True,candidates=records));return best[1]


def equilibrium(p,t,q,b,ep,inverse,V0,c,contact,prescribed,external,log,parent=None,initial_guess=None):
    oldq=q.copy();trial=q.copy() if initial_guess is None else initial_guess.copy()
    if prescribed is None or c.get('transverse_boundary')!='fixed_cell':return dict(reason='L-BFGS requires prescribed axial compression and fixed transverse cell',iterate=trial)
    seed=trial.copy()
    try:
        trial=affine_predictor(p,trial,prescribed)
        if hasattr(contact,'loading_path_clear') and seed[p.prescribed]!=prescribed and not contact.loading_path_clear(seed,trial):return dict(reason='Rigid loading predictor has no collision-free path',iterate=trial)
    except (ValueError,RuntimeError) as exc:return dict(reason=str(exc),iterate=trial)
    macro=trial[-6:].copy()
    if np.any(abs(macro[[0,1,3,4,5]])>1e-12):return dict(reason='Fixed transverse lattice changed',iterate=trial)
    memory=c.get('implicit_lbfgs_memory',10);refresh=c.get('implicit_lbfgs_refresh',25)
    if int(memory)!=memory or memory<0:return dict(reason='L-BFGS memory must be a nonnegative integer',iterate=trial)
    memory=int(memory)
    max_iterations=c.get('implicit_lbfgs_max_iterations',500);tol=c['implicit_force_tolerance_N']
    inexact_updates=c.get('implicit_lbfgs_inexact_lag_updates',0)
    if int(inexact_updates)!=inexact_updates or inexact_updates<0:return dict(reason='Inexact lag update budget must be a nonnegative integer',iterate=trial)
    inner_budget=min(max_iterations,int(inexact_updates)) if inexact_updates else max_iterations
    inner_fraction=c.get('implicit_lag_inner_tolerance_fraction',1.)
    if not 0<inner_fraction<=1:return dict(reason='Inner force tolerance fraction must be in(0,1]',iterate=trial)
    inner_tol=tol*inner_fraction
    if c.get('implicit_slip_predictor',False) and initial_guess is None:
        try:trial=slip_predictor(p,t,oldq,trial,b,ep,inverse,V0,c,contact,parent,external,log)
        except (ValueError,RuntimeError) as exc:return dict(reason=str(exc),iterate=trial)
    depth=c.get('implicit_anderson_depth',0);outputs=[];residuals=[];lag_input=trial.copy()
    displacement_scale=np.ones(p.ndof);displacement_scale[-6:]=p.L
    def values(v):
        e,g,_,state=evaluate(p.positions(v),t,b,ep,inverse,V0,c,parent=parent)
        ce,cg,_,n=contact.evaluate(v)
        return e+ce-float(external@v),np.asarray(p.A.T@g.ravel()).ravel()+cg,state,n
    total_updates=0;last_checkpoint_updates=0
    try:
        contact.set_lag(trial,oldq)
        _,initial_internal,initial_state,initial_n=values(trial);initial_norm=float(abs((initial_internal-external)[:p.nmicro]).max())
        if initial_norm<tol and (not hasattr(contact,'force_balance_ok') or contact.force_balance_ok(trial,tol)):return trial,initial_state,initial_internal,initial_n,0,initial_norm
        for lag in range(c['implicit_max_lag']):
            pairs=[];factor=None;factor_contacts=None;factor_norm=None;energy,internal,state,n=values(trial)
            for it in range(inner_budget):
                gradient=(internal-external)[:p.nmicro];norm=float(np.max(abs(gradient)))
                log(dict(lag=lag,lbfgs_iteration=it,total_updates=total_updates,compression=float(-trial[p.prescribed]),residual_N=norm,contacts=n,memory=len(pairs)))
                checkpoint_every=int(c.get('implicit_iterate_checkpoint_every',0))
                if checkpoint_every>0 and total_updates-last_checkpoint_updates>=checkpoint_every and c.get('_iterate_checkpoint_directory'):
                    checkpoint=Path(c['_iterate_checkpoint_directory'])/f'iterate-{time.time_ns()}-{total_updates:04d}.npz'
                    np.savez_compressed(checkpoint,q=trial,X=p.X,t=t,oldq=oldq,b=b,ep=ep)
                    last_checkpoint_updates=total_updates
                    log(dict(iterate_checkpoint=str(checkpoint),iterate_is_solution=False))
                if norm<inner_tol:break
                changed=(factor_contacts is not None and abs(n-factor_contacts)>=max(1,c.get('implicit_lbfgs_contact_refresh_fraction',.1)*max(factor_contacts,1)))
                growth=(factor_norm is not None and norm>c.get('implicit_lbfgs_gradient_refresh_ratio',5.)*factor_norm)
                if factor is None or it%refresh==0 or changed or growth:
                    if changed or growth:pairs=[]
                    log(dict(lbfgs_preconditioner_update=True,contacts=n,previous_contacts=factor_contacts,contact_change=changed,residual_growth=growth))
                    factor=None  # Release the old sparse LU before building another.
                    _,_,Km,_=evaluate(p.positions(trial),t,b,ep,inverse,V0,dict(c,implicit_project_material_tangent=True),True,parent)
                    _,_,Kc,_=contact.evaluate(trial,True,project_hessian=True)
                    K=(p.A.T@Km@p.A+Kc)[:p.nmicro,:p.nmicro].tocsc()
                    diagonal=np.maximum(abs(K.diagonal()),1e-6)
                    if hasattr(p,'rigid_dofs'):diagonal[p.rigid_dofs]=np.maximum(diagonal[p.rigid_dofs],c.get('implicit_rigid_step_scale_floor_N_per_mm',0.))
                    factor=factorize(K+diags(diagonal*c['implicit_regularization']),c)
                    factor_contacts=n;factor_norm=norm
                    del K,Km,Kc
                vector=gradient.copy();coefficients=[]
                for s,y,rho in reversed(pairs):
                    a=rho*float(s@vector);coefficients.append(a);vector-=a*y
                direction=factor.solve(vector)
                for (s,y,rho),a in zip(pairs,reversed(coefficients)):
                    direction+=s*(a-rho*float(y@direction))
                dq=np.zeros(p.ndof);dq[:p.nmicro]=-direction
                if not np.all(np.isfinite(dq)) or not np.isfinite(float(gradient@dq[:p.nmicro])) or float(gradient@dq[:p.nmicro])>=0:
                    pairs=[];dq[:p.nmicro]=-factor.solve(gradient)
                slope=float(gradient@dq[:p.nmicro])
                if not np.all(np.isfinite(dq)) or not np.isfinite(slope) or slope>=0:raise ValueError('L-BFGS preconditioner did not produce descent')
                accepted=False;roundoff=32*np.finfo(float).eps*max(c['young_mpa']*float(V0.sum()),abs(energy),1e-30)
                wolfe_mode=c.get('implicit_lbfgs_line_search','armijo')=='strong_wolfe'
                if wolfe_mode:
                    mapping=contact.jacobian(trial) if hasattr(contact,'jacobian') else p.A
                    maximum=float(np.max(np.linalg.norm((mapping@dq).reshape(-1,3),axis=1)))
                    if not maximum>0:raise ValueError('Zero L-BFGS search direction')
                    search_length=c.get('implicit_lbfgs_search_step_mm',.05)
                    dq=(dq/maximum)*search_length;slope=float(gradient@dq[:p.nmicro])
                    extension=c.get('ipc_ccd_maximum_displacement_mm',.1)/search_length
                    bound=contact.ccd(trial,extension*dq);maximum_alpha=.9*extension*bound
                    def line_values(micro):
                        candidate=trial.copy();candidate[:p.nmicro]=micro
                        ve,vi,vs,vn=values(candidate)
                        return ve,(vi-external)[:p.nmicro],(candidate,vi,vs,vn)
                    result=bounded_wolfe(trial[:p.nmicro],dq[:p.nmicro],energy,gradient,line_values,maximum_alpha,roundoff)
                    if result is not None:
                        candidate,vi,vs,vn=result['payload'];ve=result['energy'];vg=result['gradient'];alpha=result['alpha'];accepted=True
                        log(dict(lbfgs_wolfe=result['wolfe'],alpha=alpha,energy_decrease_Nmm=energy-ve,line_search_warnings=result['warnings']))
                    else:alpha=min(1.,maximum_alpha)
                else:
                    bound=contact.ccd(trial,dq);alpha=1. if bound>=1 else .9*bound
                    if getattr(contact,'last_ccd_horizon',1.)<1:
                        log(dict(ccd_horizon=contact.last_ccd_horizon,proposed_max_displacement_mm=contact.last_ccd_max_displacement_mm))
                for ls in range(0 if accepted else 35):
                    candidate=trial+alpha*dq
                    try:
                        ve,vi,vs,vn=values(candidate);vg=(vi-external)[:p.nmicro]
                        if ve<=energy+1e-4*alpha*slope or (ve<=energy+roundoff and np.max(abs(vg))<norm*(1-1e-4*alpha)):
                            accepted=True;break
                    except (ValueError,RuntimeError):pass
                    alpha*=.5
                if not accepted:
                    if pairs:
                        pairs=[];factor=None;log(dict(lbfgs_reset='Line search rejected history direction'));continue
                    raise ValueError('L-BFGS line search failed')
                s=(candidate-trial)[:p.nmicro];y=vg-gradient;curvature=float(s@y)
                curvature_safe=not wolfe_mode or curvature>=.1*max(0.,-float(s@gradient))
                if curvature_safe and curvature>1e-12*np.linalg.norm(s)*np.linalg.norm(y) and curvature>0:
                    if memory:
                        pairs.append((s.copy(),y.copy(),1/curvature));pairs=pairs[-memory:]
                trial=candidate;energy=ve;internal=vi;state=vs;n=vn;total_updates+=1
            else:
                if not inexact_updates:raise ValueError('L-BFGS iteration budget exhausted')
                log(dict(inexact_lag_refresh=lag,inner_updates=inner_budget,inner_convergence_claimed=False))
            contact.set_lag(trial,oldq);_,internal,state,n=values(trial)
            norm=float(np.max(abs((internal-external)[:p.nmicro])))
            log(dict(lag_refreshed=lag,original_force_maximum_N=norm,original_force_l2_N=float(np.linalg.norm((internal-external)[:p.nmicro]))))
            if norm<tol and (not hasattr(contact,'force_balance_ok') or contact.force_balance_ok(trial,tol)):return trial,state,internal,n,total_updates,norm
            if depth>1:
                outputs.append(trial.copy());residuals.append(trial-lag_input)
                outputs=outputs[-depth:];residuals=residuals[-depth:]
                candidate=anderson_candidate(outputs,residuals,displacement_scale)
                if candidate is not None:
                    candidate[-6:]=macro;dq=candidate-trial
                    bound=contact.ccd(trial,dq);fraction=1. if bound>=1 else .9*bound;candidate=trial+fraction*dq
                    improved=False
                    try:
                        contact.set_lag(candidate,oldq);_,ci,cs,cn=values(candidate)
                        cnorm=float(np.max(abs((ci-external)[:p.nmicro])))
                        if cnorm<norm:
                            log(dict(anderson_lag=lag,residual_before_N=norm,residual_after_N=cnorm))
                            trial=candidate;improved=True;total_updates+=1
                            if cnorm<tol and (not hasattr(contact,'force_balance_ok') or contact.force_balance_ok(trial,tol)):return trial,cs,ci,cn,total_updates,cnorm
                    except (ValueError,RuntimeError):pass
                    if not improved:contact.set_lag(trial,oldq)
            lag_input=trial.copy()
        raise ValueError('L-BFGS friction lag iterations did not converge')
    except (ValueError,RuntimeError) as exc:return dict(reason=str(exc),iterate=trial,iterations=total_updates)
