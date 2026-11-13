"""Newton solve of the actual coupled solid/contact/Coulomb equilibrium residual."""
import numpy as np
from pathlib import Path
import time
from scipy.sparse import diags
from scipy.sparse.linalg import splu, lsmr
from .fixed_point import affine_predictor
from .implicit_material import evaluate
from .load_predictor import search_stalled


def smooth_peak_merit(force,power):
    """Squared p-norm and its exact force derivative, scaled against overflow."""
    if not np.isfinite(power) or power<2:raise ValueError('Peak norm power must be finite and >=2')
    maximum=float(np.max(abs(force),initial=0.))
    if maximum==0:return 0.,np.zeros_like(force)
    relative=abs(force)/maximum;total=float(np.sum(relative**power))
    return .5*maximum**2*total**(2/power),force*relative**(power-2)*total**(2/power-1)


def violation_merit(force,margin):
    """Squared distance to an inner force-tolerance box and its derivative."""
    excess=np.sign(force)*np.maximum(abs(force)-margin,0.)
    return .5*float(excess@excess),excess


def rigid_block_relaxation(p,contact,q,oldq,internal,J,external,measure,derivative,c,log):
    """Numerical block smoothing; never advances material or friction history."""
    ids=p.rigid_dofs;current=q.copy()
    solid=internal-np.asarray(contact.R.T@contact.last_full_gradient.ravel()).ravel()
    gradient=(internal-external)[:p.nmicro];initial=measure(gradient);accepted=0
    columns=J[:,ids];block=J[ids,:][:,ids].toarray()
    for iteration in range(c.get('implicit_rigid_block_iterations',8)):
        norm=float(abs(gradient).max());rigid_norm=float(abs(gradient[ids]).max())
        if rigid_norm<.8*norm:break
        if iteration:
            _,gc,K,_=contact.evaluate_coupled(current,oldq,True)
            gradient=(solid+gc-external)[:p.nmicro];columns=K[:p.nmicro,ids];block=K[ids,:][:,ids].toarray()
        direction=np.zeros(p.ndof);diagonal=np.maximum(abs(block.diagonal()),1e-6)
        direction[ids]=np.linalg.lstsq(block+np.diag(diagonal*c['implicit_regularization']),-gradient[ids],rcond=1e-12)[0]
        slope=float(derivative(gradient)@(columns@direction[ids]));merit=measure(gradient)
        if not np.all(np.isfinite(direction)) or not np.isfinite(slope) or slope>=0:break
        bound=contact.ccd(current,direction);alpha=1. if bound>=1 else .9*bound
        budget=max(float(gradient@gradient),(c.get('implicit_peak_rms_fraction',c.get('implicit_hybrid_rms_fraction',.2))*c['implicit_force_tolerance_N'])**2*len(gradient))
        for _ in range(20):
            try:
                candidate=current+alpha*direction;_,gc,_,_=contact.evaluate_coupled(candidate,oldq)
                cg=(solid+gc-external)[:p.nmicro];cm=measure(cg)
                if np.all(np.isfinite(cg)) and cm<=merit+1e-4*alpha*slope and cg@cg<=budget*(1+1e-10):break
            except (ValueError,RuntimeError):pass
            alpha*=.5
        else:break
        if cm>=merit*(1-1e-5):break
        current=candidate;gradient=cg;accepted+=1
        log(dict(rigid_block_update=iteration,maximum_force_N=float(abs(cg).max()),rigid_maximum_N=float(abs(cg[ids]).max()),body_maximum_N=float(abs(cg[:p.body_nmicro]).max()),alpha=alpha,material_history_unchanged=True))
    if accepted and measure(gradient)<initial*(1-1e-4):return current
    return None


def equilibrium(p,t,q,b,ep,inverse,V0,c,contact,prescribed,external,log,parent=None,initial_guess=None):
    multipliers=c.get('implicit_friction_smoothing_multipliers')
    if multipliers:
        if multipliers[-1]!=1 or any(m<1 for m in multipliers) or any(a<=b for a,b in zip(multipliers,multipliers[1:])):raise ValueError('Friction smoothing continuation must decrease to the original value')
        import ipctk as ipc
        original=contact.friction;guess=initial_guess
        try:
            for multiplier in multipliers:
                phase=dict(c);phase.pop('implicit_friction_smoothing_multipliers')
                phase['ipc_slip_regularization_mm']=c['ipc_slip_regularization_mm']*multiplier
                if multiplier>1:phase['implicit_coupled_max_iterations']=c.get('implicit_friction_smoothing_max_iterations',30)
                contact.friction=ipc.FrictionPotential(phase['ipc_slip_regularization_mm'])
                log(dict(friction_smoothing_multiplier=multiplier,trial_initialization_only=multiplier>1,accepted_material_history_unchanged=True))
                solution=equilibrium(p,t,q,b,ep,inverse,V0,phase,contact,prescribed,external,log,parent,guess)
                guess=solution['iterate'] if isinstance(solution,dict) else solution[0]
                if isinstance(solution,dict):log(dict(friction_smoothing_phase_failed=multiplier,reason=solution['reason'],iterate_used_only_as_initial_guess=True))
            return solution
        finally:contact.friction=original
    oldq=q.copy();trial=q.copy() if initial_guess is None else initial_guess.copy()
    if prescribed is None or c.get('transverse_boundary')!='fixed_cell':return dict(reason='Coupled Newton requires prescribed axial compression and fixed transverse cell',iterate=trial)
    seed=trial.copy()
    try:
        trial=affine_predictor(p,trial,prescribed)
        if hasattr(contact,'loading_path_clear') and seed[p.prescribed]!=prescribed and not contact.loading_path_clear(seed,trial):return dict(reason='Rigid loading predictor has no collision-free loading path; cut increment',iterate=trial)
    except (ValueError,RuntimeError) as exc:return dict(reason=str(exc),iterate=trial)
    if np.any(abs(trial[-6:][[0,1,3,4,5]])>1e-12):return dict(reason='Fixed transverse lattice changed',iterate=trial)
    def values(v,tangent=False):
        _,g,Km,state=evaluate(p.positions(v),t,b,ep,inverse,V0,dict(c,implicit_project_material_tangent=False),tangent,parent)
        _,gc,Kc,n=contact.evaluate_coupled(v,oldq,tangent)
        internal=np.asarray(p.A.T@g.ravel()).ravel()+gc
        if not np.all(np.isfinite(internal)):raise ValueError('Nonfinite coupled force')
        J=None if not tangent else (p.A.T@Km@p.A+Kc)[:p.nmicro,:p.nmicro].tocsc()
        return internal,state,n,J
    norm_mode=c.get('implicit_coupled_merit_norm','l2');maximum_stage=False;norm_history=[];l2_history=[];step_history=[]
    violation_fraction=c.get('implicit_peak_violation_fraction');violation_margin=None
    if violation_fraction is not None:
        if not 0<violation_fraction<1:return dict(reason='Inner force-box fraction must be between0 and1',iterate=trial)
        violation_margin=violation_fraction*c['implicit_force_tolerance_N']
        if not 0<=c.get('implicit_force_box_inactive_weight',0.)<=1:return dict(reason='Inactive force-box search weight must be between0 and1',iterate=trial)
    if norm_mode not in ('l2','maximum','hybrid'):raise ValueError('Unknown residual merit norm')
    try:
        for it in range(c.get('implicit_coupled_max_iterations',c['implicit_max_newton'])):
            checkpoint_every=int(c.get('implicit_iterate_checkpoint_every',0))
            if checkpoint_every>0 and it>0 and it%checkpoint_every==0 and c.get('_iterate_checkpoint_directory'):
                checkpoint=Path(c['_iterate_checkpoint_directory'])/f'iterate-{time.time_ns()}-{it:04d}.npz'
                np.savez_compressed(checkpoint,q=trial,X=p.X,t=t,oldq=oldq,b=b,ep=ep)
                log(dict(iterate_checkpoint=str(checkpoint),iterate_is_solution=False))
            internal,state,n,J=values(trial,True);gradient=(internal-external)[:p.nmicro]
            norm=float(np.max(abs(gradient),initial=0.))
            weighted=c.get('implicit_coupled_merit_weighting','none')=='diagonal'
            weights=1/np.maximum(abs(J.diagonal()),1e-6) if weighted else np.ones_like(gradient)
            if norm_mode=='hybrid' and np.linalg.norm(gradient)<=c.get('implicit_hybrid_rms_fraction',.2)*c['implicit_force_tolerance_N']*np.sqrt(len(gradient)):maximum_stage=True
            maximum_merit=norm_mode=='maximum' or maximum_stage
            violation_mode=maximum_merit and violation_fraction is not None
            peak_power=c.get('implicit_peak_norm_power') if maximum_merit else None
            def measure(force):
                if violation_mode:return violation_merit(force,violation_margin)[0]
                if peak_power is not None:return smooth_peak_merit(force,peak_power)[0]
                return .5*float(np.max(abs(force),initial=0.)**2) if maximum_merit else .5*float(force@(weights*force))
            def derivative(force):
                if violation_mode:return violation_merit(force,violation_margin)[1]
                if peak_power is not None:return smooth_peak_merit(force,peak_power)[1]
                if maximum_merit:
                    active=abs(force)>=np.max(abs(force),initial=0.)*(1-1e-12)
                    return np.where(active,force/max(int(active.sum()),1),0.)
                return weights*force
            merit=measure(gradient)
            merit_force=derivative(gradient)
            log(dict(coupled_newton=it,compression=float(-trial[p.prescribed]),residual_N=norm,residual_l2_N=float(np.linalg.norm(gradient)),contacts=n,search_merit_norm='inner_force_box' if violation_mode else (f'p{peak_power}' if peak_power is not None else ('maximum' if maximum_merit else 'l2'))))
            if norm<c['implicit_force_tolerance_N'] and (not hasattr(contact,'force_balance_ok') or contact.force_balance_ok(trial,c['implicit_force_tolerance_N'])):return trial,state,internal,n,it,norm
            if violation_mode and merit==0:
                violation_margin*=.5;log(dict(inner_force_margin_tightened_N=violation_margin,reason='Original physical rigid balance still governs acceptance'));continue
            norm_history.append(norm);l2_history.append(float(np.linalg.norm(gradient)))
            if c.get('implicit_stagnation_window',0)>0 and search_stalled(norm_history,step_history,c['implicit_stagnation_window']) and search_stalled(l2_history,step_history,c['implicit_stagnation_window']):raise ValueError('Force residual stalled with negligible displacement updates; reduce load increment')
            diagonal=np.maximum(abs(J.diagonal()),1e-6)
            merit_gradient=np.asarray(J.T@merit_force).ravel()
            if maximum_merit and c.get('implicit_rigid_block_relaxation',False) and hasattr(p,'rigid_dofs') and norm<4*c['implicit_force_tolerance_N']:
                candidate=rigid_block_relaxation(p,contact,trial,oldq,internal,J,external,measure,derivative,c,log)
                if candidate is not None:
                    step_history.append(float(abs(candidate[:p.nmicro]-trial[:p.nmicro]).max()));trial=candidate;continue
            # Nonlinear smoothing of residual peaks. Solve the SAME force
            # Jacobian on a small neighborhood, keeping far-field increments
            # zero. Both maximum force and full residual must decrease.
            if c.get('implicit_coupled_local_relaxation',False) and maximum_merit and it%5==0 and norm<c.get('implicit_local_force_threshold_multiplier',4.)*c['implicit_force_tolerance_N']:
                peak=np.argsort(abs(gradient))[-min(32,len(gradient)):]
                neighborhood=np.unique(np.r_[peak,J[peak,:].tocsr().indices])
                if len(neighborhood)<=c.get('implicit_local_max_dofs',4096):
                    try:
                        block=J[neighborhood,:][:,neighborhood].tocsc()+diags(diagonal[neighborhood]*c['implicit_regularization'])
                        local_factor=splu(block,permc_spec='COLAMD');local=np.zeros(p.ndof)
                        local[neighborhood]=local_factor.solve(-gradient[neighborhood]);del local_factor
                        local_slope=float(merit_gradient@local[:p.nmicro])
                        if np.all(np.isfinite(local)) and local_slope<0:
                            local_bound=contact.ccd(trial,local);local_alpha=1. if local_bound>=1 else .9*local_bound
                            for _ in range(12):
                                candidate=trial+local_alpha*local
                                try:
                                    ci,_,_,_=values(candidate);cg=(ci-external)[:p.nmicro];cm=measure(cg)
                                    if cm<.9*merit and cm<=merit+1e-4*local_alpha*local_slope and cg@cg<=(gradient@gradient)*(1+1e-10):
                                        log(dict(local_force_relaxation=True,dofs=len(neighborhood),residual_before_N=norm,residual_after_N=float(np.max(abs(cg))),residual_l2_after_N=float(np.linalg.norm(cg)),alpha=local_alpha))
                                        step_history.append(float(np.max(abs(candidate[:p.nmicro]-trial[:p.nmicro]),initial=0.)))
                                        trial=candidate;break
                                except (ValueError,RuntimeError):pass
                                local_alpha*=.5
                            else:local_alpha=0.
                            if local_alpha>0:continue
                    except RuntimeError:pass
            dq=np.zeros(p.ndof)
            if violation_mode:
                selected=diags(np.where(abs(gradient)>violation_margin,1.,c.get('implicit_force_box_inactive_weight',0.)))@J
                columns=1/np.sqrt(np.maximum(np.asarray(selected.power(2).sum(axis=0)).ravel(),1e-12))
                linear=lsmr(selected@diags(columns),-merit_force,damp=.01,atol=1e-4,btol=1e-4,maxiter=500)
                dq[:p.nmicro]=columns*linear[0]
            else:
                try:
                    factor=splu(J+diags(diagonal*c['implicit_regularization']),permc_spec='COLAMD')
                    dq[:p.nmicro]=factor.solve(-gradient);del factor
                except RuntimeError:dq[:p.nmicro]=np.nan
            slope=float(merit_gradient@dq[:p.nmicro])
            if not np.all(np.isfinite(dq)) or not np.isfinite(slope) or slope>=0:
                column_norm=np.asarray(J.power(2).sum(axis=0)).ravel()
                dq[:p.nmicro]=-merit_gradient/np.maximum(column_norm,1e-12)
                slope=float(merit_gradient@dq[:p.nmicro]);log(dict(coupled_merit_gradient_fallback=True))
            if not np.all(np.isfinite(dq)) or not np.isfinite(slope) or slope>=0:raise ValueError('No coupled residual descent direction')
            bound=contact.ccd(trial,dq);alpha=1. if bound>=1 else .9*bound
            accepted=False
            for ls in range(35):
                candidate=trial+alpha*dq
                try:
                    ci,_,_,_=values(candidate);cg=(ci-external)[:p.nmicro];candidate_merit=measure(cg)
                    if candidate_merit<=merit+1e-4*alpha*slope:
                        accepted=True;break
                except (ValueError,RuntimeError):pass
                alpha*=.5
            # Optional trust-region-like least-squares directions avoid a nearly
            # singular Newton direction. They change only the numerical search:
            # values() and the final force acceptance remain exactly the same.
            if c.get('implicit_coupled_least_squares_fallback',False) and (not accepted or candidate_merit>merit*.9):
                scales=[(np.where(abs(gradient)>violation_margin,1.,c.get('implicit_force_box_inactive_weight',0.)),False,merit_force)] if violation_mode else [(np.sqrt(weights),False,gradient)]
                if maximum_merit and not violation_mode and c.get('implicit_coupled_peak_lsmr',False):
                    scales.insert(0,(np.sqrt(weights)*np.maximum((abs(gradient)/norm)**((peak_power-2)/2 if peak_power is not None else 2),.05),True,gradient))
                best=None if not accepted else (candidate_merit,alpha,candidate)
                for row_scale,peak_search,right_force in scales:
                    row_scaled=diags(row_scale)@J
                    column_scale=1/np.sqrt(np.maximum(np.asarray(row_scaled.power(2).sum(axis=0)).ravel(),1e-12))
                    scaled=row_scaled@diags(column_scale)
                    for damping in ([.01] if peak_search else [.01,.1,1.]):
                        linear=lsmr(scaled,-row_scale*right_force,damp=damping,atol=1e-4,btol=1e-4,maxiter=500)
                        alternative=np.zeros(p.ndof);alternative[:p.nmicro]=column_scale*linear[0]
                        alternative_slope=float(merit_gradient@alternative[:p.nmicro])
                        if not np.isfinite(alternative_slope) or alternative_slope>=0:continue
                        ab=contact.ccd(trial,alternative);aa=1. if ab>=1 else .9*ab
                        for als in range(25):
                            ac=trial+aa*alternative
                            try:
                                ai,_,_,_=values(ac);ag=(ai-external)[:p.nmicro];am=measure(ag)
                                if am<=merit+1e-4*aa*alternative_slope and (not peak_search or ag@ag<=max(gradient@gradient,(c.get('implicit_peak_rms_fraction',c.get('implicit_hybrid_rms_fraction',.2))*c['implicit_force_tolerance_N'])**2*len(gradient))*(1+1e-10)):
                                    if best is None or am<best[0]:best=(am,aa,ac)
                                    log(dict(coupled_least_squares_damping=damping,linear_iterations=int(linear[2]),merit_after=am,merit_weighting=c.get('implicit_coupled_merit_weighting','none'),merit_norm=c.get('implicit_coupled_merit_norm','l2'),alpha=aa,**(dict(peak_weighted_search=True) if peak_search else {})))
                                    break
                            except (ValueError,RuntimeError):pass
                            aa*=.5
                        if best is not None and best[0]<(1-c.get('implicit_lsmr_sufficient_reduction',.5))*merit:break
                    # A peak-focused direction must reduce maximum-force merit
                    # by 20% within the existing hybrid RMS budget. Final force
                    # acceptance is unchanged; large residuals may be redistributed.
                    if best is not None and best[0]<(1-c.get('implicit_lsmr_sufficient_reduction',.2 if peak_search else .5))*merit:break
                if best is not None:candidate_merit,alpha,candidate=best;accepted=True
            if not accepted:raise ValueError('Coupled residual line search failed')
            log(dict(coupled_alpha=alpha,merit_before=merit,merit_after=candidate_merit,merit_weighting=c.get('implicit_coupled_merit_weighting','none'),merit_norm=c.get('implicit_coupled_merit_norm','l2')))
            step_history.append(float(np.max(abs(candidate[:p.nmicro]-trial[:p.nmicro]),initial=0.)))
            trial=candidate
        raise ValueError('Coupled Newton iteration budget exhausted')
    except (ValueError,RuntimeError) as exc:return dict(reason=str(exc),iterate=trial)
