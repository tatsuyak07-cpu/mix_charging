"""Newton solve of the actual coupled solid/contact/Coulomb equilibrium residual."""
import numpy as np
from pathlib import Path
import time
from scipy.sparse import diags
from scipy.sparse.linalg import splu, lsmr
from .fixed_point import affine_predictor
from .implicit_material import evaluate


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
    trial=affine_predictor(p,trial,prescribed)
    if np.any(abs(trial[-6:][[0,1,3,4,5]])>1e-12):return dict(reason='Fixed transverse lattice changed',iterate=trial)
    def values(v,tangent=False):
        _,g,Km,state=evaluate(p.positions(v),t,b,ep,inverse,V0,dict(c,implicit_project_material_tangent=False),tangent,parent)
        _,gc,Kc,n=contact.evaluate_coupled(v,oldq,tangent)
        internal=np.asarray(p.A.T@g.ravel()).ravel()+gc
        if not np.all(np.isfinite(internal)):raise ValueError('Nonfinite coupled force')
        J=None if not tangent else (p.A.T@Km@p.A+Kc)[:p.nmicro,:p.nmicro].tocsc()
        return internal,state,n,J
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
            merit=.5*float(gradient@(weights*gradient))
            log(dict(coupled_newton=it,compression=float(-trial[p.prescribed]),residual_N=norm,residual_l2_N=float(np.linalg.norm(gradient)),contacts=n))
            if norm<c['implicit_force_tolerance_N']:return trial,state,internal,n,it,norm
            diagonal=np.maximum(abs(J.diagonal()),1e-6);dq=np.zeros(p.ndof)
            try:
                factor=splu(J+diags(diagonal*c['implicit_regularization']),permc_spec='COLAMD')
                dq[:p.nmicro]=factor.solve(-gradient);del factor
            except RuntimeError:dq[:p.nmicro]=np.nan
            merit_gradient=np.asarray(J.T@(weights*gradient)).ravel()
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
                    ci,_,_,_=values(candidate);cg=(ci-external)[:p.nmicro];candidate_merit=.5*float(cg@(weights*cg))
                    if candidate_merit<=merit+1e-4*alpha*slope:
                        accepted=True;break
                except (ValueError,RuntimeError):pass
                alpha*=.5
            # Optional trust-region-like least-squares directions avoid a nearly
            # singular Newton direction. They change only the numerical search:
            # values() and the final force acceptance remain exactly the same.
            if c.get('implicit_coupled_least_squares_fallback',False) and (not accepted or candidate_merit>merit*.9):
                row_scale=np.sqrt(weights)
                row_scaled=diags(row_scale)@J
                column_scale=1/np.sqrt(np.maximum(np.asarray(row_scaled.power(2).sum(axis=0)).ravel(),1e-12))
                scaled=row_scaled@diags(column_scale)
                best=None if not accepted else (candidate_merit,alpha,candidate)
                for damping in [.01,.1,1.]:
                    linear=lsmr(scaled,-row_scale*gradient,damp=damping,atol=1e-4,btol=1e-4,maxiter=500)
                    alternative=np.zeros(p.ndof);alternative[:p.nmicro]=column_scale*linear[0]
                    alternative_slope=float(merit_gradient@alternative[:p.nmicro])
                    if not np.isfinite(alternative_slope) or alternative_slope>=0:continue
                    ab=contact.ccd(trial,alternative);aa=1. if ab>=1 else .9*ab
                    for als in range(25):
                        ac=trial+aa*alternative
                        try:
                            ai,_,_,_=values(ac);ag=(ai-external)[:p.nmicro];am=.5*float(ag@(weights*ag))
                            if am<=merit+1e-4*aa*alternative_slope:
                                if best is None or am<best[0]:best=(am,aa,ac)
                                log(dict(coupled_least_squares_damping=damping,linear_iterations=int(linear[2]),merit_after=am,merit_weighting=c.get('implicit_coupled_merit_weighting','none'),alpha=aa))
                                break
                        except (ValueError,RuntimeError):pass
                        aa*=.5
                    if best is not None and best[0]<.5*merit:break
                if best is not None:candidate_merit,alpha,candidate=best;accepted=True
            if not accepted:raise ValueError('Coupled residual line search failed')
            log(dict(coupled_alpha=alpha,merit_before=merit,merit_after=candidate_merit,merit_weighting=c.get('implicit_coupled_merit_weighting','none')))
            trial=candidate
        raise ValueError('Coupled Newton iteration budget exhausted')
    except (ValueError,RuntimeError) as exc:return dict(reason=str(exc),iterate=trial)
