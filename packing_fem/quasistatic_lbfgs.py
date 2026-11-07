"""Fixed-cell solid FEM minimization; exact axial Dirichlet control, unchanged laws."""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from .implicit_material import evaluate
from .fixed_point import affine_predictor,anderson_candidate


def equilibrium(p,t,q,b,ep,inverse,V0,c,contact,prescribed,external,log,parent=None,initial_guess=None):
    oldq=q.copy();trial=q.copy() if initial_guess is None else initial_guess.copy()
    if prescribed is None or c.get('transverse_boundary')!='fixed_cell':return dict(reason='L-BFGS requires prescribed axial compression and fixed transverse cell',iterate=trial)
    trial=affine_predictor(p,trial,prescribed);macro=trial[-6:].copy()
    if np.any(abs(macro[[0,1,3,4,5]])>1e-12):return dict(reason='Fixed transverse lattice changed',iterate=trial)
    memory=c.get('implicit_lbfgs_memory',10);refresh=c.get('implicit_lbfgs_refresh',25)
    max_iterations=c.get('implicit_lbfgs_max_iterations',500);tol=c['implicit_force_tolerance_N']
    depth=c.get('implicit_anderson_depth',0);outputs=[];residuals=[];lag_input=trial.copy()
    displacement_scale=np.ones(p.ndof);displacement_scale[-6:]=p.L
    def values(v):
        e,g,_,state=evaluate(p.positions(v),t,b,ep,inverse,V0,c,parent=parent)
        ce,cg,_,n=contact.evaluate(v)
        return e+ce-float(external@v),np.asarray(p.A.T@g.ravel()).ravel()+cg,state,n
    try:
        contact.set_lag(trial,oldq)
        for lag in range(c['implicit_max_lag']):
            pairs=[];factor=None;factor_contacts=None;factor_norm=None;energy,internal,state,n=values(trial)
            for it in range(max_iterations):
                gradient=(internal-external)[:p.nmicro];norm=float(np.max(abs(gradient)))
                log(dict(lag=lag,lbfgs_iteration=it,compression=float(-trial[p.prescribed]),residual_N=norm,contacts=n,memory=len(pairs)))
                if norm<tol:break
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
                    factor=splu(K+diags(diagonal*c['implicit_regularization']),permc_spec='COLAMD')
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
                bound=contact.ccd(trial,dq);alpha=1. if bound>=1 else .9*bound
                if getattr(contact,'last_ccd_horizon',1.)<1:
                    log(dict(ccd_horizon=contact.last_ccd_horizon,proposed_max_displacement_mm=contact.last_ccd_max_displacement_mm))
                accepted=False;roundoff=32*np.finfo(float).eps*max(c['young_mpa']*float(V0.sum()),abs(energy),1e-30)
                for ls in range(35):
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
                if curvature>1e-12*np.linalg.norm(s)*np.linalg.norm(y) and curvature>0:
                    pairs.append((s.copy(),y.copy(),1/curvature));pairs=pairs[-memory:]
                trial=candidate;energy=ve;internal=vi;state=vs;n=vn
            else:raise ValueError('L-BFGS iteration budget exhausted')
            contact.set_lag(trial,oldq);_,internal,state,n=values(trial)
            norm=float(np.max(abs((internal-external)[:p.nmicro])))
            if norm<tol:return trial,state,internal,n,it,norm
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
                            trial=candidate;improved=True
                            if cnorm<tol:return trial,cs,ci,cn,it,cnorm
                    except (ValueError,RuntimeError):pass
                    if not improved:contact.set_lag(trial,oldq)
            lag_input=trial.copy()
        raise ValueError('L-BFGS friction lag iterations did not converge')
    except (ValueError,RuntimeError) as exc:return dict(reason=str(exc),iterate=trial)
