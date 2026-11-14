"""Coupled-force solve with frozen-friction energy globalization when needed.

Every phase uses the SAME accepted configuration and material history.
Only a solve passing the original force and world rigid-balance tests returns
a physical state; intermediate configurations are numerical initial guesses.
"""
from .quasistatic_coupled import equilibrium as coupled
from .quasistatic_lbfgs import equilibrium as minimize


def equilibrium(p,t,q,b,ep,inverse,V0,c,contact,prescribed,external,log,parent=None,initial_guess=None):
    first=c.get('implicit_hybrid_initial_newton_iterations',20)
    warm_lags=c.get('implicit_hybrid_warm_lags',2)
    warm_updates=c.get('implicit_hybrid_warm_updates_per_lag',10)
    if any(int(v)!=v or v<0 for v in [first,warm_lags,warm_updates]) or (warm_lags and not warm_updates):
        return dict(reason='Hybrid search budgets must be nonnegative integers with positive inner warm updates',iterate=q.copy(),iterations=0)
    root=dict(c,implicit_coupled_merit_norm='l2')
    guess=initial_guess;updates=0

    def finish(result):
        if isinstance(result,dict):return dict(result,iterations=updates+int(result.get('iterations',0)))
        unknown,state,internal,count,iterations,norm=result
        return unknown,state,internal,count,updates+iterations,norm

    if first:
        log(dict(hybrid_phase='initial_coupled_force',maximum_iterations=int(first),accepted_history_unchanged=True))
        result=coupled(p,t,q,b,ep,inverse,V0,dict(root,implicit_coupled_max_iterations=int(first)),contact,prescribed,external,log,parent,guess)
        if not isinstance(result,dict):return finish(result)
        updates+=int(result.get('iterations',0));guess=result['iterate']
        log(dict(hybrid_initial_rejected=result['reason'],iterate_is_initial_guess_only=True))
    if warm_lags:
        log(dict(hybrid_phase='energy_globalization',maximum_lags=int(warm_lags),inner_updates=int(warm_updates),accepted_history_unchanged=True))
        warm=dict(c,implicit_max_lag=int(warm_lags),implicit_lbfgs_inexact_lag_updates=int(warm_updates))
        result=minimize(p,t,q,b,ep,inverse,V0,warm,contact,prescribed,external,log,parent,guess)
        if not isinstance(result,dict):return finish(result)
        updates+=int(result.get('iterations',0));guess=result['iterate']
        log(dict(hybrid_warm_rejected=result['reason'],iterate_is_initial_guess_only=True))
    polish=dict(root,implicit_coupled_merit_norm=c.get('implicit_hybrid_final_merit','hybrid'),implicit_hybrid_rms_fraction=c.get('implicit_hybrid_final_rms_fraction',.02),implicit_coupled_local_relaxation=c.get('implicit_hybrid_final_local_relaxation',True))
    log(dict(hybrid_phase='original_force_polish',accepted_history_unchanged=True,preceding_updates=updates,merit_norm=polish['implicit_coupled_merit_norm'],peak_switch_rms_fraction=polish['implicit_hybrid_rms_fraction']))
    result=coupled(p,t,q,b,ep,inverse,V0,polish,contact,prescribed,external,log,parent,guess)
    if isinstance(result,dict):
        result=dict(result,reason='Hybrid original-force polish failed: '+result['reason'])
    return finish(result)
