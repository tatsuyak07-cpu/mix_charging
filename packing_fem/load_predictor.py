"""Feasible load-bisection initial guesses; accepted material history stays fixed."""
import numpy as np
from .fixed_point import affine_predictor
from .explicit_material import geometry
from .implicit_material import evaluate


def search_stalled(norms,steps,window=8,reduction=.01,step_limit=1e-7):
    return bool(len(norms)>window and len(steps)>=4 and
                (norms[-window-1]-norms[-1])/max(norms[-window-1],1e-30)<reduction and
                max(steps[-4:])<step_limit)


def bisected_initial_guess(p,t,oldq,failedq,prescribed,b,ep,inverse,V0,parent,c,contact):
    delta=failedq[p.prescribed]-oldq[p.prescribed]
    fraction=(prescribed-oldq[p.prescribed])/delta
    if not fraction>0:raise ValueError('Predictor requires a forward load increment')
    affine=affine_predictor(p,oldq,prescribed)
    target=oldq+fraction*(failedq-oldq) if fraction<=1 else affine_predictor(p,failedq,prescribed)
    target[-6:]=affine[-6:]
    feasible=affine.copy()
    for _ in range(16):
        direction=target-feasible
        if np.max(abs(direction),initial=0.)<1e-12:break
        bound=contact.ccd(feasible,direction);horizon=getattr(contact,'last_ccd_horizon',1.)
        stop=bound<horizon*(1-1e-12);alpha=.8*bound if stop else bound
        if alpha<1e-10:break
        for _ in range(16):
            candidate=feasible+alpha*direction
            try:geometry(p.positions(candidate),t);break
            except ValueError:alpha*=.5;stop=True
        else:break
        feasible=candidate
        if stop:break
    records=[];best=None
    for blend in [0.,1.,.5,.25]:
        q=affine+blend*(feasible-affine)
        try:
            _,g,_,state=evaluate(p.positions(q),t,b,ep,inverse,V0,c,parent=parent)
            _,gc,_,_=contact.evaluate_coupled(q,oldq)
            free=(np.asarray(p.A.T@g.ravel()).ravel()+gc)[:p.nmicro]
            row=dict(blend=blend,maximum_force_N=float(np.max(abs(free),initial=0.)),l2_force_N=float(np.linalg.norm(free)),minimum_volume_mm3=float(state['volume'].min()))
            if best is None or row['l2_force_N']<best[0]:best=(row['l2_force_N'],q.copy(),row)
        except (ValueError,RuntimeError) as exc:row=dict(blend=blend,rejected=str(exc))
        records.append(row)
    if best is None:raise ValueError('No feasible initial guess including affine fallback')
    return best[1],dict(load_fraction=float(fraction),interpolation_fraction=float(fraction) if fraction<=1 else None,prediction_mode='interpolate' if fraction<=1 else 'affine_from_unaccepted_geometry',initial_guess_only=True,material_history_unchanged=True,candidates=records,selected=best[2])
