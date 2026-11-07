"""Saved-state derivative / regularization diagnostics; proposals are NOT solutions."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from packing_fem.explicit_periodic import Periodic
from packing_fem.explicit_material import geometry
from packing_fem.implicit_material import evaluate
from packing_fem.ipc_periodic import PeriodicIPC
from tools.run_implicit_compaction import affine_predictor


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--state',required=True);parser.add_argument('--iterate',required=True);parser.add_argument('--out',required=True);parser.add_argument('--regularizations',nargs='+',type=float,default=[1e-7,1e-9,1e-11]);a=parser.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    c=json.loads(Path(a.config).read_text())
    with np.load(a.state) as stream:d=dict(stream)
    with np.load(a.iterate) as stream:q=stream['q'].copy()
    p=Periodic(d['X'],c['length_mm']);q=affine_predictor(p,q,d['q'][p.prescribed]-c['implicit_strain_increment'])
    inverse=np.linalg.inv(geometry(d['x'],d['t'])[2])
    def material(v,tangent=False):
        e,g,K,s=evaluate(p.positions(v),d['t'],d['b'],d['ep'],inverse,d['V0'],c,tangent,d['parent'])
        return e,np.asarray(p.A.T@g.ravel()).ravel(),None if K is None else p.A.T@K@p.A
    contact=PeriodicIPC(d['X'],d['t'],p,c);contact.set_lag(q,d['q'])
    em,gm,Km=material(q,True);ec,gc,Kc,_=contact.evaluate(q,True)
    direction=np.zeros(p.ndof);direction[:p.nmicro]=np.random.default_rng(81).normal(size=p.nmicro)/np.sqrt(p.nmicro)
    checks=[]
    for h in [1e-5,1e-6]:
        for name,K,plus,minus in [('material',Km,material(q+h*direction)[1],material(q-h*direction)[1]),('contact',Kc,contact.evaluate(q+h*direction)[1],contact.evaluate(q-h*direction)[1])]:
            fd=(plus-minus)/(2*h);difference=(K@direction-fd)[:p.nmicro]
            checks.append(dict(component=name,h=h,relative_l2_error=float(np.linalg.norm(difference)/max(np.linalg.norm(fd[:p.nmicro]),1e-30)),max_absolute_error=float(np.max(abs(difference)))))
    K=(Km+Kc)[:p.nmicro,:p.nmicro].tocsc();gradient=(gm+gc)[:p.nmicro];scale=np.maximum(abs(K.diagonal()),1e-6);steps=[]
    for regularization in a.regularizations:
        try:
            factor=splu(K+diags(scale*regularization),permc_spec='COLAMD');dq=np.zeros(p.ndof);dq[:p.nmicro]=factor.solve(-gradient)
            bound=contact.ccd(q,dq);alpha=1. if bound>=1 else .9*bound;trial=q+alpha*dq
            ve,vg,_=material(trial);ce,cg,_,_=contact.evaluate(trial)
            h=1e-4
            directional_energy=(material(q+h*dq)[0]+contact.evaluate(q+h*dq)[0]-material(q-h*dq)[0]-contact.evaluate(q-h*dq)[0])/(2*h)
            steps.append(dict(regularization=regularization,step_max_mm=float(np.max(abs(dq[:p.nmicro]))),gradient_dot_step=float(gradient@dq[:p.nmicro]),energy_directional_derivative=directional_energy,ccd_alpha=alpha,energy_change_Nmm=ve+ce-em-ec,residual_after_N=float(np.max(abs((vg+cg)[:p.nmicro])))))
        except (ValueError,RuntimeError) as exc:steps.append(dict(regularization=regularization,rejected_reason=str(exc)))
    report=dict(initial_residual_N=float(np.max(abs(gradient))),derivative_checks=checks,newton_proposals=steps,interpretation='Nonconverged diagnostic proposals only; fixed macro strain and frozen material/friction history.')
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
