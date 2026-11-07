"""Differentiate the actual equilibrium at a diagnostic, nonconverged RVE iterate."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from packing_fem.explicit_material import geometry
from packing_fem.explicit_periodic import Periodic
from packing_fem.implicit_material import evaluate
from packing_fem.ipc_periodic import PeriodicIPC


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--state',required=True);parser.add_argument('--iterate',required=True);parser.add_argument('--out',required=True);a=parser.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    c=json.loads(Path(a.config).read_text());c['implicit_project_material_tangent']=False
    with np.load(a.state) as f:old=dict(f)
    with np.load(a.iterate) as f:trace=dict(f)
    for key,target in [('X','X'),('t','t'),('oldq','q'),('b','b'),('ep','ep')]:
        if not np.array_equal(trace[key],old[target]):raise ValueError('Iterate has a different accepted material predecessor')
    p=Periodic(old['X'],c['length_mm']);q=trace['q'];inverse=np.linalg.inv(geometry(old['x'],old['t'])[2])
    contact=PeriodicIPC(old['X'],old['t'],p,c)
    def values(v,tangent=False):
        _,g,K,_=evaluate(p.positions(v),old['t'],old['b'],old['ep'],inverse,old['V0'],c,tangent,old['parent'])
        _,cg,CK,n=contact.evaluate_coupled(v,old['q'],tangent)
        return np.asarray(p.A.T@g.ravel()).ravel(),cg,None if not tangent else p.A.T@K@p.A,CK,n
    gm,gc,M,C,count=values(q,True);g=(gm+gc)[:p.nmicro];J=(M+C)[:p.nmicro,:p.nmicro].tocsc()
    scale=np.maximum(abs(J.diagonal()),1e-6);factor=splu(J+diags(scale*c['implicit_regularization']),permc_spec='COLAMD')
    raw=factor.solve(-g);del factor
    random=np.zeros(p.ndof);random[:p.nmicro]=np.random.default_rng(118).normal(size=p.nmicro)
    newton=np.zeros(p.ndof);newton[:p.nmicro]=raw
    checks=[]
    for name,direction in [('random',random),('Newton',newton)]:
        direction/=np.max(abs(direction))
        for h in [1e-5,1e-6,1e-7]:
            pm,pc,_,_,_=values(q+h*direction);mm,mc,_,_,_=values(q-h*direction)
            for component,K,plus,minus in [('material',M,pm,mm),('contact',C,pc,mc),('total',M+C,pm+pc,mm+mc)]:
                fd=((plus-minus)/(2*h))[:p.nmicro];error=(K@direction)[:p.nmicro]-fd
                checks.append(dict(direction=name,component=component,h=h,relative_l2_error=float(np.linalg.norm(error)/max(np.linalg.norm(fd),1e-30)),maximum_error_N_per_mm=float(np.max(abs(error)))))
    report=dict(iterate_is_solution=False,state=a.state,iterate=a.iterate,contacts=count,free_residual_N=float(np.max(abs(g))),free_residual_l2_N=float(np.linalg.norm(g)),maximum_raw_newton_micro_displacement_mm=float(np.max(abs(raw))),relative_linear_residual=float(np.linalg.norm(J@raw+g)/np.linalg.norm(g)),derivative_checks=checks)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
