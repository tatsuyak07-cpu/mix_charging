"""Compare exact sparse orderings on a saved FEM/contact tangent (no model changes)."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from packing_fem.explicit_periodic import Periodic
from packing_fem.explicit_material import geometry
from packing_fem.implicit_material import evaluate
from packing_fem.ipc_periodic import PeriodicIPC


def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--state',required=True);a.add_argument('--out',required=True);args=a.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    c=json.loads(Path(args.config).read_text())
    with np.load(args.state) as stream:d=dict(stream)
    p=Periodic(d['X'],c['length_mm']);start=time.monotonic()
    _,_,K,_=evaluate(d['x'],d['t'],d['b'],d['ep'],np.linalg.inv(geometry(d['x'],d['t'])[2]),d['V0'],c,True,d['parent'])
    material_s=time.monotonic()-start;start=time.monotonic()
    contact=PeriodicIPC(d['X'],d['t'],p,c);contact.set_lag(d['q'],d['q']);_,_,Kc,n=contact.evaluate(d['q'],True)
    contact_s=time.monotonic()-start
    K=(p.A.T@K@p.A+Kc)[:p.nmicro,:p.nmicro].tocsc()
    K+=diags(np.maximum(abs(K.diagonal()),1e-6)*c['implicit_regularization'])
    rhs=np.random.default_rng(20261107).normal(size=(p.nmicro,2));rows=[];reference=None
    for ordering in ['COLAMD','MMD_AT_PLUS_A']:
        start=time.monotonic();factor=splu(K,permc_spec=ordering);factor_s=time.monotonic()-start
        start=time.monotonic();solution=factor.solve(rhs);solve_s=time.monotonic()-start
        if reference is None:reference=solution
        rows.append(dict(ordering=ordering,factor_s=factor_s,solve_s=solve_s,factor_nnz=factor.L.nnz+factor.U.nnz,max_residual=float(np.max(abs(K@solution-rhs))),solution_relative_difference=float(np.max(abs(solution-reference))/max(np.max(abs(reference)),1e-30))))
        del factor
    report=dict(state=args.state,material_s=material_s,contact_s=contact_s,contacts=n,shape=K.shape,nnz=K.nnz,comparisons=rows)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if any(r['max_residual']>1e-7*max(1.,np.max(abs(rhs))) or r['solution_relative_difference']>1e-7 for r in rows):raise RuntimeError('Ordering comparison failed')

if __name__=='__main__':main()
