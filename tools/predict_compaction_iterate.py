"""Create a feasible secant initial guess; never commit predicted material history."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from packing_fem.explicit_periodic import Periodic
from packing_fem.explicit_material import geometry
from packing_fem.fixed_point import affine_predictor
from packing_fem.implicit_material import evaluate
from packing_fem.ipc_periodic import PeriodicIPC


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--previous',required=True);p.add_argument('--current',required=True);p.add_argument('--compression',required=True,type=float);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);report_path=out.with_suffix('.json')
    if out.exists() or report_path.exists():raise FileExistsError(out)
    c=json.loads(Path(a.config).read_text())
    with np.load(a.previous) as f:previous=dict(f)
    with np.load(a.current) as f:d=dict(f)
    if any(not np.array_equal(d[k],previous[k]) for k in ['X','t']):raise ValueError('Secant predictor requires unchanged periodic chart')
    periodic=Periodic(d['X'],c['length_mm']);q=d['q'];prescribed=-a.compression
    delta=q[periodic.prescribed]-previous['q'][periodic.prescribed]
    ratio=(prescribed-q[periodic.prescribed])/delta
    if not 0<ratio<=8:raise ValueError('Unsafe or nonpositive extrapolation ratio')
    affine=affine_predictor(periodic,q,prescribed);secant=q+ratio*(q-previous['q']);secant[-6:]=affine[-6:]
    direction=secant-affine
    # The finite predictor is bounded; a larger CCD query horizon only avoids
    # clipping a safe initial guess. The collision law is unchanged.
    contact=PeriodicIPC(d['X'],d['t'],periodic,dict(c,ipc_ccd_maximum_displacement_mm=1.))
    bound=contact.ccd(affine,direction);safe=1. if bound>=1 else .8*bound
    inverse=np.linalg.inv(geometry(d['x'],d['t'])[2]);rows=[];best=None
    for fraction in [0.,safe,safe/2,safe/4]:
        candidate=affine+fraction*direction
        try:
            _,g,_,state=evaluate(periodic.positions(candidate),d['t'],d['b'],d['ep'],inverse,d['V0'],c,parent=d['parent'])
            _,cg,_,n=contact.evaluate_coupled(candidate,q)
            force=(np.asarray(periodic.A.T@g.ravel()).ravel()+cg)[:periodic.nmicro]
            row=dict(fraction=float(fraction),free_residual_N=float(np.max(abs(force))),free_residual_l2_N=float(np.linalg.norm(force)),minimum_element_volume_mm3=float(state['volume'].min()),contacts=n)
            if best is None or row['free_residual_l2_N']<best[0]:best=(row['free_residual_l2_N'],candidate,row)
        except (ValueError,RuntimeError) as exc:row=dict(fraction=float(fraction),rejected=str(exc))
        rows.append(row)
    if best is None:raise ValueError('No feasible predictor including affine fallback')
    _,chosen,record=best
    np.savez_compressed(out,q=chosen,X=d['X'],t=d['t'],oldq=q,b=d['b'],ep=d['ep'])
    report=dict(initial_guess_only=True,material_history_source=a.current,previous_accepted_state=a.previous,source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in [a.previous,a.current,a.config]},target_compression=a.compression,extrapolation_ratio=float(ratio),ccd_safe_fraction=float(safe),candidates=rows,selected=record,output=str(out))
    report_path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
