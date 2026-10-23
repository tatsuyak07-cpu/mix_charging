"""Audit exact periodic reconstruction on a saved rejected remesh proposal."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import lsqr
from packing_fem.explicit_periodic import Periodic


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads((a.source/'config.json').read_text());s=json.loads((a.source/'summary.json').read_text());attempts=list(a.source.glob('surface-remesh-attempt-*.npz'));attempt=sorted(attempts)[-1]
    with np.load(a.source/'last-state.npz') as z:macro=z['generalized_position'][-6:]
    with np.load(attempt) as z:X=z['reference_points'];x=z['points']
    periodic=Periodic(X,c['length_mm']);old=lsqr(periodic.A,(x-X).ravel(),atol=1e-13,btol=1e-13,iter_lim=10*periodic.ndof)[0];new=periodic.remap_positions(x,macro)
    old_error=float(np.max(abs(periodic.positions(old)-x)));new_error=float(np.max(abs(periodic.positions(new)-x)));tol=periodic.L*1e-10
    result=dict(scope=__doc__,source_proposal=str(attempt),old_least_squares_error_mm=old_error,exact_class_reconstruction_error_mm=new_error,unchanged_tolerance_mm=tol,checks=dict(old_failure_reproduced=old_error>tol,exact_transfer_passes=new_error<tol,macro_preserved=bool(np.array_equal(new[-6:],macro))),physical_solution_verified=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'verify_rve_position_transfer.py').write_bytes(Path(__file__).read_bytes());np.savez_compressed(a.output/'reconstructed-position.npz',reference_points=X,points=x,generalized_position=new);print(json.dumps(result,indent=2))
    if not all(result['checks'].values()):raise SystemExit(1)


if __name__=='__main__':main()
