"""Pair-equivalence benchmark at the previously rejected large-quadrature pose.

The physical candidate cap stays 1,000,000 for AABB. A larger legacy cap is
used ONLY to enumerate an independent comparison, never to advance the solver.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from packing_fem.embedded_contact import EmbeddedContact
from packing_fem.explicit_periodic import Periodic,deformation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--rules',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads((a.run/'config.json').read_text());rules=json.loads(a.rules.read_text());L=c['length_mm']
    with np.load(a.run/'last-state.npz') as z:X=z['reference_points'];t=z['tets'];x=z['points'];q=z['generalized_position']
    contact=EmbeddedContact(X,t,Periodic(X,L),c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],quadrature_rules=rules,thickness=c['barrier_thickness_mm'],max_candidates=c['contact_candidate_budget'])
    guard=contact.inner.guard;H=deformation(q[-6:])*L;aug=contact.P@x;start=time.perf_counter();guard.prepare_geometry(aug,H);aabb=guard.neighbors;counts=dict(guard.search_counts);time_aabb=time.perf_counter()-start
    # Counting only would reproduce the old rejection but could not establish
    # pair equivalence. Enumerate the old method with an explicit audit cap.
    start=time.perf_counter();guard.max_candidates=10000000;legacy=guard._center_neighbors(aug,H,guard.trial_shifts);time_legacy=time.perf_counter()-start
    def signature(jobs):
        return {tuple(key):dict(ids=ids.tolist(),kind=kind,shift=list(map(int,shift)),kn=float(kn),threshold=None if threshold is None else float(threshold)) for key,ids,kind,shift,kn,threshold in jobs}
    left=signature(aabb);right=signature(legacy);same=left==right
    result=dict(scope=__doc__,passed=same,points=len(aug),virtual_points=len(aug)-len(X),physical_candidate_cap=c['contact_candidate_budget'],aabb_counts=counts,legacy_counts=guard.search_counts,
                aabb_jobs=len(aabb),legacy_jobs=len(legacy),pair_identical=same,aabb_wall_s=time_aabb,legacy_wall_s=time_legacy,
                pair_sha256=hashlib.sha256(repr(sorted(left.items())).encode()).hexdigest(),
                missing=[str(k) for k in right.keys()-left.keys()],extra=[str(k) for k in left.keys()-right.keys()])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not same:raise SystemExit(2)


if __name__=='__main__':main()
