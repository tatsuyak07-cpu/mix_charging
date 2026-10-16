"""Measure original packing contact workload without claiming a compaction run."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from packing_fem.barrier_contact import BarrierContact
from packing_fem.explicit_periodic import Periodic
from packing_fem.explicit_material import geometry


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate-budget',type=int,default=500000);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    src=Path('exports/explicit-fem-prototype-002/prepared-40percent-model');mesh=src/'model-mesh.npz'
    with np.load(mesh) as data:X,t=data['points'],data['tets']
    c=json.loads((src/'config.json').read_text());start=time.perf_counter();contact=None
    result=dict(source=str(src),source_mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),length_mm=c['length_mm'],
        nodes=len(X),tetrahedra=len(t),initial_porosity=float(1-geometry(X,t)[0].sum()/c['length_mm']**3),
        candidate_budget=a.candidate_budget,full_compaction_started=False,physical_solution_validated=False)
    try:
        contact=BarrierContact(X,t,Periodic(X,c['length_mm']),20.,.3,.1,thickness=.02,max_candidates=a.candidate_budget)
        f,extra,hist,report=contact.evaluate(X,np.zeros_like(X),np.eye(3)*c['length_mm'],np.zeros((3,3)),0.)
        result.update(initial_contact_evaluation_completed=True,initial_contacts=report['contacts'],minimum_separation_mm=report['minimum_separation_mm'])
    except (ValueError,RuntimeError) as exc:result.update(initial_contact_evaluation_completed=False,failure=repr(exc))
    result.update(elapsed_s=time.perf_counter()-start,search_counts=contact.search_counts if contact is not None else {})
    (a.output/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
