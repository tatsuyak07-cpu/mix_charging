"""Check nested-refinement eligibility at a saved packing state, without advancing it."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from packing_fem.checkpoint import load
from packing_fem.periodic_contact import Contact
from packing_fem.explicit_periodic import Periodic
from packing_fem.surface_remesh import bisect
from packing_fem.embedded_remesh import refine_rules


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state,metadata=load(a.checkpoint);c=metadata['config'];X=state['X'];t=state['t'];L=c['length_mm'];periodic=Periodic(X,L)
    surface=Contact(X,t,periodic,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'])
    rules=state['runtime_rules'];old=SimpleNamespace(X=X,L=L,faces=surface.faces,rules=rules,kn=c['contact_penalty_mpa_mm'],thickness=c.get('barrier_thickness_mm',.02),quadrature_tolerance=c.get('surface_quadrature_tolerance',.001))
    active={key[1] for key in state['contact']['history']};start=len(X);faces=[]
    for key,rule in sorted(rules.items(),key=lambda item:int(item[0])):
        stop=start+len(rule['weights'])
        if any(start<=node<stop for node in active):faces.append(int(key))
        start=stop
    if not faces:raise ValueError('Expected an active embedded face in the saved packing')
    ids=surface.faces[faces[0]];results=[]
    for edge in (ids[[0,1]],ids[[1,2]],ids[[2,0]]):
        row=dict(edge=edge.tolist(),eligible=False)
        try:
            proposal=bisect(X,state['x'],t,state['b'],state['ep'],state['V0'],periodic.velocities(state['v']),edge,L)
            new_surface=Contact(proposal['X'],proposal['t'],Periodic(proposal['X'],L),c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'])
            refined,_,_=refine_rules(old,new_surface.faces,proposal)
            row.update(eligible=True,refined_faces=len(refined))
        except ValueError as exc:row['reason']=str(exc)
        results.append(row)
    result=dict(scope=__doc__,checkpoint=str(a.checkpoint),step=state['steps'],active_histories=len(active),source_face=faces[0],candidates=results,
                eligible_candidates=sum(r['eligible'] for r in results),packing_remesh_completed=False,
                interpretation='Eligibility only; no contact/material transfer or physical trajectory accepted.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
