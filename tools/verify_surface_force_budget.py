"""Compare component-relative and solver-normalized surface integration budgets."""
import argparse,json,time
from pathlib import Path
import numpy as np
from packing_fem.checkpoint import load
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.embedded_contact import EmbeddedContact
from tools import experimental_surface_budget as surface_patch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state,metadata=load(a.checkpoint);c=metadata['config'];X=state['X'];t=state['t'];L=c['length_mm'];periodic=Periodic(X,L)
    contact=EmbeddedContact(X,t,periodic,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],thickness=c['barrier_thickness_mm'])
    rows=[];start=time.perf_counter()
    for compression in (0.,1e-5,1e-4):
        q=state['q'].copy();q[-4]-=compression;x=periodic.positions(q);H=deformation(q[-6:])*L
        contact._x=x;contact._H=H;contact.inner.guard.trial_shifts=state['contact']['guard']['trial_shifts']
        contact.inner.guard.neighbor_builds=int(compression*1e8);contact._candidate_build=-1
        contact.inner.guard.neighbor_geometry=(contact.P@x,H)
        contact._candidates((1521,1522,1652,1653))
        for face in (1521,1522,1652,1653):
            keys=sorted(set(contact._face_candidates[face]));targets=np.array([x[contact.faces[j]]+H@shift for j,shift in keys])
            groups=surface_patch.components(np.array([contact.faces[j] for j,_ in keys]),[shift for _,shift in keys],X,L)
            identities=[(int(periodic.classes[node]),*np.rint((X[node]+L*np.asarray(shift))/(L*1e-10)).astype(np.int64)) for j,shift in keys for node in contact.faces[j]]
            unique=sorted(set(identities));mapping=np.array([unique.index(k) for k in identities]).reshape(-1,3)
            source=x[contact.faces[face]];ref=X[contact.faces[face]];A=np.linalg.norm(np.cross(ref[1]-ref[0],ref[2]-ref[0]))/2
            options=dict(kn=.5*c['contact_penalty_mpa_mm'],thickness=c['barrier_thickness_mm'],tolerance=.00025,vertex_map=mapping,groups=groups,pointwise=True)
            old=surface_patch.integrate(source,targets,A,**options)
            new=surface_patch.integrate(source,targets,A,force_error_norm='resultant',**options)
            applied=new['weights']@surface_patch.grouped_field(new['bary'],source,targets,groups,options['kn'],options['thickness'],mapping,True)[0]
            exact=old['value'];scale=max(np.linalg.norm(exact[1:10].reshape(3,3).sum(0)),np.linalg.norm(exact[10:].reshape(-1,3),axis=1).sum(),1e-16)
            eerr=float(abs(applied[0]-exact[0])/max(abs(exact[0]),1e-18));ferr=float(abs(applied[1:]-exact[1:]).max()/scale)
            row=dict(extra_macro_compression=compression,face=face,component_points=len(old['weights']),resultant_points=len(new['weights']),energy_error=eerr,force_error=ferr,passed=bool(max(eerr,ferr)<.001))
            rows.append(row);print(json.dumps(row),flush=True)
    summary=dict(scope=__doc__,passed=all(row['passed'] for row in rows),cases=rows,wall_s=time.perf_counter()-start,
                 note='Identical material, barrier, reference area and solver acceptance tolerance. Manufactured additional poses are integration controls, not equilibria.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not summary['passed']:raise SystemExit(1)


if __name__=='__main__':main()
