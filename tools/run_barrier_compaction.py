"""Pressure-controlled periodic FEM compaction, explicit numerical diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem import sintered
from packing_fem.explicit_material import hex_to_tet
from packing_fem.explicit_solver import solve


def fixture(refinement=1,pattern="isotropic"):
    if not isinstance(refinement,int) or refinement<1:raise ValueError("Positive integer refinement required")
    if pattern not in ("isotropic","bulk"):raise ValueError("Unknown refinement pattern")
    n=refinement; nz=3*n if pattern=="isotropic" else 2*n+1
    mask=np.ones((4*n,4*n,nz),bool);mask[n:3*n,n:3*n,n:(2*n if pattern=="isotropic" else n+1)]=False
    X,h,_=sintered.mesh(mask,1/n)
    z=np.r_[np.linspace(0,1.98,n+1),np.linspace(1.98,2.02,(n+1 if pattern=="isotropic" else 2))[1:],np.linspace(2.02,4.,n+1)[1:]]
    X[:,2]=z[np.rint(X[:,2]*n).astype(int)]
    return X,hex_to_tet(X,h)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/explicit-area-pressure.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',choices=['small','packing40','impact'],default='small')
    p.add_argument('--max-steps',type=int);p.add_argument('--time-scale',type=float,default=1.)
    p.add_argument('--cfl',type=float);p.add_argument('--thickness',type=float)
    p.add_argument('--penalty-scale',type=float,default=1.);p.add_argument('--refinement',type=int,default=1)
    p.add_argument('--refinement-pattern',choices=['isotropic','bulk'],default='isotropic')
    p.add_argument('--contact-backend',choices=['distance_barrier','area_guarded'])
    p.add_argument('--mass-frame',choices=['pinned','zero_momentum']);p.add_argument('--loading-mode',choices=['nominal_pressure','porosity_target']);p.add_argument('--target-porosity',type=float);p.add_argument('--unload-seconds',type=float,default=0.);p.add_argument('--recovery-seconds',type=float,default=0.);p.add_argument('--hold-seconds',type=float);p.add_argument('--damping-scale',type=float,default=1.);p.add_argument('--candidate-budget',type=int,default=1000000);p.add_argument('--pressure',type=float);p.add_argument('--timestep-method',choices=['element_wave','assembled_mass_bound']);a=p.parse_args()
    c=json.loads(a.config.read_text());c.update(contact_backend=a.contact_backend or c.get('contact_backend','area_guarded'),loading_mode=a.loading_mode or c.get('loading_mode','nominal_pressure'),pressure_mpa=a.pressure if a.pressure is not None else c.get('pressure_mpa',.1),
        length_mm=4.,mass_frame=a.mass_frame or c.get('mass_frame','pinned'),unload_s=a.unload_seconds,recovery_hold_s=a.recovery_seconds,damping_per_s=c['damping_per_s']*a.damping_scale,contact_candidate_budget=a.candidate_budget,timestep_method=a.timestep_method or c.get('timestep_method','element_wave'),barrier_thickness_mm=a.thickness if a.thickness is not None else c.get('barrier_thickness_mm',.02),cfl=a.cfl if a.cfl is not None else c['cfl'],max_steps=a.max_steps if a.max_steps is not None else c['max_steps'],ramp_s=c['ramp_s']*a.time_scale,hold_s=c['hold_s']*a.time_scale,
        contact_penalty_mpa_mm=c['contact_penalty_mpa_mm']*a.penalty_scale,output_every=100,remesh_every=100,minimum_quality=.03)
    if a.target_porosity is not None:c['target_final_porosity']=a.target_porosity
    if a.hold_seconds is not None:c['hold_s']=a.hold_seconds
    if a.case=='small':
        c.update(verification_refinement=a.refinement,verification_refinement_pattern=a.refinement_pattern)
        X,t=fixture(a.refinement,a.refinement_pattern)
    else:
        root=Path('exports/explicit-fem-prototype-002')/('prepared-40percent-model' if a.case=='packing40' else 'dynamic_impact')
        name='model-mesh.npz' if a.case=='packing40' else 'initial-mesh.npz'
        with np.load(root/name) as data:X,t=data['points'],data['tets']
        if a.case=='packing40':c.update(length_mm=25.)
        else:
            prior=json.loads((root/'config.json').read_text());c={**prior,**{k:c[k] for k in ('contact_backend','barrier_thickness_mm','max_steps','cfl','contact_penalty_mpa_mm')}}
    result=solve(a.output,X,t,c)
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if all(result["checks"].values()) else 2)

if __name__=='__main__':main()
