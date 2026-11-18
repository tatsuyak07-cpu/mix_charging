"""Zero macroscopic strain control with unchanged original contact/material laws."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
from packing_fem.rigid_inclusion import make_periodic,make_contact,porosities
from packing_fem.explicit_material import geometry
from packing_fem.implicit_material import evaluate
from packing_fem.quasistatic_lbfgs import equilibrium
from tools.audit_rigid_compaction import audit


def locate(state,force,p):
    ep=state['ep'];i=int(np.argmax(ep));classes=p.classes[state['t'][i]]
    norms=np.r_[0.,np.linalg.norm(force[:p.body_nmicro].reshape(-1,3),axis=1)]
    peak_class=int(np.argmax(norms));V=state['V0']
    return dict(maximum_ep=float(ep[i]),maximum_ep_element=i,
        maximum_ep_element_center_mm=state['X'][state['t'][i]].mean(axis=0).tolist(),
        plastic_reference_volume_fraction=float(V[ep>1e-12].sum()/V.sum()),
        plastic_elements=int(np.count_nonzero(ep>1e-12)),
        maximum_ep_element_initial_nodal_contact_force_N=float(norms[classes].max()),
        initial_peak_force_class_in_maximum_ep_element=bool(peak_class in classes))


def run(source,out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    c=json.loads((source/'config.json').read_text());(out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    shutil.copy2(source/'state-0000.npz',out/'state-0000.npz')
    with np.load(out/'state-0000.npz') as f:d=dict(f)
    with np.load(source/'state-0001.npz') as f:loaded=dict(f)
    X,t,q,b,ep,V0,parent=[d[k] for k in ['X','t','q','b','ep','V0','parent']]
    if np.any(q) or np.any(ep):raise ValueError('Expected virgin zero-strain geometry')
    p=make_periodic(X,c['length_mm'],c);contact=make_contact(X,t,p,c)
    inverse=np.linalg.inv(geometry(d['x'],t)[2])
    _,gm,_,virgin=evaluate(d['x'],t,b,ep,inverse,V0,c,parent=parent)
    _,initial_force,_,_=contact.evaluate_coupled(q,q)
    report=dict(source=str(source),initial_state_sha256=hashlib.sha256((out/'state-0000.npz').read_bytes()).hexdigest(),
        original_material_and_contact_parameters_unchanged=True,
        no_contact_zero_strain_material_force_max_N=float(abs(gm).max()),
        no_contact_zero_strain_maximum_ep=float(virgin['ep'].max()),
        initial_contact_force_max_N=float(abs(initial_force[:p.nmicro]).max()),
        small_load=locate(loaded,initial_force,p),
        scope='All six macroscopic strains fixed to zero; microscopic periodic fluctuations and rigid6DOF free. Not zero macroscopic stress.')
    (out/'pre-solve.json').write_text(json.dumps(report,indent=2)+'\n')
    result=equilibrium(p,t,q,b,ep,inverse,V0,c,contact,0.,np.zeros(p.ndof),lambda e:print(json.dumps(e),flush=True),parent)
    if isinstance(result,dict):
        np.savez_compressed(out/'rejected-iterate.npz',q=result['iterate'],X=X,t=t,oldq=q,b=b,ep=ep)
        (out/'failure.json').write_text(json.dumps(dict(reason=result['reason'],iterate_is_solution=False),indent=2)+'\n')
        raise RuntimeError(result['reason'])
    nq,st,g,count,iterations,norm=result
    if np.any(nq[-6:]):raise ValueError('Macroscopic strain changed')
    final=dict(d,x=p.positions(nq),q=nq,b=st['b'],ep=st['ep'])
    np.savez_compressed(out/'state-0001.npz',**final)
    phi,total=porosities(st['volume'].sum(),p.L**3,c)
    row=dict(step=1,phase='zero_strain_relaxation',compression=0.,porosity=phi,nominal_pressure_MPa=float(-g[p.prescribed]/p.L**3),free_residual_N=norm)
    with (out/'history.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    report.update(zero_strain=locate(final,initial_force,p),iterations=iterations,final_residual_N=norm,
        zero_strain_macro_pressure_MPa=row['nominal_pressure_MPa'],
        plastic_elements_shared_with_small_load=int(np.count_nonzero((st['ep']>1e-12)&(loaded['ep']>1e-12))))
    physics=source.parent/'input/original-physics.json';shutil.copy2(physics,out/'original-physics.json')
    if not audit(out,out/'audit.json',out/'original-physics.json',True):raise RuntimeError('Original physics replay failed')
    report['original_physics_audit_passed']=True
    (out/'assessment.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.run,a.out)
