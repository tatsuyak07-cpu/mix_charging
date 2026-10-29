"""Measured-porosity compaction/hold/unload; no pressure cap or load target."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from packing_fem.packing_rve import build
from packing_fem.explicit_solver import solve
from packing_fem.checkpoint import load


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-replays',type=int,default=32)
    p.add_argument('--resume',type=Path);p.add_argument('--replay-checkpoint',type=Path)
    p.add_argument('--prepared-mesh',type=Path)
    p.add_argument('--allow-contact-operator-update',action='store_true',help='Only at a certified contact-free checkpoint; changes are audited by solver')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    original=json.loads(a.config.read_text())
    if original.get('loading_mode')!='porosity_cycle' or 'pressure_stages' in original or 'pressure_mpa' in original:raise ValueError('Porosity-only cycle requires no pressure cap/stages')
    original['stop_request_file']=str(a.output/'STOP_REQUESTED')
    if a.prepared_mesh:
        geometry=json.loads((a.prepared_mesh/'metadata.json').read_text())
        for name,value in [('cell_length_mm',original['length_mm']),('diameter_mm',original['diameter_mm']),('neck_diameter_mm',original['neck_diameter_mm'])]:
            if geometry.get(name)!=value:raise ValueError('Prepared mesh dimensions do not match cycle configuration')
        with np.load(a.prepared_mesh/'mesh.npz') as z:X=z['points'];t=z['tets']
        mask=np.empty((0,0,0),bool)
    elif original.get('mesh_kind','voxel')=='cad':
        from tools.build_periodic_sintered import build as cad_build
        X,t,mask,geometry=cad_build(original,a.output/'cad-mesh')
    else:X,t,mask,geometry=build(original['rve_grid_n'],original['length_mm'],original['diameter_mm'],original['neck_diameter_mm'],original.get('mesh_phase_mm',[0.,0.,0.]))
    checks=dict(initial_porosity=abs(geometry['initial_porosity']-original['requested_initial_porosity'])<=original['geometry_porosity_tolerance'],
                periodic_connected=geometry['periodic_components']==1,winding=geometry['winding_rank']==3,
                neck_resolution=geometry['cells_across_neck']>=original['minimum_cells_across_neck'])
    (a.output/'geometry.json').write_text(json.dumps(dict(geometry=geometry,checks=checks),indent=2)+'\n')
    np.savez_compressed(a.output/'model.npz',X=X,t=t,mask=mask)
    if not all(checks.values()):raise ValueError('Cycle geometry preflight failed')
    source=a.output/'source';source.mkdir()
    for file in Path('packing_fem').glob('*.py'):(source/file.name).write_bytes(file.read_bytes())
    (source/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    rules={};inactive=str(a.replay_checkpoint) if a.replay_checkpoint else None;resume=str(a.resume) if a.resume else None
    if resume:
        state,metadata=load(resume);rules=state.get('runtime_rules',{})
        if state.get('contact_free_prefix'):inactive=resume
    attempts=[];seen=set();result=None;stop=None
    for attempt in range(a.max_replays+1):
        c=dict(original,embedded_quadrature_rules=rules)
        checkpoint=resume if attempt==0 else inactive
        if checkpoint:c.update(resume_checkpoint=checkpoint,inactive_quadrature_replay=True,allow_contact_operator_update=a.allow_contact_operator_update)
        directory=a.output/f'attempt-{attempt:03d}'
        try:result=solve(directory,X,t,c)
        except (ValueError,RuntimeError) as e:
            stop='Initialization/restart rejected: '+repr(e)
            (a.output/f'initialization-error-{attempt:03d}.json').write_text(json.dumps(dict(error=stop,accepted_result=False),indent=2)+'\n')
            break
        record=dict(attempt=attempt,steps=result['steps'],porosity=result['final_porosity'],failure=result['failure'],
                    checks=result['checks'],cycle=result.get('porosity_cycle'),refined_faces=len(rules))
        attempts.append(record);(a.output/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n');print(json.dumps(record),flush=True)
        for path in result.get('checkpoint_paths',[]):
            state,_=load(path)
            if state.get('contact_free_prefix'):inactive=path
        feature=result.get('failure_feature') or {}
        if not feature.get('requires_quadrature_refinement'):stop=result['failure'];break
        if not inactive:stop='No certified noncontact state for coupled quadrature replay';break
        if attempt==a.max_replays:stop='Quadrature replay budget exhausted';break
        if not feature.get('rules'):stop='Contact feature has no certified refinement rule';break
        updated=dict(rules)
        for face,rule in feature['rules'].items():updated[str(face)]={k:rule[k] for k in ('bary','weights','partition') if k in rule}
        digest=hashlib.sha256(json.dumps(updated,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest in seen:stop='Repeated quadrature rule without verified progress';break
        seen.add(digest);rules=updated
        (a.output/f'replay-rules-{attempt+1:03d}.json').write_text(json.dumps(rules)+'\n')
    summary=dict(scope=__doc__,target_loaded_porosity=original['target_final_porosity'],no_pressure_cap=True,
                 attempts=attempts,stop=stop,last_certified_noncontact_checkpoint=inactive,wall_s=time.perf_counter()-start,
                 completed=bool(result and all(result['checks'].values())),convergence_verified=False,
                 limitations=['Small BCC trial geometry; material uncalibrated.',
                              'Boundary-preserving volume reconnections; general surface regeneration is not claimed.',
                              'New quadrature points still require full coupled replay, never zeroing active history.'])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    if not summary['completed']:raise SystemExit(2)


if __name__=='__main__':main()
