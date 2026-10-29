"""Stage-gated BCC compression with inactive-state material-quadrature replay."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from packing_fem.packing_rve import build
from packing_fem.explicit_solver import solve


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-replays',type=int,default=12);p.add_argument('--inactive-checkpoint',type=Path);p.add_argument('--allow-solver-update',action='store_true');p.add_argument('--replay-checkpoint',type=Path);p.add_argument('--extend-hold-budget',action='store_true')
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=False)
    original=json.loads(args.config.read_text());X,t,mask,geometry=build(original['rve_grid_n'],original['length_mm'],original['diameter_mm'],original['neck_diameter_mm'],original.get('mesh_phase_mm',[0.,0.,0.]))
    original['stop_request_file']=str(root/'STOP_REQUESTED')
    (root/'geometry.json').write_text(json.dumps(geometry,indent=2)+'\n');np.savez_compressed(root/'model.npz',X=X,t=t,mask=mask)
    checks=dict(initial_porosity=abs(geometry['initial_porosity']-original['requested_initial_porosity'])<=original['geometry_porosity_tolerance'],periodic_connected=geometry['periodic_components']==1,winding=geometry['winding_rank']==3,neck_resolution=geometry['cells_across_neck']>=original['minimum_cells_across_neck'])
    if not all(checks.values()):raise ValueError('Geometry preflight failed')
    source=root/'source';source.mkdir()
    for file in Path('packing_fem').glob('*.py'):(source/file.name).write_bytes(file.read_bytes())
    (source/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    rules={};attempts=[];seen=set();continuation=str(args.inactive_checkpoint) if args.inactive_checkpoint else None;inactive_checkpoint=str(args.replay_checkpoint) if args.replay_checkpoint else None;start=time.perf_counter();stop=None
    from packing_fem.checkpoint import load
    if continuation:
        state,metadata=load(continuation);rules=state.get('runtime_rules',metadata['config'].get('embedded_quadrature_rules',{}))
        if state.get('contact_free_prefix',False):inactive_checkpoint=continuation
    for attempt in range(args.max_replays+1):
        c=dict(original,embedded_quadrature_rules=rules)
        resume=continuation if attempt==0 else inactive_checkpoint
        if resume:c.update(resume_checkpoint=resume,inactive_quadrature_replay=True,allow_contact_operator_update=True)
        if args.extend_hold_budget:c['extend_stage_hold_budget']=True
        if args.allow_solver_update:c['allow_solver_update_at_inactive_checkpoint']=True
        directory=root/f'attempt-{attempt:03d}'
        result=solve(directory,X,t,c)
        history=np.loadtxt(directory/'history.csv',delimiter=',',skiprows=1,ndmin=2)
        if history[0,10]+history[0,11]>1e-16:
            stop='Initial checkpoint is not contact-energy free; inactive replay invalid';break
        record=dict(attempt=attempt,steps=result['steps'],pressure_MPa=result['peak_command_pressure_MPa'],porosity=result['final_porosity'],checks=result['checks'],failure=result['failure'],stage_records=result['pressure_stages'],rules=len(rules))
        attempts.append(record);(root/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n');print(json.dumps(record),flush=True)
        from packing_fem.checkpoint import load
        for path in result.get('checkpoint_paths',[]):
            state,_=load(path)
            if state.get('contact_free_prefix',False):inactive_checkpoint=path
        feature=result.get('failure_feature') or {}
        if not feature.get('requires_quadrature_refinement'):
            stop=result['failure'];break
        if not inactive_checkpoint:stop='No certified inactive checkpoint for quadrature replay';break
        if attempt==args.max_replays:stop='Initial-state quadrature replay budget exhausted';break
        requests=feature.get('rules')
        if not requests:stop='Unsupported quadrature request schema';break
        newrules=dict(rules)
        for face,rule in requests.items():newrules[str(face)]={k:rule[k] for k in ('bary','weights')}
        digest=hashlib.sha256(json.dumps(newrules,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest in seen:stop='Repeated quadrature rule; no verified refinement progress';break
        seen.add(digest);rules=newrules
        (root/f'replay-rules-{attempt+1:03d}.json').write_text(json.dumps(rules)+'\n')
    summary=dict(scope='Small BCC solid FEM; stage-gated Cauchy pressure; coupled initial-state quadrature replay',preflight=checks,
                 attempts=attempts,stop=stop,last_inactive_checkpoint=inactive_checkpoint,wall_s=time.perf_counter()-start,
                 reached_0_1_MPa=bool(result['checks'].get('requested_pressure_reached',False)),all_solver_checks=all(result['checks'].values()),
                 mesh_convergence_verified=False,representative_100mm_pack_verified=False)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    if not summary['reached_0_1_MPa'] or not summary['all_solver_checks']:raise SystemExit(2)


if __name__=='__main__':main()
