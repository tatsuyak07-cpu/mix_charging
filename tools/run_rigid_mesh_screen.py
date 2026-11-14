"""Three-mesh initial-load screen; it never certifies the full compaction path."""
import argparse,csv,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
from tools.run_frozen_compaction import freeze,digest


def run(spec,out):
    spec=Path(spec);out=Path(out);definition=json.loads(spec.read_text());out.mkdir(parents=True,exist_ok=False)
    (out/'spec.json').write_text(json.dumps(definition,indent=2)+'\n')
    source=out/'study-runtime';freeze(Path(__file__).resolve().parents[1],source)
    inputs=out/'input';inputs.mkdir();copied=[]
    for index,item in enumerate(definition['cases']):
        record=dict(item,index=index)
        for key in ['config','mesh','metadata']:
            original=Path(item[key]);target=inputs/f'level-{index:03d}-{key}{original.suffix}';shutil.copy2(original,target);record[key]=str(target);record[key+'_sha256']=digest(target)
        copied.append(record)
    physics=inputs/'original-physics.json';shutil.copy2(definition['physics_config'],physics)
    thresholds=json.loads(Path(definition['threshold_spec']).read_text())['developer_screening_thresholds']
    (out/'inputs.json').write_text(json.dumps(dict(cases=copied,physics_config_sha256=digest(physics),thresholds=thresholds),indent=2)+'\n')
    env=dict(os.environ,PYTHONPATH=str(source.resolve()),PYTHONDONTWRITEBYTECODE='1');records=[]
    for item in copied:
        index=item['index'];directory=out/f'level-{index:03d}'
        command=[sys.executable,str((source/'tools/run_frozen_compaction.py').resolve()),'--config',item['config'],'--mesh',item['mesh'],'--out',str(directory)]
        print(json.dumps(dict(starting_mesh_level=index)),flush=True)
        with (out/f'level-{index:03d}.log').open('x') as stream:code=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        record=dict(level=index,run=str(directory),calculation_return_code=code,passed=False)
        if code==0 and (directory/'history.csv').exists():
            with (directory/'history.csv').open() as stream:rows=list(csv.DictReader(stream))
            if rows:
                last=rows[-1];record.update(compression=float(last['compression']),pressure_MPa=float(last['nominal_pressure_MPa']),porosity=float(last['porosity']),free_residual_N=float(last['free_residual_N']))
                checkpoint=directory/f"state-{int(last['step']):04d}.npz"
                with zipfile.ZipFile(checkpoint) as z:valid_npz=z.testzip() is None
                if valid_npz and abs(record['compression']-definition['common_compression'])<1e-10:
                    command=[sys.executable,str((source/'tools/audit_rigid_compaction.py').resolve()),'--run',str(directory),'--physics-config',str(physics),'--out',str(directory/'audit.json')]
                    with (out/f'level-{index:03d}-audit.log').open('x') as stream:audit_code=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
                    record['passed']=bool(audit_code==0 and json.loads((directory/'audit.json').read_text())['passed'] and record['pressure_MPa']>0)
        metadata=json.loads(Path(item['metadata']).read_text());record.update(input_nodes=metadata['nodes'],input_tetrahedra=metadata['tetrahedra'],cad_solid_volume_mm3=metadata['cad_solid_volume_mm3'],geometry_volume_relative_error=metadata['geometry_volume_relative_error'])
        records.append(record);print(json.dumps(record),flush=True)
    complete=bool(len(records)>=3 and all(r['passed'] for r in records));comparisons=[];production=None;trend=False;sequence=False;production_pass=False
    if complete:
        for a,b in zip(records,records[1:]):comparisons.append(dict(pressure_relative=abs(b['pressure_MPa']-a['pressure_MPa'])/abs(b['pressure_MPa']),porosity_absolute=abs(b['porosity']-a['porosity'])))
        trend=all(comparisons[-1][k]<=comparisons[-2][k]+1e-12 for k in comparisons[-1])
        sequence=bool(trend and comparisons[-1]['pressure_relative']<=thresholds['pressure_relative'] and comparisons[-1]['porosity_absolute']<=thresholds['porosity_absolute'])
        production=dict(pressure_relative=abs(records[-1]['pressure_MPa']-records[0]['pressure_MPa'])/abs(records[-1]['pressure_MPa']),porosity_absolute=abs(records[-1]['porosity']-records[0]['porosity']))
        production_pass=bool(sequence and production['pressure_relative']<=thresholds['pressure_relative'] and production['porosity_absolute']<=thresholds['porosity_absolute'])
    report=dict(scope='Initial0.5% compression screen on identical CAD; no full40% to10% mesh convergence claim.',screen_completed=complete,early_refinement_sequence_passed=sequence,production_coarsest_mesh_screen_passed=production_pass,full_loaded_path_mesh_convergence_verified=False,threshold_source='Previously declared developer screening thresholds; not material specifications.',thresholds=thresholds,cases=records,adjacent_differences=comparisons,errors_decrease=trend,production_coarsest_vs_finest=production)
    (out/'assessment.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    return complete


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if not run(a.spec,a.out):raise SystemExit(1)
