"""Execute individually audited numerical-contact controls from frozen sources."""
import argparse,csv,json,os,shutil,subprocess,sys
from pathlib import Path
from tools.run_frozen_compaction import freeze,digest


def run(spec,out):
    spec=Path(spec);out=Path(out);definition=json.loads(spec.read_text());out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(spec,out/'spec.json');runtime=out/'study-runtime';freeze(Path(__file__).resolve().parents[1],runtime)
    env=dict(os.environ,PYTHONPATH=str(runtime.resolve()),PYTHONDONTWRITEBYTECODE='1');inputs=out/'input';inputs.mkdir();rows=[]
    for index,case in enumerate(definition['cases']):
        copied={};hashes={}
        for key in ['config','mesh','physics','restart','initial_iterate']:
            if key in case:
                target=inputs/f'{index:03d}-{key}{Path(case[key]).suffix}';shutil.copy2(case[key],target);copied[key]=str(target);hashes[key]=digest(target)
                if key=='restart':
                    source_config=Path(str(case[key])+'.source-config.json')
                    if not source_config.exists():source_config=Path(case[key]).parent/'config.json'
                    if source_config.exists():shutil.copy2(source_config,Path(str(target)+'.source-config.json'))
        directory=out/case['name'];command=[sys.executable,str(runtime.resolve()/'tools/run_frozen_compaction.py')]
        for key in ['config','mesh','restart','initial_iterate']:
            if key in copied:command.extend(['--'+key.replace('_','-'),copied[key]])
        command.extend(['--out',str(directory)]);print(json.dumps(dict(starting=case['name'])),flush=True)
        with (out/(case['name']+'.log')).open('x') as f:code=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        row=dict(name=case['name'],run=str(directory),return_code=code,inputs=copied,sha256=hashes,passed=False)
        if code==0 and (directory/'history.csv').exists():
            with (directory/'history.csv').open() as f:history=list(csv.DictReader(f))
            if history:
                row.update(accepted_increments=len(history),final=history[-1])
                command=[sys.executable,str(runtime.resolve()/'tools/audit_rigid_compaction.py'),'--run',str(directory),'--physics-config',copied['physics'],'--local-predecessors-only','--out',str(directory/'audit.json')]
                with (out/(case['name']+'-audit.log')).open('x') as f:audit_code=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
                row['passed']=bool(audit_code==0 and abs(float(history[-1]['compression'])-case['compression'])<1e-10)
        rows.append(row);(out/'assessment.json').write_text(json.dumps(dict(cases=rows,completed=len(rows)==len(definition['cases']),all_audits_passed=all(r['passed'] for r in rows),full_path_convergence_verified=False),indent=2)+'\n')
        print(json.dumps(row),flush=True)
    return all(r['passed'] for r in rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if not run(a.spec,a.out):raise SystemExit(1)
