"""Run with an isolated Python-source snapshot; no installation or compilation."""
import argparse,datetime,hashlib,json,os,shutil,signal,subprocess,sys
from pathlib import Path


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def freeze(source,runtime):
    runtime=Path(runtime);source=Path(source);runtime.parent.mkdir(parents=True,exist_ok=True);runtime.mkdir(exist_ok=False)
    for name in ['packing_fem','tools']:
        shutil.copytree(source/name,runtime/name,ignore=shutil.ignore_patterns('__pycache__'))
    files=sorted(p for p in runtime.rglob('*') if p.is_file())
    manifest={p.relative_to(runtime).as_posix():digest(p) for p in files}
    for p in files:p.chmod(0o444)
    (runtime/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def run(args):
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    runtime=out.parent/'runtime'/out.name;source=Path(__file__).resolve().parents[1];manifest=freeze(source,runtime)
    command=[sys.executable,str((runtime/'tools/run_implicit_compaction.py').resolve())]
    for name in ['config','mesh','out','restart','initial_iterate','rejected_geometry']:
        value=getattr(args,name)
        if value is not None:command.extend(['--'+name.replace('_','-'),value])
    env=dict(os.environ,PYTHONPATH=str(runtime.resolve()),PYTHONDONTWRITEBYTECODE='1')
    record=dict(state='started',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,snapshot_isolated_from_working_source=True,environment_overrides={k:env[k] for k in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE']})
    child=subprocess.Popen(command,env=env);record['pid']=child.pid
    receipt=runtime/'launch.json';receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(frozen_runtime=str(runtime),child_pid=child.pid,source_files=len(manifest))),flush=True)
    try:code=child.wait()
    except KeyboardInterrupt:
        if child.poll() is None:child.send_signal(signal.SIGINT)
        code=child.wait()
    unchanged=all(digest(runtime/name)==expected for name,expected in manifest.items())
    record.update(state='process_exited',child_return_code=code,runtime_hashes_unchanged=unchanged)
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    if not unchanged:raise RuntimeError('Frozen calculation source changed; retain outputs as unverified')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mesh',required=True);p.add_argument('--out',required=True);p.add_argument('--restart');p.add_argument('--initial-iterate');p.add_argument('--rejected-geometry')
    raise SystemExit(run(p.parse_args()))
