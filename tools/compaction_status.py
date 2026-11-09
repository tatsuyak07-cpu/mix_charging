"""Read-only status: distinguish saved running flags from live environment processes."""
import argparse,csv,json
from pathlib import Path
from datetime import datetime,timezone


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--run',action='append');a=parser.parse_args();root=Path(a.root).resolve()
    processes={}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if not any(Path(arg).name=='run_implicit_compaction.py' for arg in args) or '--out' not in args:continue
            out=Path(args[args.index('--out')+1])
            if not out.is_absolute():
                try:out=(proc/'cwd').resolve(strict=True)/out
                except OSError:
                    matches=[(ancestor/out).resolve() for ancestor in [root,*root.parents] if (ancestor/out).resolve().parent==root]
                    if not matches:continue
                    out=matches[0]
            processes.setdefault(str(out.resolve()),[]).append(int(proc.name))
        except (OSError,ValueError,IndexError):continue
    reports=[]
    runs=[root/name for name in a.run] if a.run else sorted(root.glob('compression-[0-9][0-9][0-9]'))
    for run in runs:
        status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        rows=[]
        if (run/'history.csv').exists():
            with (run/'history.csv').open() as f:rows=list(csv.DictReader(f))
        rows=[row for row in rows if (run/f"state-{int(row['step']):04d}.npz").exists()]
        last=None;smoothing=None;log=root/(run.name+'.log')
        if log.exists():
            for line in log.read_text().splitlines():
                try:entry=json.loads(line)
                except json.JSONDecodeError:continue
                if 'friction_smoothing_multiplier' in entry:smoothing=entry['friction_smoothing_multiplier']
                if 'coupled_newton' in entry:last=entry
        pids=processes.get(str(run.resolve()),[])
        effective='running' if pids else ('loaded_target_reached' if status.get('target_reached') else 'no_matching_process_observed')
        reports.append(dict(run=run.name,effective_status=effective,environment_pids=pids,saved_status=status.get('status'),running_flag_without_observed_process=bool(not pids and status.get('status')=='running'),saved_new_accepted_states=len(rows),last_accepted=rows[-1] if rows else None,last_iteration=last,iteration_is_solution=False,smoothing_multiplier=smoothing,last_log_utc=None if not log.exists() else datetime.fromtimestamp(log.stat().st_mtime,timezone.utc).isoformat()))
    print(json.dumps(dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),scope='Selected execution environment only; no inference of external task status or stop cause.',runs=reports),indent=2))


if __name__=='__main__':main()
