"""Compare cached swept broad-phase contact against fresh searches on a real RVE."""
import argparse,json,time
from pathlib import Path
import numpy as np
from packing_fem.explicit_periodic import Periodic
from packing_fem.ipc_periodic import PeriodicIPC


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--current',required=True);p.add_argument('--previous',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    c=json.loads(Path(a.config).read_text())
    with np.load(a.current) as f:d=dict(f)
    with np.load(a.previous) as f:old=f['q'].copy()
    periodic=Periodic(d['X'],c['length_mm']);q=d['q'];direction=np.zeros_like(q)
    direction[:periodic.nmicro]=np.random.default_rng(778).normal(scale=.001,size=periodic.nmicro)
    cached=PeriodicIPC(d['X'],d['t'],periodic,dict(c,ipc_cache_swept_candidates=True))
    fresh=PeriodicIPC(d['X'],d['t'],periodic,dict(c,ipc_cache_swept_candidates=False))
    start=time.monotonic();bound=cached.ccd(q,direction);cache_ccd_s=time.monotonic()-start
    start=time.monotonic();reference_bound=fresh.ccd(q,direction);fresh_ccd_s=time.monotonic()-start
    rows=[]
    for fraction in [.1,.5,.9]:
        query=q+fraction*bound*direction
        start=time.monotonic();ea,ga,_,na=cached.evaluate_coupled(query,old);cached_s=time.monotonic()-start
        hit=cached.last_collision_cache_hit
        start=time.monotonic();eb,gb,_,nb=fresh.evaluate_coupled(query,old);fresh_s=time.monotonic()-start
        rows.append(dict(fraction=fraction,cache_hit=hit,contacts_cached=na,contacts_fresh=nb,energy_absolute_difference_Nmm=abs(ea-eb),force_max_difference_N=float(np.max(abs(ga-gb))),cached_s=cached_s,fresh_s=fresh_s))
    report=dict(state=a.current,ccd_cached=bound,ccd_fresh=reference_bound,ccd_cached_s=cache_ccd_s,ccd_fresh_s=fresh_ccd_s,queries=rows,passed=bool(abs(bound-reference_bound)<1e-8 and all(r['cache_hit'] and r['contacts_cached']==r['contacts_fresh'] and r['energy_absolute_difference_Nmm']<1e-8 and r['force_max_difference_N']<1e-8 for r in rows)))
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
