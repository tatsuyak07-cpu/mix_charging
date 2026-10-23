from pathlib import Path
import json,time,numpy as np
from packing_fem.explicit_solver import solve
with np.load('results/packing-rve-development-001/bcc-16.npz') as z:X=z['points'];t=z['tets'];L=float(z['L'])
c=json.loads(Path('configs/explicit-area-cauchy-pressure.json').read_text());c.update(length_mm=L,multiple_patches=True,edge_stationarity_filter=True,certified_local_filter=False,remesh_every=0,ramp_s=.016,hold_s=.004,max_steps=200,output_every=20,usage_scope='performance_diagnostic_intentionally_limited_200_steps_not_solution')
st=time.perf_counter();r=solve('results/packing-rve-development-001/benchmark-001',X,t,c);print(json.dumps(dict(wall_s=time.perf_counter()-st,result=r)),flush=True)
