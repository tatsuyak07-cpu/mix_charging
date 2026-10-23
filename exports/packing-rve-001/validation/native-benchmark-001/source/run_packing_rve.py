"""Prepare and optionally compress a small periodic sintered BCC trial cell."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from packing_fem.packing_rve import build
from packing_fem.explicit_solver import solve


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads(a.config.read_text());X,t,mask,m=build(c['rve_grid_n'],c['length_mm'],c['diameter_mm'],c['neck_diameter_mm'],c.get('mesh_phase_mm',[0.,0.,0.]))
    (a.output/'geometry.json').write_text(json.dumps(m,indent=2)+'\n');(a.output/'config.json').write_text(json.dumps(c,indent=2)+'\n');np.savez_compressed(a.output/'model-mesh.npz',points=X,tets=t,mask=mask)
    source=a.output/'source';source.mkdir()
    for f in Path('packing_fem').glob('*.py'):(source/f.name).write_bytes(f.read_bytes())
    (source/'run_packing_rve.py').write_bytes(Path(__file__).read_bytes())
    if c.get('element_backend','numpy')=='native':
        (source/'element_kernel.cpp').write_bytes(Path('native/element_kernel.cpp').read_bytes())
        (source/'build_native_kernel.py').write_bytes(Path('tools/build_native_kernel.py').read_bytes())
    checks=dict(initial_porosity_within_declared_tolerance=abs(m['initial_porosity']-c['requested_initial_porosity'])<=c['geometry_porosity_tolerance'],face_connected=m['periodic_components']==1,three_axis_winding=m['winding_rank']==3,neck_has_declared_minimum_cells=m['cells_across_neck']>=c['minimum_cells_across_neck'])
    (a.output/'preflight.json').write_text(json.dumps(dict(checks=checks,passed=all(checks.values()),mesh_convergence_verified=False),indent=2)+'\n')
    if not all(checks.values()):raise ValueError('RVE geometry preflight failed; mesh and metadata retained')
    if not a.prepare_only:
        start=time.perf_counter();r=solve(a.output/'calculation',X,t,c);(a.output/'wall-time.json').write_text(json.dumps(dict(wall_s=time.perf_counter()-start),indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)


if __name__=='__main__':main()
