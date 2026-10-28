"""Prepare the sintered-particle model for the research explicit backend."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from packing_fem.explicit_material import hex_to_tet,geometry
from packing_fem.explicit_solver import solve


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('exports/porosity40-to10-001'));p.add_argument('--config',type=Path,default=Path('configs/explicit-prototype.json'));p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--allow-large',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    base=json.loads(a.config.read_text());old=json.loads((a.source/'config.json').read_text())
    c={**base,**{k:old[k] for k in ['length_mm','young_mpa','poisson','yield_mpa','hardening_mpa']},'compression_fraction':.45,'ramp_s':.02,'hold_s':.01,'max_steps':2000000,'requested_initial_porosity':old['porosity'],'target_final_porosity':old['target_final_porosity']}
    with np.load(a.source/'sintered-geometry.npz') as g:X,t=g['points'],hex_to_tet(g['points'],g['elements'])
    np.savez_compressed(a.output/'model-mesh.npz',points=X,tets=t)
    metadata={'source':str(a.source),'source_geometry_sha256':hashlib.sha256((a.source/'sintered-geometry.npz').read_bytes()).hexdigest(),'nodes':len(X),'tetrahedra':len(t),'initial_porosity':float(1-geometry(X,t)[0].sum()/c['length_mm']**3),'element':'compatible linear tetrahedron','material':'Kirchhoff-yield finite-strain Hencky J2; not identical to native CalculiX integration','mesh':'conforming tetrahedral subdivision of archived voxel geometry; no sphere surface improvement','target_10percent_validated':False,'full_calculation_started':False}
    (a.output/'config.json').write_text(json.dumps(c,indent=2)+'\n');(a.output/'model.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)
    if not a.prepare_only:
        if len(t)>5000 and not a.allow_large:raise ValueError('Research runner limit 5000 tetrahedra; model retained, use --allow-large for explicit large calculation')
        metadata['full_calculation_started']=True;(a.output/'model.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps(solve(a.output/'calculation',X,t,c)),flush=True)

if __name__=='__main__':main()
