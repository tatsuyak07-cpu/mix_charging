"""Run independent small FEM benchmarks; retain diagnostics and original inputs."""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem import sintered
from packing_fem.explicit_material import hex_to_tet,geometry,oriented
from packing_fem.explicit_solver import solve
from packing_fem.adaptive_tet import propose


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--case',choices=['solid','void','all'],default='all');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    base=json.loads(Path('configs/explicit-prototype.json').read_text());results={}
    jobs=[]
    if a.case in ('solid','all'):
        c={**base,'length_mm':1.,'yield_mpa':1e6,'compression_fraction':.001,'contact_penalty_mpa_mm':1.,'remesh_every':100,'ramp_s':.0006,'hold_s':.0003,'damping_per_s':20000.}
        points,hexes,_=sintered.mesh(np.ones((1,1,1),bool),1.)
        jobs.append(('solid',points,hex_to_tet(points,hexes),c))
    if a.case in ('void','all'):
        mask=np.ones((3,3,3),bool);mask[1,1,1]=False;points,hexes,_=sintered.mesh(mask,1.)
        for scale,cfl in [(1.,.2),(2.,.2),(1.,.1)]:
            c={**base,'ramp_s':base['ramp_s']*scale,'hold_s':base['hold_s']*scale,'cfl':cfl}
            jobs.append((f'void_time{scale:g}_cfl{cfl:g}',points,hex_to_tet(points,hexes),c))
        c={**base,'density_tonne_mm3':base['density_tonne_mm3']*.5}
        jobs.append(('void_density0.5',points,hex_to_tet(points,hexes),c))
        c={**base,'damping_per_s':base['damping_per_s']*.5}
        jobs.append(('void_damping0.5',points,hex_to_tet(points,hexes),c))
    for name,points,t,c in jobs:
        try:
            result=solve(a.output/name,points,t,c);results[name]=result
            print(json.dumps({'case':name,'status':result['status'],'checks':result['checks'],'steps':result['steps'],'energy_error':result['energy_relative_error']}),flush=True)
        except Exception as exc:
            results[name]={'error':repr(exc)}
            (a.output/f'{name}-error.json').write_text(json.dumps(results[name],indent=2)+'\n');print(json.dumps(results[name]),flush=True)
    (a.output/'verification.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
