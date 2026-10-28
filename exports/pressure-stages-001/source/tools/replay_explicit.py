"""Repeat a prepared or archived explicit model in a NEW result directory."""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem.explicit_solver import solve


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-large',action='store_true');a=p.parse_args()
    c=json.loads((a.source/'config.json').read_text());mesh=a.source/'initial-mesh.npz'
    if not mesh.exists():mesh=a.source/'model-mesh.npz'
    with np.load(mesh) as data:X,t=data['points'],data['tets']
    if len(t)>5000 and not a.allow_large:raise ValueError('Research runner exceeds 5000-element budget; --allow-large required')
    print(json.dumps(solve(a.output,X,t,c),indent=2),flush=True)

if __name__=='__main__':main()
