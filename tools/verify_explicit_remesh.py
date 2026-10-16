"""Coupled explicit run with a manufactured poor internal triangulation.

An exact cube boundary is retained. A deliberately flattened internal
bipyramid has a valid quality-improving 2-to-3 reconnection. Initial PEEQ=0.2
is manufactured history for transfer verification, not measured prestrain.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from packing_fem.explicit_material import oriented,geometry
from packing_fem.explicit_solver import solve
from packing_fem.adaptive_tet import quality


def seed_bipyramid(X,t,owner,radius):
    outer=X[t[owner]];center=outer.mean(axis=0)
    inner=np.array([[radius,0,0],[-radius/2,np.sqrt(3)*radius/2,0],[-radius/2,-np.sqrt(3)*radius/2,0],[0,0,.15*radius],[0,0,-.15*radius]])+center
    local=np.r_[outer,inner];part=oriented(local,Delaunay(local).simplices)
    remove=np.array([7 in row and 8 in row for row in part]);assert remove.sum()==3
    part=oriented(local,np.r_[part[~remove],[[4,5,6,7],[4,6,5,8]]])
    assert np.isclose(geometry(local,part)[0].sum(),geometry(X,t[owner:owner+1])[0].sum(),rtol=1e-12)
    ids=np.r_[t[owner],np.arange(len(X),len(X)+5)]
    return np.r_[X,inner],np.r_[np.delete(t,owner,axis=0),ids[part]]


def fixture():
    from packing_fem import sintered
    from packing_fem.explicit_material import hex_to_tet
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),2.)
    return seed_bipyramid(X,hex_to_tet(X,h),0,.15)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    X,t=fixture();c=json.loads(Path('configs/explicit-prototype.json').read_text());c.update(length_mm=2.,initial_peeq=.2,compression_fraction=.002,remesh_every=10,remesh_quality=.6,contact_penalty_mpa_mm=1.,max_steps=30000)
    r=solve(a.output,X,t,c);assert any(e['accepted'] for e in r['remesh_events']),r
    print(json.dumps(r,indent=2),flush=True)

if __name__=='__main__':main()
