"""Connected near-closed slit: explicit integration, self-contact and remeshing.

The narrow manufactured slit is a contact benchmark, not the 40% packing.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from packing_fem import sintered
from packing_fem.explicit_material import oriented,hex_to_tet,geometry
from packing_fem.explicit_solver import solve


def fixture():
    mask=np.ones((5,5,3),bool);mask[1:4,1:4,1]=False
    X,h,_=sintered.mesh(mask,1.);X[:,2]=np.array([0,2.45,2.55,5.])[X[:,2].astype(int)]
    t=hex_to_tet(X,h)
    # Insert a small internal cavity triangulation into one bulk tetrahedron.
    volumes,_,_=geometry(X,t);owner=int(np.argmax(volumes));outer=X[t[owner]];center=outer.mean(axis=0)
    radius=.08
    inner=np.array([[radius,0,0],[-radius/2,np.sqrt(3)*radius/2,0],[-radius/2,-np.sqrt(3)*radius/2,0],[0,0,.15*radius],[0,0,-.15*radius]])+center
    local=np.r_[outer,inner];part=oriented(local,Delaunay(local).simplices)
    remove=np.array([7 in row and 8 in row for row in part]);assert remove.sum()==3
    part=oriented(local,np.r_[part[~remove],[[4,5,6,7],[4,6,5,8]]])
    ids=np.r_[t[owner],np.arange(len(X),len(X)+5)];X=np.r_[X,inner]
    t=np.r_[np.delete(t,owner,axis=0),ids[part]]
    return X,t


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    X,t=fixture();c=json.loads(Path('configs/explicit-prototype.json').read_text());c.update(length_mm=5.,compression_fraction=.04,remesh_every=100,remesh_quality=.36,cfl=.15,max_steps=100000,output_every=100,remesh_energy_tolerance=.001)
    r=solve(a.output,X,t,c)
    print(json.dumps(r,indent=2),flush=True)

if __name__=='__main__':main()
