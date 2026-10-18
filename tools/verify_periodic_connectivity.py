"""Periodic solid connectivity and winding; topology is not a stiffness proof."""
import argparse,hashlib,json
from collections import deque
from pathlib import Path
import numpy as np
from packing_fem.explicit_periodic import Periodic
from packing_fem.explicit_material import geometry
from tools.verify_plane_contact import fixture as layered
from tools.run_barrier_compaction import fixture as connected


def winding(X,t,L):
    p=Periodic(X,L);classes=p.classes;count=int(classes.max())+1
    image=np.rint((X-np.mod(X,L))/L).astype(int)
    # Match the same tolerance-based aliases as the periodic solver.
    canonical=np.mod(X,L);near=np.isclose(canonical,L,atol=1e-9,rtol=0)
    image+=near.astype(int)
    edges=[[] for _ in range(count)]
    for tet in t:
        for node in tet[1:]:
            i,j=int(classes[tet[0]]),int(classes[node]);d=image[node]-image[tet[0]]
            edges[i].append((j,d));edges[j].append((i,-d))
    labels=np.full(count,-1,int);offset=np.zeros((count,3),int);components=[]
    for start in range(count):
        if labels[start]>=0:continue
        label=len(components);labels[start]=label;queue=deque([start]);nodes=[];cycles=set()
        while queue:
            i=queue.popleft();nodes.append(i)
            for j,d in edges[i]:
                proposed=offset[i]+d
                if labels[j]<0:labels[j]=label;offset[j]=proposed;queue.append(j)
                else:
                    cycle=proposed-offset[j]
                    if np.any(cycle):cycles.add(tuple(map(int,cycle)))
        vectors=np.array(sorted(cycles),int).reshape(-1,3);rank=int(np.linalg.matrix_rank(vectors.astype(float))) if len(vectors) else 0
        components.append(dict(periodic_nodes=len(nodes),winding_rank=rank,winding_vectors=vectors.tolist(),winds_x=bool(len(vectors) and np.any(vectors[:,0])),winds_y=bool(len(vectors) and np.any(vectors[:,1])),winds_z=bool(len(vectors) and np.any(vectors[:,2]))))
    volumes=geometry(X,t)[0];which=labels[classes[t[:,0]]]
    for k,c in enumerate(components):c['reference_solid_volume_mm3']=float(volumes[which==k].sum())
    return dict(nodes=len(X),tetrahedra=len(t),length_mm=L,initial_porosity=float(1-volumes.sum()/L**3),periodic_components=len(components),components=components)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    X,t,*_=layered(1,.015,0.);rows=dict(layered_1mm=winding(X,t,1.));X,t=connected();rows['connected_4mm']=winding(X,t,4.)
    source=Path('exports/explicit-fem-prototype-002/prepared-40percent-model');mesh=source/'model-mesh.npz'
    with np.load(mesh) as z:X=z['points'];t=z['tets']
    c=json.loads((source/'config.json').read_text());rows['prepared_packing']=dict(winding(X,t,c['length_mm']),source=str(source),mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest())
    checks=dict(layered_no_axial_winding=not any(c['winds_z'] for c in rows['layered_1mm']['components']),connected_fixture_spans_3D=any(c['winding_rank']==3 for c in rows['connected_4mm']['components']),prepared_packing_spans_3D=any(c['winding_rank']==3 for c in rows['prepared_packing']['components']))
    summary=dict(scope='periodic_mesh_topology_only_not_a_stiffness_or_compaction_solution',models=rows,checks=checks,initial_packing_stiffness_verified=False,particle_compaction_completed=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'verification-script.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(summary,indent=2));raise SystemExit(0 if all(checks.values()) else 2)
if __name__=='__main__':main()
