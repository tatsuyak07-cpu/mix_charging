"""Manufactured normal-contact tests; does not run or validate compaction."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from packing_fem import sintered
from packing_fem.explicit_material import hex_to_tet
from packing_fem.adaptive_tet import boundary
from packing_fem.contact_geometry import closed_probe


def surface(mask,h):
    x,hexa,_=sintered.mesh(mask,h)
    return x,boundary(hex_to_tet(x,hexa))


def projection_only(point,x,faces,k,skin=.2):
    # Isolate the former plane-projection/barycentric geometric criterion.
    # This omits slave normals, friction, two-pass areas and topology and is
    # NOT a substitute replay of the full Contact kernel.
    distances=[]
    for tri in x[faces]:
        n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);n/=np.linalg.norm(n)
        gap=(point-tri[0])@n
        if not -skin<gap<0:continue
        edges=(tri[1:]-tri[0]).T
        uv=np.linalg.lstsq(edges,point-gap*n-tri[0],rcond=None)[0]
        if min(1-uv.sum(),*uv)>=-1e-9:distances.append(-gap)
    return .5*k*min(distances)**2 if distances else 0.


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);k=100.;r=.1;rows=[];checks={}
    for refinement in (1,2):
        mask=np.ones((2*refinement,2*refinement,refinement),bool);mask[refinement:,refinement:,:]=False
        x,f=surface(mask,1/refinement)
        np.savez_compressed(a.output/f'concave-edge-mesh-{refinement}.npz',points=x,faces=f)
        for samples in (101,201):
            force_errors=[];energy_errors=[];forces=[];energies=[]
            for theta in np.linspace(-.3,np.pi/2+.3,samples):
                pnt=np.array([1-r*np.cos(theta),1-r*np.sin(theta),.5]);result=closed_probe(pnt,x,f,k)
                force=k*r*np.array([np.cos(theta),0,0]) if theta<0 else (k*r*np.array([0,np.sin(theta),0]) if theta>np.pi/2 else k*r*np.array([np.cos(theta),np.sin(theta),0]))
                exact=.5*(force@force)/k
                energy_errors.append(abs(result['energy']-exact))
                force_errors.append(float(np.max(abs(result['force']-force))))
                forces.append(result['force']);energies.append(result['energy'])
                rows.append([refinement,samples,theta,*pnt,result['energy'],exact,*result['force'],*force,projection_only(pnt,x,f,k)])
            label=f'edge_refine{refinement}_samples{samples}'
            checks[label]=dict(max_force_error_N=max(force_errors),max_energy_error_Nmm=max(energy_errors),passed=bool(max(force_errors)<1e-10 and max(energy_errors)<1e-10),
                largest_sample_force_increment_N=float(np.max(np.linalg.norm(np.diff(forces,axis=0),axis=1))),
                largest_sample_energy_increment_Nmm=float(np.max(abs(np.diff(energies)))))
    # Plane triangulation seam: fixed penetration, tangential translation.
    x,f=surface(np.ones((1,1,1),bool),1.)
    seam=[closed_probe(np.array([u,.43,.02]),x,f,k) for u in np.linspace(.3,.6,101)]
    seam_energy=max(abs(s['energy']-.5*k*.02**2) for s in seam)
    seam_force=max(float(np.max(abs(s['force']-np.array([0,0,-2.])))) for s in seam)
    checks['coplanar_seam']=dict(max_force_error_N=seam_force,max_energy_error_Nmm=seam_energy,passed=bool(seam_force<1e-10 and seam_energy<1e-10))
    # Concave cavity corner gives a genuine closest VERTEX, not a face projection.
    mask=np.ones((3,3,3),bool);mask[1,1,1]=False;x,f=surface(mask,1.)
    vertex_energy=[];vertex_force=[]
    for d in (.03,.06,.1):
        result=closed_probe(np.array([1-d]*3),x,f,k)
        vertex_energy.append(abs(result['energy']-.5*k*3*d*d))
        vertex_force.append(float(np.max(abs(result['force']-k*d))))
    checks['concave_vertex']=dict(max_force_error_N=max(vertex_force),max_energy_error_Nmm=max(vertex_energy),passed=bool(max(vertex_force)<1e-10 and max(vertex_energy)<1e-10))
    # Reject external points even if they lie behind the extension of a face.
    outside=[closed_probe(np.array(q),x,f,k) for q in ([1.02,1.02,1.02],[1.1,1.2,1.3],[-.01,.2,.2])]
    checks['outside_no_contact']=dict(passed=all(not z['inside'] and z['energy']==0 for z in outside))
    with (a.output/'edge-path.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['mesh_refinement','samples','theta_rad','x_mm','y_mm','z_mm','energy_Nmm','analytic_energy_Nmm','fx_N','fy_N','fz_N','analytic_fx_N','analytic_fy_N','analytic_fz_N','projection_only_energy_Nmm']);writer.writerows(rows)
    summary=dict(purpose='closed-surface frictionless geometric verification only',stiffness_N_mm=k,
        radius_mm=r,checks=checks,all_checks_passed=all(v['passed'] for v in checks.values()),
        integrated_with_explicit_solver=False,periodic_self_contact_validated=False,compaction_40_to_10_validated=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if not summary['all_checks_passed']:raise RuntimeError('Feature verification failed')

if __name__=='__main__':main()
