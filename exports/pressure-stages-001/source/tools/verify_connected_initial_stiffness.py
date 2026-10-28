"""Check that the existing connected 4 mm pore fixture has no closing mechanism.

Small-strain tangent at the unloaded state, with periodic micro DOFs relaxed.
This is not a pressure trajectory, mesh convergence or particle-packing result.
"""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem.explicit_periodic import Periodic
from packing_fem.explicit_material import geometry,advance,internal_force
from tools.run_barrier_compaction import fixture


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--refinement',type=int,default=1);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    X,t=fixture(a.refinement);c=json.loads(Path('configs/explicit-area-cauchy-pressure.json').read_text());V,g,D=geometry(X,t);periodic=Periodic(X,4.)
    E,nu=c['young_mpa'],c['poisson'];mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu));K=np.zeros((3*len(X),3*len(X)))
    for ids,grad,volume in zip(t,g,V):
        for i in range(4):
            for j in range(4):
                block=volume*(lam*np.outer(grad[i],grad[j])+mu*(np.eye(3)*(grad[i]@grad[j])+np.outer(grad[j],grad[i])))
                K[3*ids[i]:3*ids[i]+3,3*ids[j]:3*ids[j]+3]+=block
    A=periodic.A.toarray();reduced=A.T@K@A;n=periodic.nmicro;relax=-np.linalg.solve(reduced[:n,:n],reduced[:n,n:]);macro=reduced[n:,n:]+reduced[n:,:n]@relax
    eig=np.linalg.eigvalsh(macro);microeig=np.linalg.eigvalsh(reduced[:n,:n]);errors=[];eps=1e-7
    b0=np.repeat(np.eye(3)[None],len(t),axis=0);ep=np.zeros(len(t))
    for col in range(6):
        q=np.r_[relax[:,col],np.eye(6)[:,col]];forces=[]
        for sign in (1.,-1.):
            x=periodic.positions(sign*eps*q);inc=geometry(x,t)[2]@np.linalg.inv(D);b,pe=advance(b0,ep,inc,c);fi=internal_force(x,t,b,pe,c)[0];forces.append(-periodic.force(fi))
        derivative=(forces[0]-forces[1])/(2*eps);expected=reduced@q
        errors.append(float(np.linalg.norm(derivative-expected)/np.linalg.norm(expected)))
    checks=dict(micro_stiffness_positive=bool(microeig.min()>1e-10*microeig.max()),relaxed_macro_stiffness_positive=bool(eig.min()>1e-10*eig.max()),tangent_matches_finite_FEM_forces=bool(max(errors)<1e-6),symmetry=bool(np.max(abs(macro-macro.T))<1e-10))
    summary=dict(refinement=a.refinement,scope='initial_elastic_tangent_existing_connected_4mm_fixture_not_particle_packing',nodes=len(X),tetrahedra=len(t),initial_porosity=float(1-V.sum()/4**3),units='N mm for macro stiffness, N/mm for micro stiffness',macro_stiffness=macro.tolist(),macro_eigenvalues=eig.tolist(),micro_min_eigenvalue=float(microeig.min()),six_force_gradient_relative_errors=errors,checks=checks,pressure_trajectory_computed=False,mesh_convergence_verified=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'verification-script.py').write_bytes(Path(__file__).read_bytes());np.savez_compressed(a.output/'initial-mesh.npz',points=X,tets=t);print(json.dumps(summary,indent=2));raise SystemExit(0 if all(checks.values()) else 2)
if __name__=='__main__':main()
