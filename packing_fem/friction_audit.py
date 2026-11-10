"""Weighted Coulomb-stencil and dissipated-work checks for saved IPC states."""
import numpy as np


def friction_report(contact,q,oldq,mu):
    contact.set_lag(q,oldq)
    u=contact.displacement(q,oldq) if hasattr(contact,'displacement') else np.asfortranarray((contact.R@(q-oldq)).reshape(-1,3))
    count=0;negative=0;ratio=0.;excess=0.;reconstruction=0.;coefficient_error=0.;work=0.;positive_work=0.;negative_work=0.;zero_normal_violations=0
    for name in ['vv_collisions','ev_collisions','ee_collisions','fv_collisions']:
        for collision in getattr(contact.lag,name):
            local=collision.dof(u,contact.edges,contact.faces)
            gradient=contact.friction.gradient(collision,local)
            distribution=collision.relative_velocity_jacobian().T@collision.tangent_basis
            tangential=np.linalg.lstsq(distribution,gradient,rcond=None)[0]
            force=float(np.linalg.norm(tangential))
            normal=float(collision.weight*collision.normal_force_magnitude)
            bound=mu*abs(normal)
            if bound>1e-20:ratio=max(ratio,force/bound)
            elif force>1e-10:zero_normal_violations+=1
            excess=max(excess,force-bound)
            reconstruction=max(reconstruction,float(np.max(abs(distribution@tangential-gradient),initial=0.)))
            coefficient_error=max(coefficient_error,abs(float(collision.mu_k)-mu),abs(float(collision.mu_s)-mu))
            contribution=float(local@gradient);work+=contribution;positive_work+=max(0.,contribution);negative_work+=min(0.,contribution);count+=1;negative+=int(collision.weight<0)
    aggregate=float(u.ravel()@contact.friction.gradient(contact.lag,contact.mesh,u))
    report=dict(stencils=count,negative_weight_stencils=negative,friction_coefficient=mu,maximum_coefficient_error=coefficient_error,maximum_weighted_coulomb_ratio=ratio,maximum_bound_excess_N=excess,maximum_tangent_reconstruction_error_N=reconstruction,dissipated_work_Nmm=aggregate,positive_stencil_work_Nmm=positive_work,negative_stencil_work_Nmm=negative_work,zero_normal_force_violations=zero_normal_violations,physical_friction_work_Nmm=-aggregate,work_assembly_error_Nmm=abs(aggregate-work),interpretation='Magnitude bound on weighted integration stencils; negative-weight stencils, if present, are not separate physical contact pairs.')
    report['passed']=bool(zero_normal_violations==0 and np.isfinite(ratio) and excess<1e-9 and reconstruction<1e-9 and coefficient_error<1e-12 and aggregate>=-1e-9 and abs(aggregate-work)<1e-8*max(1.,abs(aggregate)))
    return report
