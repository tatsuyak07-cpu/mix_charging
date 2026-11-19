"""Explicit opt-in guard for a stress-free virgin contact configuration."""
import numpy as np


def virgin_body(p,q,b,ep):
    n=getattr(p,'body_nmicro',p.nmicro)
    return bool(not np.any(q[:n]) and not np.any(q[-6:]) and not np.any(ep)
                and np.max(abs(b-np.eye(3)),initial=0.)<1e-12)


def inspect(contact,q,c):
    value,g,_,count=contact.evaluate_coupled(q,q)
    force_tol=c.get('implicit_initial_contact_force_tolerance_N',1e-10)
    energy_tol=c.get('implicit_initial_contact_energy_tolerance_Nmm',1e-14)
    if not np.isfinite(force_tol) or force_tol<=0 or not np.isfinite(energy_tol) or energy_tol<0:
        raise ValueError('Invalid stress-free initial contact tolerances')
    p=contact.p;free=float(np.max(abs(g[:p.nmicro]),initial=0.));macro=float(abs(g[-6:]).max()/p.L)
    rigid_ok=not hasattr(contact,'force_balance_ok') or contact.force_balance_ok(q,force_tol)
    passed=bool(np.isfinite(value) and abs(value)<=energy_tol and free<=force_tol and macro<=force_tol and rigid_ok)
    return dict(passed=passed,contact_stencils=count,contact_potential_Nmm=float(value),
        free_contact_force_max_N=free,macro_contact_force_max_N=macro,
        force_tolerance_N=force_tol,energy_tolerance_Nmm=energy_tol,ipc_dhat_mm=c['ipc_dhat_mm'],
        parameters_or_material_history_changed=False)
