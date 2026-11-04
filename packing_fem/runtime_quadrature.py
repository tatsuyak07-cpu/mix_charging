"""Transactional look-ahead reweighting; never change force at an accepted pose.

Preserve the accepted applied potential and committed history energies while
fitting the rejected next pose. A same-pose force correction cannot generally
conserve that same inaccurate force. Unknown sample/history creation rejects.
"""
from copy import deepcopy
import numpy as np
from .embedded_contact import EmbeddedContact
from .checkpoint import contact_state,restore_contact
from .conservative_quadrature import redistribute


def repair(contact,t,accepted_x,accepted_velocity,accepted_H,trial_x,trial_velocity,trial_H,trial_Hdot,dt,accepted_response,prediction_only=False):
    if not contact.parent_groups:raise ValueError('Runtime redistribution requires material parent groups')
    new=EmbeddedContact(contact.X,t,contact.periodic,contact.kn,contact.mu,contact.skin,
                        quadrature_rules=contact.rules,**contact._constructor_options)
    restore_contact(new,deepcopy(contact_state(contact)))
    new.capture_audit=True;new._parent_transfer_probe=True
    try:
        new.evaluate(trial_x,trial_velocity,trial_H,trial_Hdot,dt)
        trial_contexts=deepcopy(list(new.parent_contexts.values()))
        before=new.replay_force_state(accepted_x,accepted_velocity,accepted_H)
        f0=contact.periodic.force(accepted_response[0],accepted_response[1]);f1=contact.periodic.force(before[0],before[1])
        baseline_error=float(np.linalg.norm(f1-f0)/max(np.linalg.norm(f0),1e-16))
        if baseline_error>1e-10:raise ValueError('Accepted contact replay force mismatch')
        for name in ('normal_energy','tangent_energy'):
            if abs(before[-1][name]-accepted_response[-1][name])>1e-8*max(abs(accepted_response[-1][name]),1e-18):raise ValueError('Accepted contact replay energy mismatch')
        references={}
        for face,(nodes,B,w) in new.face_points.items():
            rule=new.rules[str(face)]
            reference=np.asarray(rule.setdefault('runtime_reference_weights',w.tolist()),float)
            if reference.shape!=w.shape:raise ValueError('Runtime reference sample lineage mismatch')
            references.update(zip(map(int,nodes),reference))
        N=len(new.X);identity=dict(weights=np.c_[np.ones(N),np.zeros(N)],supports=np.repeat(np.arange(N)[:,None],2,axis=1))
        report=redistribute(contact,new,identity,additional_contexts=trial_contexts,reference_weights=references)
        new._parent_transfer_probe=False
        after=new.replay_force_state(accepted_x,accepted_velocity,accepted_H)
        f2=contact.periodic.force(after[0],after[1]);work_error=float(np.linalg.norm(f2-f1)/max(np.linalg.norm(f1),1e-16))
        energy_errors={name:abs(after[-1][name]-before[-1][name])/max(abs(before[-1][name]),1e-18) for name in ('normal_energy','tangent_energy')}
        if max(work_error,*energy_errors.values())>1e-8:raise ValueError('Runtime redistribution changes accepted work/energy')
        if abs(after[-1]['friction_increment'])>1e-12 or abs(after[-1]['released_history_energy'])>1e-12:raise ValueError('Runtime redistribution invents dissipation/release')
        accepted_state=deepcopy(contact_state(new)) if prediction_only else None
        response=new.evaluate(trial_x,trial_velocity,trial_H,trial_Hdot,dt)
        if prediction_only:
            # Forecast history is never committed. Rebuild accepted caches and
            # stiffness bounds while retaining the applied lagged potential.
            restore_contact(new,accepted_state)
            response=new.replay_force_state(accepted_x,accepted_velocity,accepted_H)
            new.inner.guard.accept_geometry(new.P@accepted_x,accepted_H)
            new.inner.accepted_audit=deepcopy(new.inner.trial_audit)
            new.inner.accepted_history=new.history
            if work_error>1e-10:raise ValueError('Preemptive replay force mismatch')
        report.update(accepted_replay_error=baseline_error,accepted_work_error=work_error,accepted_energy_errors=energy_errors,
                      parent_audit_passed=True,new_samples=0,history_interpolation=False,fixed_reference_bounds=True,prediction_only=bool(prediction_only))
        new.capture_audit=contact.capture_audit
        return new,response,report
    finally:new._parent_transfer_probe=False
