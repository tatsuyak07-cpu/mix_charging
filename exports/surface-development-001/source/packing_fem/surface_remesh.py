"""Conforming periodic edge bisection with transactional state transfer.

This refines existing planar facets, not curved-surface reconstruction or a
quality-repair/coarsening algorithm. Failed contact/energy audits reject the
whole proposal. Reference and current configurations are both retained.
"""
from copy import deepcopy
import numpy as np
from .explicit_material import geometry,oriented,response
from .adaptive_tet import boundary,quality
from .area_contact import frame


def bisect(X,x,t,b,ep,V0,velocity,edge,L):
    """Split every incident tet and every periodic copy of a selected edge."""
    if len(np.unique(X,axis=0))!=len(X):raise ValueError('Coincident distinct nodes require explicit body correspondence')
    edge=np.sort(np.asarray(edge,int))
    if edge.shape!=(2,) or edge[0]==edge[1]:raise ValueError('Two distinct edge nodes required')
    edges=np.unique(np.sort(np.concatenate([t[:,[i,j]] for i in range(4) for j in range(i+1,4)]),axis=1),axis=0)
    if not np.any(np.all(edges==edge,axis=1)):raise ValueError('Selected edge not in mesh')
    # Match translated endpoint pairs, never identify distinct coincident bodies.
    a,bpoint=X[edge];matches=[]
    for pair in edges:
        for ordered in (pair,pair[::-1]):
            shift=(X[ordered[0]]-a)/L
            if np.allclose(shift,np.rint(shift),atol=1e-10,rtol=0) and np.allclose(X[ordered[1]]-bpoint,L*np.rint(shift),atol=L*1e-10,rtol=0):
                matches.append(pair);break
    # One bisection per tet in a transaction, so no hanging nodes or ordering ambiguity.
    incident=[np.flatnonzero(np.sum(np.isin(t,pair),axis=1)==2) for pair in matches]
    flat=np.concatenate(incident)
    if len(np.unique(flat))!=len(flat):raise ValueError('Periodic edge images share a tetrahedron; refine a different edge')
    supports=[(int(i),int(i)) for i in range(len(X))];rows=[];parents=[];removed=set(map(int,flat))
    for pair,owners in zip(matches,incident):
        midpoint=len(supports);supports.append(tuple(map(int,pair)))
        for owner in owners:
            rest=[int(i) for i in t[owner] if i not in pair]
            rows.extend([[int(pair[0]),midpoint,*rest],[midpoint,int(pair[1]),*rest]]);parents.extend([int(owner)]*2)
    keep=[i for i in range(len(t)) if i not in removed];parents=np.r_[keep,parents].astype(int)
    supports=np.asarray(supports);interpolate=lambda value:(value[supports[:,0]]+value[supports[:,1]])/2
    X2=interpolate(X);x2=interpolate(x);t2=oriented(X2,np.r_[t[keep],np.asarray(rows,int)])
    ref=geometry(X2,t2)[0];refold=geometry(X,t)[0];fractions=ref/refold[parents]
    V2=V0[parents]*fractions
    geometry(x2,t2)
    return dict(X=X2,x=x2,t=t2,b=b[parents].copy(),ep=ep[parents].copy(),V0=V2,velocity=interpolate(velocity),supports=supports,parents=parents,
                report=dict(operation='periodic_edge_bisection',periodic_edges=len(matches),new_nodes=len(X2)-len(X),reference_volume_error=float(abs(V2.sum()-V0.sum())),current_volume_error=float(abs(geometry(x2,t2)[0].sum()-geometry(x,t)[0].sum())),minimum_quality_before=float(quality(x,t).min()),minimum_quality_after=float(quality(x2,t2).min())))


def transfer_contact(old,new,proposal,H):
    """Interpolate elastic slip, not plastic offsets, then audit externally.

New active quadrature nodes require matching histories at both edge endpoints.
Unknown patch creation, loss, split, or ambiguous interpolation is rejected.
Old anchor nodes survive bisection and remain material points.
"""
    x=proposal['x'];new.capture_audit=False
    baseline=new.evaluate(x,x*0,H,H*0,0.);fresh=baseline[2];mapped={};covered=set()
    for key,state in fresh.items():
        node=key[1];sources=proposal['supports'][node];states=[]
        for source in sources:
            matches=[]
            for oldkey,value in old.history.items():
                if oldkey[1]!=source:continue
                # Reference target facets must overlap in area, not just at a vertex.
                oldtri=old.X[old.faces[value['face']]]+old.L*np.asarray(value['shift'])
                newtri=new.X[new.faces[state['face']]]+new.L*np.asarray(state['shift'])
                D=np.column_stack((oldtri[1]-oldtri[0],oldtri[2]-oldtri[0]));w=np.linalg.lstsq(D,(newtri-oldtri[0]).T,rcond=None)[0].T
                projected=oldtri[0]+w@D.T
                coplanar=np.max(abs(projected-newtri))<old.L*1e-9
                contained=np.all(w>=-1e-9) and np.all(w.sum(1)<=1+1e-9)
                adjacent=np.min(np.linalg.norm(oldtri[:,None]-newtri[None,:],axis=2))<old.L*1e-9
                if coplanar and (contained or adjacent):matches.append((oldkey,value))
            if len(matches)!=1:raise ValueError('Remesh contact history lacks a unique parent patch')
            matched_key,matched_state=matches[0]
            if node<len(old.X):covered.add(matched_key)
            states.append(matched_state)
        # New history uses its own current material frame and coordinates.
        def world(value,source):
            z=np.zeros((9,3));z[0]=x[source];z[4]=x[value['peer']];z[8]=x[value['second']]
            return frame(z,value['director'])@value['spring']
        world_spring=sum(world(value,int(source)) for source,value in zip(sources,states))/2
        state=deepcopy(state);z=np.zeros((9,3));z[0]=x[node];z[4]=x[state['peer']];z[8]=x[state['second']]
        T=frame(z,state['director']);spring=T.T@world_spring
        if abs(np.linalg.norm(T@spring)-np.linalg.norm(world_spring))>1e-9:raise ValueError('Remesh friction normal mismatch')
        state.update(spring=spring,plastic=state['coordinate']-spring)
        mapped[key]=state
    # Losing active history without accounting for its energy is forbidden.
    if covered!=set(old.history):raise ValueError('Remesh changes active source histories')
    new.history=mapped
    evaluated=new.evaluate(x,x*0,H,H*0,0.)
    new.history=evaluated[2]
    return evaluated


def audit(old,proposal,c,old_contact_report,new_contact_report,tolerance=1e-6,old_response=None,new_response=None):
    """Return explicit acceptance flags; caller commits only if every flag passes."""
    X,x,t,b,ep,V0,velocity=old
    def metrics(t,V,v,z):
        mass=np.zeros(len(z));np.add.at(mass,t.ravel(),np.repeat(V*c['density_tonne_mm3']/4,4))
        return mass.sum(),(mass[:,None]*v).sum(0),np.cross(z,mass[:,None]*v).sum(0),float(.5*np.sum(mass[:,None]*v*v))
    a=metrics(t,V0,velocity,x);p=proposal;d=metrics(p['t'],p['V0'],p['velocity'],p['x'])
    E0=float(V0@response(b,ep,c)[1]);E1=float(p['V0']@response(p['b'],p['ep'],c)[1]);scale=max(abs(E0),a[3],c['young_mpa']*V0.sum()*1e-12)
    energy=lambda r:r['normal_energy']+r['tangent_energy']
    pscale=max(np.sqrt(2*a[0]*scale),1e-30);L=c['length_mm']
    errors=dict(mass=abs(d[0]-a[0])/a[0],linear_momentum=float(np.linalg.norm(d[1]-a[1])/pscale),angular_momentum=float(np.linalg.norm(d[2]-a[2])/(pscale*L)),kinetic_energy=abs(d[3]-a[3])/scale,material_energy=abs(E1-E0)/scale,contact_energy=abs(energy(new_contact_report)-energy(old_contact_report))/max(abs(energy(old_contact_report)),scale*1e-8))
    for name in ('normal_energy','tangent_energy'):
        errors[name]=float(abs(new_contact_report[name]-old_contact_report[name])/max(abs(old_contact_report[name]),scale*1e-8))
    if old_response is not None and new_response is not None:
        from .explicit_periodic import affine_basis
        macro=lambda z,r:np.einsum('aij,ai->j',affine_basis(z),r[0])+r[1]
        force0=macro(X,old_response);force1=macro(p['X'],new_response)
        errors['contact_macro_force']=float(np.linalg.norm(force1-force0)/max(np.linalg.norm(force0),c['young_mpa']*L**3*1e-12))
    checks={name:bool(value<=tolerance) for name,value in errors.items()};checks['coulomb']=new_contact_report['max_coulomb_ratio']<=c['friction']+1e-10
    return dict(accepted=all(checks.values()),checks=checks,scaled_errors=errors,tolerance=tolerance)
