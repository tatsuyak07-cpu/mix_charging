"""Diagnostic capture, not a restart checkpoint; independent energy differences."""
import json
import numpy as np


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def save_capture(path,contact,total_force,external,time,steps):
    if contact.accepted_audit is None:raise ValueError('No accepted audit capture')
    history=[dict(key=list(k),state=plain(v)) for k,v in contact.accepted_history.items()]
    data=dict(schema=1,scope='accepted force-evaluation witnesses plus committed friction history; not a restart checkpoint',time_s=float(time),steps=int(steps),accepted=plain(contact.accepted_audit),committed_history=history,solver_internal_generalized_force=plain(total_force),external_generalized_force=plain(external))
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def potential(z,record):
    ids=record['branch'];a=z[ids[0]]
    if len(ids)==1:closest=a
    else:
        edges=(z[ids[1:]]-a).T;closest=a+edges@np.linalg.lstsq(edges,z[0]-a,rcond=None)[0]
    d=np.linalg.norm(z[0]-closest);delta=record['thickness'];s=(d/delta)**2
    normal=0. if d>=delta else -record['kn']*delta**2/8*(1-s)**2*np.log(s)
    edge=z[4]-z[0];edge/=np.linalg.norm(edge);aux=np.cross(edge,z[8]-z[0]);aux/=np.linalg.norm(aux)
    director=np.asarray(record['director']);n=director[0]*edge+director[1]*np.cross(aux,edge)+director[2]*aux
    tangent=z[4]-z[0];tangent-=np.dot(tangent,n)*n;tangent/=np.linalg.norm(tangent)
    offset=z[0]-np.asarray(record['bary'])@z[5:8];spring=np.array([tangent@offset,np.cross(n,tangent)@offset])-record['plastic']
    # With mu=0 the force law disables the tangential spring entirely.
    return normal+(.5*record['kt']*(spring@spring) if record['mu'] else 0.)


def energy_difference_force(record,step):
    z=np.asarray(record['z'],float);force=np.zeros_like(z)
    for i in range(len(z)):
        for axis in range(3):
            dz=np.zeros_like(z);dz[i,axis]=step
            force[i,axis]=-(potential(z+dz,record)-potential(z-dz,record))/(2*step)
    return force


def reassemble(data,step):
    accepted=data['accepted'];force=np.zeros_like(accepted['points'],dtype=float);extra=np.zeros(6)
    for record in accepted['records']:
        local=energy_difference_force(record,step);np.add.at(force,record['support'],local)
        L=np.asarray(accepted['cell']); # cell columns are current periodic translations
        # Macro variation acts on reference L0*integer image, not current H.
        # The reference cell length is supplied explicitly by the audit caller.
        image=np.asarray(record['images'])*data['reference_length_mm']
        fx,fy,fz=local.T;rx,ry,rz=image.T
        extra+=np.array([fx@rx,fy@ry,fz@rz,fx@ry+fy@rx,fx@rz+fz@rx,fy@rz+fz@ry])
    return force,extra
