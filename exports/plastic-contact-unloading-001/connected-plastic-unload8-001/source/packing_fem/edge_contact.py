"""Experimental oriented INTERIOR edge-pair contact potential.

Persistent contact is created by a swept crossing. Parallel edges and endpoint
handoff are rejected. Complex-step differentiation includes rotating-frame
terms: distributing a tangential traction alone would leave a spurious couple.
"""
import numpy as np
from .contact_tracking import closest_segments


def frame(points,orientation):
    u=points[1]-points[0];v=points[3]-points[2]
    tangent=u/np.sqrt(u@u);cross=np.cross(u,v)
    normal=orientation*cross/np.sqrt(cross@cross)
    return np.stack([tangent,np.cross(normal,tangent),normal],axis=1)


def local(points,state):
    R=frame(points,state['orientation']);a=state['s'];b=state['t']
    relative=(1-a)*points[0]+a*points[1]-(1-b)*points[2]-b*points[3]
    return R[:,:2].T@relative-state['origin']


def start_pair(before,after,event_fraction):
    at=before+event_fraction*(after-before);hit=closest_segments(*at)
    if min(hit['s'],1-hit['s'],hit['t'],1-hit['t'])<=1e-9:raise ValueError('Edge endpoint handoff needs separate contact transition')
    u=before[1]-before[0];v=before[3]-before[2];cross=np.cross(u,v)
    if np.linalg.norm(cross)<1e-8*np.linalg.norm(u)*np.linalg.norm(v):raise ValueError('Parallel edge response is undefined')
    gap=float((before[2]-before[0])@cross/np.linalg.norm(cross))
    if abs(gap)<1e-12:raise ValueError('Initial edge side is ambiguous')
    state=dict(orientation=float(np.sign(gap)),s=hit['s'],t=hit['t'],origin=np.zeros(2),plastic=np.zeros(2),spring=np.zeros(2),coordinate=np.zeros(2),energy=0.)
    state['origin']=local(at,state)
    return state


def potential(points,state,kn,kt,plastic):
    R=frame(points,state['orientation']);g=(points[2]-points[0])@R[:,2]
    spring=local(points,state)-plastic
    return .5*kn*g*g+.5*kt*(spring@spring)


def response(points,state,kn,kt,mu):
    points=np.asarray(points,float)
    if min(kn,kt)<=0 or mu<0:raise ValueError('Invalid edge contact coefficients')
    u=points[1]-points[0];v=points[3]-points[2]
    if np.linalg.norm(np.cross(u,v))<1e-8*np.linalg.norm(u)*np.linalg.norm(v):raise ValueError('Parallel edge response is undefined')
    hit=closest_segments(*points)
    R=frame(points,state['orientation']);gap=float((points[2]-points[0])@R[:,2])
    if gap>=0:return None
    if min(hit['s'],1-hit['s'],hit['t'],1-hit['t'])<=1e-9:raise ValueError('Active edge endpoint handoff is not implemented')
    coordinate=local(points,state);trial=coordinate-state['plastic'];fn=-kn*gap
    size=np.linalg.norm(trial);spring=trial*min(1.,mu*fn/(kt*max(size,1e-300)))
    plastic=coordinate-spring;increment=plastic-state['plastic']
    friction=float(kt*spring@increment)
    du=coordinate-state['coordinate'];elastic_change=.5*kt*(spring@spring-state['spring']@state['spring'])
    work=.5*kt*float((state['spring']+spring)@du)
    correction=float(work-elastic_change-friction)
    forces=np.zeros((4,3));h=1e-30
    for i in range(4):
        for axis in range(3):
            perturbed=points.astype(complex);perturbed[i,axis]+=1j*h
            forces[i,axis]=-np.imag(potential(perturbed,state,kn,kt,plastic))/h
    updated={**state,'plastic':plastic,'spring':spring,'coordinate':coordinate,'energy':.5*kt*float(spring@spring)}
    return dict(force=forces,state=updated,normal_energy=.5*kn*gap**2,tangent_energy=updated['energy'],
                friction_increment=friction,algorithmic_increment=correction,depth_mm=-gap,
                coulomb_ratio=float(kt*np.linalg.norm(spring)/fn),normal_force_N=fn,
                potential=float(potential(points,state,kn,kt,plastic)))
