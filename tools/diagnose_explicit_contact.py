"""Replay contact activation without changing forces or acceptance criteria."""
import argparse
import json
from pathlib import Path
import numpy as np
from packing_fem.explicit_solver import solve
from packing_fem.periodic_contact import Contact


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cfl-scale',type=float,default=1.)
    a=p.parse_args()
    if not 0<a.cfl_scale<=1:raise ValueError('CFL scale must be in (0,1]')
    c=json.loads((a.source/'config.json').read_text())
    c['cfl']*=a.cfl_scale
    c['output_every']=1  # Diagnostic sampling only; integration is unchanged.
    with np.load(a.source/'initial-mesh.npz') as m:
        X,t=m['points'],m['tets']
    original=Contact.evaluate
    events=[]
    elapsed=0.
    previous=None
    def observed(self,x,velocity,H,Hdot,dt):
        nonlocal elapsed,previous
        elapsed+=dt
        result=original(self,x,velocity,H,Hdot,dt)
        keys=set(result[2])-set(self.history)
        if keys:
            entries=[]
            for node,face,*shift in sorted(keys):
                tri=x[self.faces[face]]+H@np.asarray(shift)
                n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);n/=np.linalg.norm(n)
                depth=-float((x[node]-tri[0])@n)
                entry=dict(node=node,face=face,shift=shift,depth_mm=depth,
                    normal_energy_Nmm=.25*self.kn*self.node_area0[node]*depth**2)
                if previous is not None:
                    px,pH=previous;pt=px[self.faces];pc=np.cross(pt[:,1]-pt[:,0],pt[:,2]-pt[:,0])
                    pn=pc/np.linalg.norm(pc,axis=1)[:,None];nn=np.zeros_like(px)
                    for k in range(3):np.add.at(nn,self.faces[:,k],pc)
                    nn/=np.maximum(np.linalg.norm(nn,axis=1)[:,None],1e-30)
                    master=pt[face]+pH@np.asarray(shift);gap=float((px[node]-master[0])@pn[face])
                    edges=(master[1:]-master[0]).T
                    uv=np.linalg.lstsq(edges,px[node]-gap*pn[face]-master[0],rcond=None)[0]
                    entry['previous_geometry']=dict(gap_mm=gap,facing_dot=float(nn[node]@pn[face]),
                        minimum_barycentric=float(min(1-uv.sum(),*uv)),adjacent=self.adjacent(node,face,shift))
                entries.append(entry)
            events.append(dict(evaluation_time_s=elapsed,new_pairs=entries,
                normal_energy_Nmm=result[3]['normal_energy']))
        previous=(x.copy(),H.copy())
        return result
    Contact.evaluate=observed
    try:
        summary=solve(a.output,X,t,c)
    finally:
        Contact.evaluate=original
    (a.output/'activation-events.json').write_text(json.dumps(events,indent=2)+'\n')
    h=np.genfromtxt(a.output/'history.csv',delimiter=',',names=True)
    active=np.flatnonzero(h['contacts']>0)
    i=int(active[0]) if len(active) else None
    diagnostic={'first_contact_index':i,'first_contact':None if i is None else
        {k:float(h[k][i]) for k in h.dtype.names},
        'largest_energy_increment_Nmm':float(np.max(np.abs(np.diff(h['energy_residual_Nmm'])))),
        'max_energy_residual_Nmm':float(np.max(np.abs(h['energy_residual_Nmm']))),
        'new_pair_events':len(events),'status':summary['status'],
        'interpretation':'Dynamic diagnostic only; event evaluations include the final rejected trial if any.'}
    (a.output/'diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    print(json.dumps(diagnostic,indent=2))

if __name__=='__main__':main()
