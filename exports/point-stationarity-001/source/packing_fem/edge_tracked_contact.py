"""Opt-in interior edge-pair response; ambiguous feature handoff still stops."""
import numpy as np
from .tracked_contact import TrackedContact
from .edge_contact import start_pair,response
from .explicit_periodic import affine_basis


class EdgeTrackedContact(TrackedContact):
    def __init__(self,*args,edge_penalty_scale=1.,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.isfinite(edge_penalty_scale) or edge_penalty_scale<=0:raise ValueError('Positive edge penalty scale required')
        self.edge_penalty_scale=edge_penalty_scale;self.edge_history={};self.trial_edge_history={}
        f=self.faces
        self.edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)

    def accept_geometry(self,x,H):
        super().accept_geometry(x,H)
        self.edge_history=self.trial_edge_history

    def handle_edge_event(self,i,j,shift,before,after,event):
        key=(i,j,*shift)
        if key in self.edge_history:return
        if event['reason']!='swept edge intersection':raise ValueError('Ambiguous edge contact: '+event['reason'])
        self.events[key]=start_pair(before,after,event['time_fraction'])

    def evaluate(self,x,velocity,H,Hdot,dt):
        self.events={};force,extra,hist,report=super().evaluate(x,velocity,H,Hdot,dt)
        trial={};report['edge_contacts']=0
        states={**self.events,**self.edge_history}
        for key,state in states.items():
            i,j,*shift=key;a=self.edges[i];b=self.edges[j];shift=np.asarray(shift)
            points=np.r_[x[a],x[b]+H@shift]
            kn=self.edge_penalty_scale*self.kn*.125*float(self.node_area0[np.r_[a,b]].sum())
            result=response(points,state,kn,.5*kn,self.mu)
            if result is None:
                report['released_history_energy']+=state['energy'];continue
            # Do not silently sum two different quadratures for the same event.
            for node,face,*image in hist:
                ids=self.faces[face]
                ab=(node in a and set(b).issubset(ids) and tuple(image)==tuple(shift))
                ba=(node in b and set(a).issubset(ids) and tuple(image)==tuple(-shift))
                if ab or ba:raise ValueError('Edge-node contact handoff needs energy-consistent transfer')
            trial[key]=result['state'];ff=result['force']
            np.add.at(force,a,ff[:2]);np.add.at(force,b,ff[2:])
            extra+=np.einsum('aij,ai->j',affine_basis(np.repeat((shift*self.L)[None],2,axis=0)),ff[2:])
            for name in ('normal_energy','tangent_energy','friction_increment','algorithmic_increment'):report[name]+=result[name]
            report['penetration_mm']=max(report['penetration_mm'],result['depth_mm'])
            report['max_coulomb_ratio']=max(report['max_coulomb_ratio'],result['coulomb_ratio'])
            report['contacts']+=1;report['edge_contacts']+=1
        self.trial_edge_history=trial
        return force,extra,hist,report

    def edge_stiffness_diagonal(self):
        # A conservative scalar assembly estimate for the active spring part.
        # Rotating-frame geometric terms are not a rigorous spectral bound.
        diagonal=np.zeros(len(self.X))
        for i,j,*shift in self.edge_history:
            ids=np.r_[self.edges[i],self.edges[j]]
            kn=self.edge_penalty_scale*self.kn*.125*float(self.node_area0[ids].sum())
            np.add.at(diagonal,ids,3*kn)
        return diagonal
