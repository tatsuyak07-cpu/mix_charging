"""Plot accepted edge verification histories and actual mesh edges."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from packing_fem.tracked_contact import TrackedContact
from packing_fem.explicit_periodic import Periodic


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pair',type=Path,required=True);p.add_argument('--solver',type=Path,required=True);a=p.parse_args()
    data=np.genfromtxt(a.pair/'pair-path.csv',names=True,delimiter=',');fig,ax=plt.subplots(1,3,figsize=(13,4))
    for n in (100,200):
        h=data[data['intervals']==n];ax[0].plot(h['path_fraction'],h['work_Nmm'],label=f'{n} intervals')
    ax[0].plot(h['path_fraction'],.0008*h['path_fraction']**2,'k:',label='Analytic');ax[0].set(xlabel='Prescribed path fraction',ylabel='Work [N mm]')
    for name in ('base','half_dt','edge_penalty2'):
        h=np.genfromtxt(a.solver/name/'history.csv',names=True,delimiter=',');last=h['time_s'][-1];tail=h['time_s']>last-2e-7
        ax[1].step(h['time_s'][tail]*1e6,h['edge_contacts'][tail],where='post',label=name)
        ax[2].plot(h['time_s'][tail]*1e6,h['normal_contact_Nmm'][tail],label=name)
    ax[1].set(xlabel='Numerical time [microseconds]',ylabel='Active interior edge pairs')
    ax[2].set(xlabel='Numerical time [microseconds]',ylabel='Normal contact energy [N mm]')
    for axis in ax:axis.grid(True);axis.legend(fontsize=8)
    fig.suptitle('Interior edge response verified; slit paths STOP at unresolved feature handoff')
    fig.tight_layout();fig.savefig(a.solver/'edge-verification.png',dpi=160)
    with np.load(a.solver/'base/initial-mesh.npz') as m:X,t=m['points'],m['tets']
    with np.load(a.solver/'base/last-state.npz') as m:x=m['points']
    contact=TrackedContact(X,t,Periodic(X,5.),20.,.3,.1);faces=contact.faces
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    local=np.all(X[faces][:,:,0]>=1.9,axis=1)&np.all(X[faces][:,:,0]<=3.1,axis=1)
    local&=np.all((X[faces][:,:,1]>=1.9)&(X[faces][:,:,1]<=3.1),axis=1)
    selected=faces[local];lines=x[selected][:,[0,1,2,0]]
    fig=plt.figure(figsize=(7,6));axis=fig.add_subplot(111,projection='3d')
    axis.add_collection3d(Line3DCollection(lines,colors='grey',linewidths=.5))
    for edge,color in ((37,'crimson'),(47,'royalblue')):
        points=x[edges[edge]];axis.plot(*points.T,color=color,linewidth=2,label=f'Edge {edge}')
    axis.set(xlim=(1.95,3.05),ylim=(1.95,3.05),zlim=(2.44,2.56),xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]')
    axis.set_box_aspect((1,1,.35));axis.legend();axis.view_init(20,-60)
    fig.suptitle('Actual local mesh at last accepted state; displacement scale 1')
    fig.tight_layout();fig.savefig(a.solver/'edge-mesh.png',dpi=160)

if __name__=='__main__':main()
