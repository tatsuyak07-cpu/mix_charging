"""Plot accepted integration states, not rejected crossing configurations."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from packing_fem.adaptive_tet import boundary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stages',type=Path,required=True);p.add_argument('--impact',type=Path,required=True);a=p.parse_args()
    track=np.genfromtxt(a.stages/'tracking.csv',names=True,delimiter=',')
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    ax[0].plot(track['coordinate_mm'],track['tangent_Nmm']);ax[0].set_xlabel('Tangential coordinate [mm]');ax[0].set_ylabel('Spring energy [N mm]')
    for i in np.flatnonzero(track['transferred_pairs']):ax[0].axvline(track['coordinate_mm'][i],color='grey',linestyle=':')
    for name in ('base','half_dt'):
        h=np.genfromtxt(a.impact/name/'history.csv',names=True,delimiter=',')
        ax[1].plot(h['time_s']*1e6,h['penetration_mm']*1e3,label=name)
        ax[2].plot(h['time_s']*1e6,h['energy_residual_Nmm'],label=name)
    ax[1].set_ylabel('Penetration [micrometres]');ax[2].set_ylabel('Energy residual [N mm]')
    for i in (1,2):ax[i].set_xlabel('Numerical time [microseconds]');ax[i].legend()
    for axis in ax:axis.grid(True)
    fig.suptitle('Tracked contact: manufactured dynamic verification, NOT quasi-static compaction')
    fig.tight_layout();fig.savefig(a.stages/'integration-verification.png',dpi=160)
    with np.load(a.impact/'base/last-state.npz') as m:
        X=m['reference_points'];x=m['points'];t=m['tets']
    surface=boundary(t[:-1]);tri=X[surface]
    pore=np.all((tri>=1-1e-10)&(tri<=2+1e-10),axis=(1,2))&np.all(np.isclose(tri[:,:,2],1.),axis=1);surface=surface[pore]
    fig=plt.figure(figsize=(10,5))
    for i,(points,label) in enumerate(((X,'Initial mesh'),(x,'Last accepted mesh'))):
        axis=fig.add_subplot(1,2,i+1,projection='3d')
        edges=np.unique(np.sort(np.concatenate([surface[:,[0,1]],surface[:,[1,2]],surface[:,[2,0]]]),axis=1),axis=0)
        axis.add_collection3d(Line3DCollection(points[edges],colors='grey',linewidths=.7))
        tip=t[-1];pairs=np.array([[tip[i],tip[j]] for i in range(4) for j in range(i+1,4)])
        axis.add_collection3d(Line3DCollection(points[pairs],colors='crimson',linewidths=1.5))
        axis.scatter(*points[tip[0]],color='crimson',s=15)
        axis.set(xlim=(1,2),ylim=(1,2),zlim=(.97,1.4),xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title=label)
        axis.set_box_aspect((1,1,.6));axis.view_init(22,-60)
    fig.suptitle('Actual pore-floor surface and tip tetra mesh; displacement scale 1')
    fig.tight_layout();fig.savefig(a.stages/'integration-mesh.png',dpi=160)

if __name__=='__main__':main()
