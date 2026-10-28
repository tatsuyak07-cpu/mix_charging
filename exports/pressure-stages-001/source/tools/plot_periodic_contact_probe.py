"""Plot archived independent-probe and known-patch friction verification."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    periodic=np.genfromtxt(a.source/'periodic-path.csv',delimiter=',',names=True)
    seam=np.genfromtxt(a.source/'friction-seam.csv',delimiter=',',names=True)
    rotation=np.genfromtxt(a.source/'friction-rotation.csv',delimiter=',',names=True)
    fig,ax=plt.subplots(2,2,figsize=(11,8))
    for n in (1,2):
        h=periodic[(periodic['refinement']==n)&(periodic['axis']==0)]
        ax[0,0].plot(h['coordinate_mm'],-h['fy_N'],label=f'Surface spacing {1/n:g} mm')
    ax[0,0].axvline(0,color='grey',linestyle=':');ax[0,0].set_xlabel('Coordinate across periodic cut [mm]')
    ax[0,0].set_ylabel('Normal force magnitude [N]');ax[0,0].set_ylim(3.95,4.05)
    for samples in (201,401):
        h=seam[seam['samples']==samples]
        ax[0,1].plot(h['coordinate_mm'],-h['fx_N'],label=f'{samples} samples')
    ax[0,1].axvline(.43,color='grey',linestyle=':',label='Triangle change')
    ax[0,1].axhline(.6,color='black',linewidth=.8,linestyle='--',label='Coulomb cap')
    ax[0,1].set_xlabel('Tangential coordinate [mm]');ax[0,1].set_ylabel('Friction force magnitude [N]')
    h=seam[seam['samples']==201]
    for name,label in [('work_Nmm','Backward-Euler work'),('friction_Nmm','Friction dissipation'),('elastic_Nmm','Elastic energy'),('algorithmic_Nmm','Algorithmic loss')]:
        ax[1,0].plot(h['coordinate_mm'],h[name],label=label)
    ax[1,0].set_xlabel('Tangential coordinate [mm]');ax[1,0].set_ylabel('Energy [N mm]')
    ax[1,1].plot(rotation['theta_rad']*180/np.pi,rotation['fx_N'],label='Friction x')
    ax[1,1].plot(rotation['theta_rad']*180/np.pi,rotation['fy_N'],label='Friction y')
    ax[1,1].set_xlabel('Prescribed normal rotation [degrees]');ax[1,1].set_ylabel('Friction force [N]')
    for axis in ax.ravel():axis.grid(True);axis.legend(fontsize=8)
    fig.suptitle('Periodic independent probe and known-patch friction: NOT a compaction solution')
    fig.tight_layout();fig.savefig(a.source/'verification.png',dpi=160)

if __name__=='__main__':main()
