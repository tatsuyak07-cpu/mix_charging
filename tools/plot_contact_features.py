"""Plot manufactured contact verification CSV, without altering raw data."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    h=np.genfromtxt(a.source/'edge-path.csv',names=True,delimiter=',')
    base=h[(h['mesh_refinement']==1)&(h['samples']==201)]
    fig,ax=plt.subplots(2,1,figsize=(8,7),sharex=True)
    angle=base['theta_rad']*180/np.pi
    ax[0].plot(angle,base['analytic_energy_Nmm'],'k-',label='Analytic')
    ax[0].plot(angle,base['energy_Nmm'],'--',label='Closed-surface closest feature')
    ax[0].plot(angle,base['projection_only_energy_Nmm'],':',label='Plane-projection criterion only')
    ax[0].set_ylabel('Normal penalty energy [N mm]')
    for component in ('x','y'):
        ax[1].plot(angle,base[f'f{component}_N'],label=f'Force {component}')
        ax[1].plot(angle,base[f'analytic_f{component}_N'],'k:',linewidth=.8)
    ax[1].set_ylabel('Probe force [N]');ax[1].set_xlabel('Path angle [degrees]')
    for axis in ax:
        axis.axvline(0,color='grey',linewidth=.7);axis.axvline(90,color='grey',linewidth=.7)
        axis.grid(True);axis.legend()
    fig.suptitle('Manufactured concave edge: geometry verification only')
    fig.tight_layout();fig.savefig(a.source/'concave-edge-verification.png',dpi=160)

if __name__=='__main__':main()
