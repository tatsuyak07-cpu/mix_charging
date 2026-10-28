"""Compare nominal-load increments without interpolating missing endpoints."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('coarse',type=Path);p.add_argument('fine',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    datasets=[np.genfromtxt(d/'pressure-porosity.csv',delimiter=',',names=True) for d in (a.coarse,a.fine)]
    rows=[]
    for coarse in datasets[0]:
        matches=datasets[1][np.isclose(datasets[1]['prescribed_nominal_MPa'],coarse['prescribed_nominal_MPa'],rtol=0,atol=1e-10)]
        if len(matches)!=1:continue
        fine=matches[0];strain=coarse['axial_compression_percent'];phi=coarse['geometry_porosity_percent']
        rows.append([coarse['prescribed_nominal_MPa'],strain,fine['axial_compression_percent'],phi,fine['geometry_porosity_percent'],abs(fine['axial_compression_percent']-strain)/abs(strain) if strain else 0,abs(fine['geometry_porosity_percent']-phi)/100])
    with (a.output/'increment-comparison.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['nominal_MPa','coarse_compression_percent','fine_compression_percent','coarse_porosity_percent','fine_porosity_percent','relative_strain_difference','absolute_porosity_difference']);w.writerows(rows)
    report={'sources':[str(a.coarse),str(a.fine)],'common_endpoints':len(rows),'max_relative_strain_difference':max(r[5] for r in rows),'max_absolute_porosity_difference':max(r[6] for r in rows),'interpretation':'Load-increment sensitivity only; not mesh convergence. Both cases retain their failed/incomplete checks.'}
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for h,label,style in zip(datasets,['0.001 MPa increments','0.0001 MPa increments'],['o-','.-']):
        axes[0].plot(h['prescribed_nominal_MPa'],h['geometry_porosity_percent'],style,label=label,markersize=3)
        axes[1].plot(h['prescribed_nominal_MPa'],h['axial_compression_percent'],style,label=label,markersize=3)
    for ax in axes:ax.set_xlabel('Prescribed nominal compression (MPa)');ax.grid(alpha=.3);ax.legend()
    axes[0].set_ylabel('Geometry porosity (%)');axes[1].set_ylabel('Axial compression (%)')
    fig.suptitle('Exploratory periodic 25 mm cell: increasing load\nIncomplete ramps; no self-contact; material and mesh not validated')
    fig.savefig(a.output/'load-increment-comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
