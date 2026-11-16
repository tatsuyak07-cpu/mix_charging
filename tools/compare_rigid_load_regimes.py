"""Bound-to-state comparison of mesh errors at two loads with unchanged physics."""
import argparse,csv,json
from pathlib import Path
from tools.compare_rigid_mesh_accuracy import read_study


def compare(small,loaded,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    a,first=read_study(small);b,second=read_study(loaded)
    if len(first)!=len(second) or len(first)<3:raise ValueError('Need matching three-level studies')
    rows=[]
    for x,y in zip(first,second,strict=True):
        if any(x[k]!=y[k] for k in ['level','tetrahedra','mesh_sha256','physical_sha256']):
            raise ValueError('Mesh or original physics differs')
        if not 0<x['compression']<y['compression']:raise ValueError('Loads are not ordered')
        rows.append(dict(level=x['level'],tetrahedra=x['tetrahedra'],
            small_compression=x['compression'],loaded_compression=y['compression'],
            small_pressure_MPa=x['pressure_MPa'],loaded_pressure_MPa=y['pressure_MPa'],
            small_secant_modulus_MPa=x['pressure_MPa']/x['compression'],
            loaded_secant_modulus_MPa=y['pressure_MPa']/y['compression'],
            small_maximum_ep=x['maximum_ep'],loaded_maximum_ep=y['maximum_ep'],
            small_plastic_fraction=x['reference_volume_plastic_fraction'],
            loaded_plastic_fraction=y['reference_volume_plastic_fraction']))
    elastic=all(r['small_maximum_ep']==0. for r in rows)
    report=dict(scope='Initial-load diagnosis only. Original material parameters, geometry and boundary conditions retained.',
        small_load_study=a,loaded_study=b,rows=rows,
        small_load_is_elastic_verified_from_saved_history=elastic,
        mesh_difference_exists_before_plasticity=bool(elastic and a['production_coarsest_vs_finest']['pressure_relative']>a['thresholds']['pressure_relative']),
        locking_or_unique_mechanical_cause_established=False,
        full_loaded_path_convergence_verified=False)
    (out/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'comparison.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');counts=[r['tetrahedra'] for r in rows]
    for key,label in [('small',f"{100*first[0]['compression']:g}% compression"),('loaded',f"{100*second[0]['compression']:g}% compression")]:
        axes[0].plot(counts,[r[key+'_secant_modulus_MPa'] for r in rows],'o-',label=label)
        axes[1].plot(counts,[100*r[key+'_plastic_fraction'] for r in rows],'o-',label=label)
    axes[0].set(ylabel='Pressure / axial compression (MPa)',title='Macroscopic secant response')
    axes[1].set(ylabel='Reference-volume plastic fraction (%)',title='Saved plastic history')
    for ax in axes:
        ax.set_xlabel('T4 elements');ax.set_xticks(counts,[f'{count:,}' for count in counts]);ax.grid(alpha=.25);ax.legend()
    fig.savefig(out/'load-regimes.png',dpi=180);plt.close(fig)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--small',required=True);p.add_argument('--loaded',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();compare(a.small,a.loaded,a.out)
