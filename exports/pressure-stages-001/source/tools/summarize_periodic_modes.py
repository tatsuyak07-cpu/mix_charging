"""Export compact, immutable periodic-mode verification artifacts and plots."""
import argparse,csv,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    cases=['C3D8_n20','C3D8I_n20','C3D8I_n40']; summaries={}; histories={}; energies={}
    names=['config.json','estimate.json','provenance.json','mode-metadata.json','plastic.inp','mode.log','plastic.sta','sintered-geometry.npz','mode-history.csv','mode-energy.csv','mode-summary.json','mode-last-frame.vtu']
    for case in cases:
        src=args.source/case;dst=out/case;dst.mkdir()
        for name in names:shutil.copyfile(src/name,dst/name)
        summaries[case]=json.loads((src/'mode-summary.json').read_text())
        histories[case]=np.genfromtxt(src/'mode-history.csv',names=True,delimiter=',')
        energies[case]=np.genfromtxt(src/'mode-energy.csv',names=True,delimiter=',')
    shutil.copyfile(args.source/'geometry/size-selection.json',out/'size-selection.json')
    metrics=['peak_cauchy_compression_MPa','peak_nominal_compression_MPa','residual_compression_strain','final_geometry_porosity','peak_peeq','max_native_green_strain_error','max_omitting_modes_green_error','macro_micro_work_relative_error','cycle_work_relative_error']
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case',*metrics])
        for case in cases:writer.writerow([case,*[summaries[case][k] for k in metrics]])
    a,b=[summaries[k] for k in cases[1:]]
    differences={k:abs(b[k]-a[k])/abs(a[k]) for k in metrics[:3]}
    additional={}
    for case,h in histories.items():
        additional[case]={'hardening_table_range':bool(np.max(h['peak_peeq'])<=5),
          'physical_material_porosity':bool(np.all((h['material_porosity']>=-1e-8)&(h['material_porosity']<=1+1e-8))),
          'native_EVOL_reference_only':bool(np.allclose(h['solver_reference_EVOL_mm3'],6372,rtol=1e-6))}
    report={'cases':summaries,'additional_reporting_checks':additional,'n20_to_n40_relative_differences':differences,
      'step_count_trial_2percent':all(v<.02 for v in differences.values()),
      'identical_geometry':len({json.loads((out/c/'provenance.json').read_text())['geometry_sha256'] for c in cases})==1,
      'scope':'25 mm cube, 1% displacement compression then release; not 100 mm / 0.1 MPa',
      'limitations':['uncalibrated J2 material','no self-contact or damage','no mesh or RVE-size convergence','native restart reader restricted to CalculiX 2.20 layout'],
      'archive_note':'Helper C3D8 template metadata omitted; mode-metadata.json and plastic.inp describe executed model. Large native DAT/FRD/restart remain in original cloud result directories.'}
    (out/'verification-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for case,h in histories.items():
        axes[0,0].plot(100*h['axial_compression_strain'],h['cauchy_compression_MPa'],label=case)
    axes[0,0].set(xlabel='Axial compression (%)',ylabel='Cauchy compression (MPa)');axes[0,0].legend()
    h=histories[cases[-1]]
    for field,label,style in [('geometry_porosity','Corner geometry','-'),('material_porosity','Full material gradients','--')]:
        axes[0,1].plot(h['step_time'],100*h[field],style,label=label)
    axes[0,1].set(xlabel='Normalized load/release stage',ylabel='Porosity (%)');axes[0,1].legend()
    axes[1,0].bar([0,1],[b['max_native_green_strain_error'],b['max_omitting_modes_green_error']])
    axes[1,0].set(xticks=[0,1],xticklabels=['With internal modes','Omit internal modes'],ylabel='Max absolute Green strain error',yscale='log')
    en=energies[cases[-1]]
    for field,label,style in [('macro_work_Nmm','Macro','-'),('micro_work_Nmm','Micro','--'),('solver_internal_Nmm','Native ELSE',':')]:
        axes[1,1].plot(en['step_time'],en[field],style,label=label)
    axes[1,1].set(xlabel='Normalized load/release stage',ylabel='Accumulated work (N mm)');axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.25)
    fig.suptitle('Experimental periodic FEM: 25 mm cell, E=10 MPa, 1% compression\nMesh and self-contact not validated')
    fig.savefig(out/'periodic-mode-verification.png',dpi=180);plt.close(fig)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(out.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'relative_differences':differences,'extra_checks':additional,'zip_bytes':archive.stat().st_size},indent=2))

if __name__=='__main__':main()
