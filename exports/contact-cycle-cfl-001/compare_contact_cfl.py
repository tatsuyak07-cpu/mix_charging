"""Predeclared full contact/plastic cycle timestep comparison, same mesh and physics."""
import argparse,json,hashlib,shutil,zipfile
from pathlib import Path
import numpy as np

# Absolute strain/porosity and relative field tolerances declared before fine run.
TOLERANCES=dict(residual_strain_absolute=1e-4,porosity_absolute=1e-7,
                path_strain_relative_to_peak=1e-3,peak_peeq_relative=1e-3)

def compare(coarse,fine):
    runs=[p/'run' for p in (coarse,fine)]
    configs=[json.loads((p/'config.json').read_text()) for p in runs]
    differing=[k for k in configs[0].keys()|configs[1].keys() if configs[0].get(k)!=configs[1].get(k)]
    histories=[np.loadtxt(p/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1) for p in runs]
    summaries=[json.loads((p/'summary.json').read_text()) for p in runs]
    endpoints=[json.loads((p/'endpoint-audit.json').read_text()) for p in (coarse,fine)]
    a,b=histories;times=np.unique(np.r_[a[:,0],b[:,0]])
    curves=[np.column_stack([np.interp(times,h[:,0],h[:,i]) for i in (1,2,3)]) for h in histories]
    diff=abs(curves[1]-curves[0]);strain_peak=max(abs(a[:,1]).max(),1e-12)
    metrics=dict(residual_strain_absolute=abs(summaries[1]['residual_compression_strain']-summaries[0]['residual_compression_strain']),porosity_absolute=float(diff[:,1].max()),path_strain_relative_to_peak=float(diff[:,0].max()/strain_peak),peak_peeq_relative=abs(endpoints[1]['peak_peeq']-endpoints[0]['peak_peeq'])/max(endpoints[0]['peak_peeq'],1e-12))
    checks={k:metrics[k]<v for k,v in TOLERANCES.items()}
    checks.update(only_cfl_changed=differing==['cfl'],cfl_halved=configs[1]['cfl']==configs[0]['cfl']/2,both_native_checks=all(all(s['checks'].values()) for s in summaries),both_endpoint_checks=all(all(s['checks'].values()) for s in endpoints),same_end_time=abs(a[-1,0]-b[-1,0])<1e-14)
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(tolerances=TOLERANCES,metrics=metrics,checks=checks,all_checks_passed=all(checks.values()),config_differences=differing,coarse_steps=len(a)-1,fine_steps=len(b)-1,residual_strain=[s['residual_compression_strain'] for s in summaries],final_porosity=[s['final_porosity'] for s in summaries],scope='One CFL halving on the 4 mm, 0.25% pore coupon; not mesh convergence or 40% to 10% particle compaction.')
    return result,times,curves,diff

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('coarse','fine','fine-audit','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--coarse-audit',type=Path,default=Path('exports/contact-reaction-audit-001/independent-audit'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    r,t,curves,diff=compare(a.coarse,a.fine)
    audit=json.loads((a.fine_audit/'summary.json').read_text());r['checks']['fine_independent_force_audit']=audit['all_checks_passed'];r['checks']['coarse_independent_force_audit']=json.loads((a.coarse_audit/'summary.json').read_text())['all_checks_passed'];r['all_checks_passed']=all(r['checks'].values())
    for source,name in ((a.coarse,'coarse'),(a.fine,'fine'),(a.fine_audit,'fine-independent-audit'),(a.coarse_audit,'coarse-independent-audit')):shutil.copytree(source,a.output/name)
    np.savetxt(a.output/'path-differences.csv',np.column_stack((t,diff)),delimiter=',',header='time_s,strain_absolute,porosity_absolute,pressure_absolute_MPa',comments='')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for c,label in zip(curves,('CFL 0.2','CFL 0.1')):
        axs[0,0].plot(t*1000,c[:,0]*100,label=label);axs[0,1].plot(t*1000,c[:,1]*100,label=label)
    axs[0,0].set(ylabel='Axial compression (%)');axs[0,1].set(ylabel='Porosity (%)')
    axs[1,0].plot(t*1000,diff[:,0]);axs[1,0].set(ylabel='Absolute strain difference')
    axs[1,1].plot(t*1000,diff[:,1]);axs[1,1].set(ylabel='Absolute porosity difference')
    for ax in axs.flat:ax.set_xlabel('Time (ms)');ax.grid(alpha=.3)
    axs[0,0].legend();fig.suptitle('Contact/plastic loading and unloading: timestep sensitivity\n4 mm verification coupon; not the particle packing');fig.tight_layout();fig.savefig(a.output/'cfl-comparison.png',dpi=160);plt.close(fig)
    (a.output/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
    shutil.copy2(__file__,a.output/'compare_contact_cfl.py')
    (a.output/'README.md').write_text('Full-cycle CFL comparison. Units mm, N, MPa, s. See summary.json for all checks and tolerances. This is a 4 mm verification coupon, initial porosity 0.25%, not the requested particle packing. One halving does not establish mesh or general contact convergence. Source snapshots require the repository environment; captured history is not a restart checkpoint.\n')
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*') if f.is_file()}
    (a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    archive=a.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for f in a.output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('Archive CRC failure')
    print(json.dumps(r,indent=2));raise SystemExit(0 if r['all_checks_passed'] else 2)
if __name__=='__main__':main()
