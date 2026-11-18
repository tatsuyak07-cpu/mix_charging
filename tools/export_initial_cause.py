"""Export the completed zero-strain and matched-boundary diagnostic controls."""
import csv,itertools,json,shutil
from pathlib import Path
import numpy as np


def export(root,out):
    root=Path(root);out=Path(out);out.mkdir(exist_ok=False,parents=True)
    elastic=json.loads((root/'elastic-001/assessment.json').read_text());rows=[]
    for i in range(3):
        case=root/f'level-{i:03d}';zero=json.loads((case/'assessment.json').read_text())
        if not zero['original_physics_audit_passed']:raise ValueError('Original physics audit failed')
        with (case/'small-load-history.csv').open() as f:small=list(csv.DictReader(f))[-1]
        differential=(float(small['nominal_pressure_MPa'])-zero['zero_strain_macro_pressure_MPa'])/.0001
        C=elastic['original_levels'][i]['effective_C33_MPa']
        rows.append(dict(level=i,tetrahedra=elastic['original_levels'][i]['tetrahedra'],
            initial_contact_force_max_N=zero['initial_contact_force_max_N'],
            zero_strain_maximum_ep=zero['zero_strain']['maximum_ep'],
            small_load_maximum_ep=zero['small_load']['maximum_ep'],
            zero_strain_plastic_elements=zero['zero_strain']['plastic_elements'],
            shared_plastic_elements=zero['plastic_elements_shared_with_small_load'],
            zero_strain_macro_pressure_MPa=zero['zero_strain_macro_pressure_MPa'],
            elastic_C33_MPa=C,matched_control_pressure_difference_over_strain_MPa=differential,
            relative_difference_from_elastic_tangent=abs(differential/C-1)))
        for name in ['assessment.json','audit.json','history.csv','inputs.json']:
            shutil.copy2(case/name,out/f'level-{i:03d}-{name}')
    fine=json.loads((root/'level-002/assessment.json').read_text())
    stencils=json.loads((root/'initial-stencils.json').read_text())
    result=dict(rows=rows,elastic=elastic,
        initial_barrier_drives_zero_strain_local_plasticity_confirmed=bool(fine['zero_strain']['maximum_ep']>0 and fine['no_contact_zero_strain_maximum_ep']==0 and fine['plastic_elements_shared_with_small_load']==2 and fine['zero_strain']['maximum_ep_element']==fine['small_load']['maximum_ep_element']),
        same_boundary_discretization_stiffness_sensitivity_confirmed=bool(elastic['same_boundary_refinement_relative_change']>.05),
        rigid_contact_in_initial_stencils=stencils['any_rigid_contact'],
        plasticity_or_friction_not_required_for_initial_mesh_stiffness_difference=True,
        specific_locking_mechanism_proven=False,full_compaction_convergence_verified=False,
        note='Linear controls are separate diagnostics, not contact-free replacement production calculations. Scalar pressure differences compare two virgin-history solves; no material histories are subtracted or pre-relaxed loading path claimed.')
    (out/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for source,target in [('initial-stencils.json','initial-stencils.json'),('elastic-001/assessment.json','elastic.json'),('tests.log','tests.log'),('analytic-tests.log','analytic-tests.log'),('execution-verification.json','execution-verification.json')]:shutil.copy2(root/source,out/target)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    counts=[r['tetrahedra'] for r in rows]
    ax[0].plot(counts,[r['elastic_C33_MPa'] for r in rows],'o-',label='No-contact initial elastic tangent')
    ax[0].plot(counts,[r['matched_control_pressure_difference_over_strain_MPa'] for r in rows],'x--',label='(P at 0.01% - P at 0) / 0.0001')
    ax[0].set_xticks(counts,[f'{n:,}' for n in counts]);ax[0].set(xlabel='T4 elements',ylabel='Axial constrained modulus (MPa)');ax[0].legend(fontsize=8)
    ref=elastic['same_faceted_boundary_refinement'];coarse=rows[0]['elastic_C33_MPa']
    ax[1].bar(['65,407 T4','523,256 T4'],[coarse,ref['effective_C33_MPa']],color=['steelblue','darkorange'])
    ax[1].set(ylabel='Initial elastic modulus (MPa)',title='Exactly the same faceted boundary')
    ax[1].text(.5,.5,f"Stiffness decreases {100*elastic['same_boundary_refinement_relative_change']:.2f}%",ha='center',transform=ax[1].transAxes)
    for a in ax:a.grid(axis='y',alpha=.2)
    fig.savefig(out/'elastic-cause.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained');pos=np.arange(3)
    ax.bar(pos-.17,[r['zero_strain_maximum_ep'] for r in rows],.34,label='Zero macro strain')
    ax.bar(pos+.17,[r['small_load_maximum_ep'] for r in rows],.34,label='0.01% compression')
    ax.set_xticks(pos,[f'{n:,} T4' for n in counts]);ax.set(ylabel='Maximum equivalent plastic strain',title='Original material and contact laws');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'initial-plasticity.png',dpi=180);plt.close(fig)
    with np.load(root/'level-002/state-0000.npz') as f:X=f['X'];t=f['t']
    with np.load(root/'level-002/state-0001.npz') as f:plastic=np.flatnonzero(f['ep']>1e-12)
    center=np.array(fine['zero_strain']['maximum_ep_element_center_mm']);L=24.140717635130514
    points=X-center;points-=L*np.round(points/L);points+=center
    nearby=np.flatnonzero(np.linalg.norm(points[t].mean(axis=1)-center,axis=1)<.25)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    for indices,color,width in [(nearby,'lightgray',.4),(plastic,'crimson',1.8)]:
        for tet in t[indices]:
            for a,b in itertools.combinations(tet,2):ax.plot(points[[a,b],0],points[[a,b],2],color=color,lw=width)
    ax.axhline(0,color='black',ls=':',lw=1,label='Periodic seam z=0')
    ax.plot([],[],color='crimson',label='Plastic T4 at zero macro strain')
    ax.set(xlabel='Reference x (mm)',ylabel='Reference z (mm)',title='Initial geometry: local x-z projection',xlim=(center[0]-.3,center[0]+.3),ylim=(-.2,.25));ax.set_aspect('equal');ax.legend(fontsize=8)
    fig.savefig(out/'plastic-location.png',dpi=180);plt.close(fig)
    return result


if __name__=='__main__':
    export('results/rigid-initial-cause-001','exports/rigid-initial-cause-001')
