"""Inspect missing history at newly active quadrature nodes; never commit it."""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem.area_contact import AreaContact
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.contact_audit import plain


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads((a.source/'config.json').read_text());capture=json.loads((a.source/'contact-audit.json').read_text())
    files=sorted(a.source.glob('surface-remesh-attempt-*.npz'))
    with np.load(a.source/'last-state.npz') as z:old={k:z[k] for k in z.files}
    with np.load(files[-1]) as z:proposal={k:z[k] for k in z.files}
    X=proposal['reference_points'];x=proposal['points'];t=proposal['tets'];H=deformation(old['generalized_position'][-6:])*c['length_mm']
    contact=AreaContact(X,t,Periodic(X,c['length_mm']),c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],thickness=c['barrier_thickness_mm'],multiple_patches=c['multiple_patches'],edge_stationarity_filter=c['edge_stationarity_filter'],closest_tie_rule=c['closest_tie_rule'],edge_quadrature_relative_tolerance=c.get('edge_quadrature_relative_tolerance',0.))
    _,_,fresh,report=contact.evaluate(x,x*0,H,H*0,0.)
    previous={tuple(item['key']):item['state'] for item in capture['committed_history']}
    rows=[]
    for key,state in fresh.items():
        node=key[1]
        if node<len(old['reference_points']):continue
        supports=proposal['supports'][node]
        rows.append(dict(new_node=node,supports=supports.tolist(),weights=proposal['weights'][node].tolist(),
                         endpoint_history_counts=[sum(k[1]==i for k in previous) for i in supports],
                         target_face=contact.faces[state['face']].tolist()))
    history=np.loadtxt(a.source/'history.csv',delimiter=',',skiprows=1)
    result=plain(dict(scope=__doc__,new_active_nodes=rows,old_normal_energy_Nmm=history[-1,10],proposal_normal_energy_Nmm=report['normal_energy'],
                      normal_quadrature_relative_change=abs(report['normal_energy']-history[-1,10])/max(abs(history[-1,10]),1e-30),
                      fresh_friction_history_not_transferred=True,time_advanced=False,proposal_accepted=False,
                      interpretation='Fresh histories are used only to identify geometry and normal quadrature; their zero slip is NOT adopted as transferred material history.'))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'diagnose_new_history.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))


if __name__=='__main__':main()
