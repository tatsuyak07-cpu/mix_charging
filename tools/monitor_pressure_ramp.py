"""Stop an exploratory ramp at a volume audit limit or observed porosity target.

Read-only inspection until the limit is exceeded. Only a native ccx process
whose parent runner explicitly targets the requested result may be stopped.
"""
import argparse,json,os,signal,time
from pathlib import Path
import numpy as np
from packing_fem.incompatible import read_restart_groups,restart_state,mode_gradients,_ip
from packing_fem.results import read_dat,find
from packing_fem.continuum import deformation_gradients
from packing_fem.sintered import hex_quadrature


def inspect(directory):
    m=json.loads((directory/'mode-metadata.json').read_text());c=m['config']
    with np.load(directory/'sintered-geometry.npz') as g:p,e=g['points'],g['elements']
    groups=iter(read_restart_groups(directory/'plastic.rout'));last=None
    for t,s in read_dat(directory/'plastic.dat'):
        if abs(t-round(t))>1e-7:continue
        du=find(s,'displacements','PHYSICAL');du=du[np.argsort(du[:,0])][:,1:4]
        dm=find(s,'displacements','MACRO');dm=dm[np.argsort(dm[:,0])][:,1:4]
        # Require all native outputs, including buffered tail, before stopping.
        try:
            for prefix in ('strains','stresses','equivalent plastic strain'):_ip(s,prefix,len(e))
            find(s,'total internal energy','SOLID');find(s,'total volume','SOLID')
        except ValueError:break # Keep the preceding fully flushed endpoint.
        u,mu,um=restart_state(next(groups),p,e,du,dm);last=(t,u,mu,um)
    if last is None:return None
    t,u,mu,um=last;q=mu[:,0]/m['macro_scale']/c['length_mm']
    F=np.eye(3)+np.array([[q[0],q[3],q[4]],[q[3],q[1],q[5]],[q[4],q[5],q[2]]])
    local=deformation_gradients(p,p+u,e)+np.einsum('eai,eqaj->eqij',um,mode_gradients(p,e))
    volume=np.linalg.det(F)*c['length_mm']**3
    phi_g=1-hex_quadrature(p+u,e).sum()/volume
    phi_m=1-(np.linalg.det(local)*hex_quadrature(p,e)).sum()/volume
    return {'completed_step':t,'prescribed_nominal_MPa':c['target_mpa']*t/m['compression_steps'] if c.get('control')=='force' else None,
            'axial_compression_strain':float(-q[2]),'target_final_porosity':c.get('target_final_porosity'),
            'geometry_porosity':float(phi_g),'material_porosity':float(phi_m),'absolute_difference':float(abs(phi_g-phi_m))}


def native_pids(directory):
    result=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            status=(proc/'status').read_text()
            parent=int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
            parent_args=(Path('/proc')/str(parent)/'cmdline').read_bytes().split(b'\0')
            if not any(arg.endswith((b'tools/run_pressure_ramp.py',b'tools/run_porosity_target.py')) for arg in parent_args):continue
            if b'--output' not in parent_args:continue
            output=os.fsdecode(parent_args[parent_args.index(b'--output')+1])
            if Path(output).resolve()!=directory:continue
            args=(proc/'cmdline').read_bytes().split(b'\0')
            if len(args)>2 and Path(os.fsdecode(args[0])).name=='ccx' and args[1:3]==[b'-i',b'plastic']:
                result.append(int(proc.name))
        except (OSError,RuntimeError):continue
    return result


def stop_reason(measurement,limit):
    if min(measurement['geometry_porosity'],measurement['material_porosity'])<0 or max(measurement['geometry_porosity'],measurement['material_porosity'])>1:
        return 'Nonphysical porosity; target not validated'
    if measurement['absolute_difference']>limit:
        return 'Existing volume_definition_difference audit limit exceeded; target not validated'
    target=measurement['target_final_porosity']
    if target is not None and max(measurement['geometry_porosity'],measurement['material_porosity'])<=target:
        return 'Observed target porosity; independent audit still required'
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directories',nargs='+',type=Path)
    p.add_argument('--limit',type=float,default=1e-4);a=p.parse_args()
    pending=[d.resolve() for d in a.directories]
    while pending:
        for d in pending.copy():
            pids=native_pids(d)
            if not pids:pending.remove(d);continue
            try:measurement=inspect(d)
            except (OSError,ValueError,StopIteration):continue # Writer may be between records.
            if not measurement:continue
            reason=stop_reason(measurement,a.limit)
            if reason:
                record={'reason':reason,
                        'limit':a.limit,'measurement':measurement,'action':'SIGTERM only this result directory native solver','pids':pids}
                with (d/'ramp-stop.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
                for pid in pids:
                    try:os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                print(json.dumps({'directory':str(d),**record}),flush=True);pending.remove(d)
        if pending:time.sleep(5)

if __name__=='__main__':main()
