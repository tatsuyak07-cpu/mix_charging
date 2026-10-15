"""Recover only fully written step endpoints without editing interrupted runs."""
import argparse,hashlib,json,re,shutil,struct
from pathlib import Path
import numpy as np
from packing_fem.incompatible import _ip,read_restart_groups,analyze
from packing_fem.results import read_dat,find


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    with np.load(a.source/'sintered-geometry.npz') as g:ne=len(g['elements'])
    accepted=[];excluded=[]
    for t,s in read_dat(a.source/'plastic.dat'):
        try:
            if abs(t-round(t))>1e-7:raise ValueError('Non-endpoint frame')
            for prefix in ('strains','stresses','equivalent plastic strain'):_ip(s,prefix,ne)
            find(s,'total internal energy','SOLID');find(s,'total volume','SOLID')
            find(s,'displacements','PHYSICAL');find(s,'displacements','MACRO')
        except ValueError as exc:excluded.append({'raw_time':t,'reason':str(exc)});break
        accepted.append(t)
    if not accepted:raise ValueError('No complete endpoints')
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ['config.json','estimate.json','mode-metadata.json','sintered-geometry.npz','plastic.inp','plastic.sta','mode.log','provenance.json','ramp-stop.json','analysis-error.json']:
        if (a.source/name).exists():shutil.copyfile(a.source/name,a.output/name)
    # Preserve the exact DAT byte prefix, not a lossy reserialization.
    with (a.source/'plastic.dat').open('rb') as src,(a.output/'plastic.dat').open('xb') as dst:
        for line in src:
            match=re.search(rb'\btime\s+([\d.EeDd+\-]+)\s*$',line)
            if match and float(match[1].replace(b'D',b'E'))>accepted[-1]+1e-7:break
            dst.write(line)
    # Native sequential records: keep complete groups corresponding to the prefix.
    groups=iter(read_restart_groups(a.source/'plastic.rout'))
    with (a.output/'plastic.rout').open('xb') as f:
        for _ in accepted:
            for record in next(groups):
                marker=struct.pack('<I',len(record));f.write(marker);f.write(record);f.write(marker)
    record={'source':str(a.source),'complete_raw_times':accepted,'excluded_tail':excluded,
            'original_DAT_sha256':hashlib.sha256((a.source/'plastic.dat').read_bytes()).hexdigest(),
            'procedure':'Byte-preserved complete DAT prefix and corresponding native restart records; original interrupted files retained. No solution reconstruction or imputation.'}
    (a.output/'recovery.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(analyze(a.output,False,-15)),flush=True)

if __name__=='__main__':main()
