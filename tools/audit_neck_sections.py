"""Compare isolated neck mid-gap T4 sections with their exact CAD circles."""
import argparse,csv,hashlib,itertools,json
from pathlib import Path
import numpy as np
from packing_fem.mesh_sections import section_area


def audit(spec,packing,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    definition=json.loads(Path(spec).read_text());data=np.load(packing)
    L=float(data['length_mm']);centers=data['fractional_centers']*L
    bonds=data['bonds'];images=data['bond_images']
    configs=[json.loads(Path(c['config']).read_text()) for c in definition['cases']]
    D=configs[0]['diameter_mm'];r=configs[0]['neck_diameter_mm']/2
    if any(c['diameter_mm']!=D or c['neck_diameter_mm']!=2*r or c['length_mm']!=L for c in configs):
        raise ValueError('Geometry differs between cases')
    halfwidth=1.1*r;reach=np.sqrt(2)*halfwidth
    shifts=L*np.array(list(itertools.product([-1,0,1],repeat=3)))
    starts=(centers[bonds[:,0]][:,None,:]+shifts[None,:,:]).reshape(-1,3)
    directions=np.repeat(centers[bonds[:,1]]+L*images-centers[bonds[:,0]],27,axis=0)
    ids=np.repeat(np.arange(len(bonds)),27);selected=[];excluded=[]
    for index,((a,b),image) in enumerate(zip(bonds,images)):
        A=centers[a];B=centers[b]+L*image;delta=B-A;length=np.linalg.norm(delta);mid=((A+B)/2)%L
        reason=None
        if length<=D:reason='Endpoint spheres overlap the mid-gap plane'
        elif np.min(np.minimum(mid,L-mid))<=reach:reason='Window reaches a periodic cell face'
        else:
            other=centers[[j for j in range(len(centers)) if j not in (a,b)]]-mid
            other-=L*np.round(other/L)
            if np.linalg.norm(other,axis=1).min()<=D/2+reach:reason='Another sphere may intersect the window'
            else:
                w=mid-starts;alpha=np.clip(np.sum(w*directions,axis=1)/np.sum(directions**2,axis=1),0,1)
                distance=np.linalg.norm(w-alpha[:,None]*directions,axis=1)
                if np.min(distance[ids!=index])<=r+reach:reason='Another neck may intersect the window'
        if reason:excluded.append(dict(neck=index,reason=reason))
        else:selected.append(dict(neck=index,center=mid,normal=delta/length,gap_mm=float(length-D)))
    if not selected:raise ValueError('No isolated interior neck sections')
    rows=[];summaries=[];sources=[]
    for level,case in enumerate(definition['cases']):
        path=Path(case['mesh'])
        with np.load(path) as mesh:points=mesh['points'];tets=mesh['tets']
        sources.append(dict(level=level,mesh=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for neck in selected:
            area,count=section_area(points,tets,neck['center'],neck['normal'],halfwidth)
            if not 0<area<=np.pi*r*r*(1+1e-7):raise ValueError('Section is inconsistent with an isolated inscribed CAD cylinder')
            rows.append(dict(level=level,tetrahedra=len(tets),neck=neck['neck'],gap_mm=neck['gap_mm'],
                exact_area_mm2=float(np.pi*r*r),mesh_area_mm2=area,relative_area_deficit=float(1-area/(np.pi*r*r)),cut_tetrahedra=count))
        errors=[row['relative_area_deficit'] for row in rows if row['level']==level]
        summaries.append(dict(level=level,sections=len(errors),minimum_relative_area_deficit=min(errors),mean_relative_area_deficit=float(np.mean(errors)),maximum_relative_area_deficit=max(errors)))
    report=dict(scope='Geometry-only diagnosis: isolated interior neck mid-gap sections; no mechanical causation or convergence claim.',
        exact_area_mm2=float(np.pi*r*r),window_halfwidth_mm=halfwidth,
        packing_sha256=hashlib.sha256(Path(packing).read_bytes()).hexdigest(),
        selected_necks=[int(n['neck']) for n in selected],excluded=excluded,sources=sources,summaries=summaries,
        mechanical_cause_established=False,full_loaded_path_convergence_verified=False)
    (out/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'sections.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summaries),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--packing',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();audit(a.spec,a.packing,a.out)
