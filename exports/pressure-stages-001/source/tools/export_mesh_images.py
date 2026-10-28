"""Export exact, undeformed voxel calculation mesh; no smoothing or remeshing."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree


def faces(mask, h):
    result=[]
    for axis in range(3):
        other=[a for a in range(3) if a!=axis]
        for sign in (-1,1):
            neighbor=np.zeros_like(mask)
            dst=[slice(None)]*3; src=dst.copy()
            dst[axis]=slice(None,-1) if sign==1 else slice(1,None)
            src[axis]=slice(1,None) if sign==1 else slice(None,-1)
            neighbor[tuple(dst)]=mask[tuple(src)]
            base=np.argwhere(mask & ~neighbor).astype(float)
            base[:,axis]+=sign==1
            offsets=np.zeros((4,3))
            offsets[:,other]=[[0,0],[1,0],[1,1],[0,1]]
            result.append((base[:,None,:]+offsets)*h)
    return np.concatenate(result)


def render(polygons, path, title, bounds, width):
    fig=plt.figure(figsize=(10,9))
    ax=fig.add_subplot(projection='3d')
    ax.add_collection3d(Poly3DCollection(polygons,facecolors='#86b8d3',
        edgecolors='#263e4a',linewidths=width,alpha=1,rasterized=True))
    for dim,setlim in enumerate((ax.set_xlim,ax.set_ylim,ax.set_zlim)):
        setlim(*bounds[dim])
    ax.set_box_aspect([hi-lo for lo,hi in bounds])
    ax.view_init(elev=24,azim=-55)
    ax.set(xlabel='X [mm]',ylabel='Y [mm]',zlabel='Z [mm]',title=title)
    fig.savefig(path,dpi=220,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    data=np.load(a.source)
    mask=data['mask']; points=data['points'];elements=data['elements'];centers=data['centers']
    h=float(np.min(np.diff(np.unique(points[:,0]))))
    assert int(mask.sum())==len(elements)
    assert len(points)==len(np.unique(elements))
    L=mask.shape[0]*h
    a.output.mkdir(parents=True,exist_ok=False)
    polygons=faces(mask,h)
    render(polygons,a.output/'mesh_overview.png',
        f'Undeformed calculation mesh | {len(elements):,} C3D8 elements\n1 mm voxel edges | 100 mm periodic cell',
        [(0,L)]*3,.12)
    fig,ax=plt.subplots(figsize=(10,9))
    k=mask.shape[2]//2
    xy=np.arange(mask.shape[0]+1)*h
    ax.pcolormesh(xy,xy,mask[:,:,k].T,cmap=matplotlib.colors.ListedColormap(['white','#86b8d3']),
        vmin=0,vmax=1,edgecolors='#b7c1c7',linewidth=.18)
    ax.set(aspect='equal',xlabel='X [mm]',ylabel='Y [mm]',
        title=f'Calculation mesh section | Z = {(k+.5)*h:g} mm\nBlue: solid elements; white: pores | grid spacing {h:g} mm')
    fig.savefig(a.output/'mesh_section.png',dpi=220,bbox_inches='tight');plt.close(fig)
    _,neighbors=cKDTree(centers,boxsize=L).query(centers,k=7)
    candidates=[]
    for i,row in enumerate(neighbors):
        for j in row[1:]:
            if i>=j:continue
            delta=centers[j]-centers[i]
            dist=np.linalg.norm(delta)
            mid=(centers[i]+centers[j])/2
            if 11<dist<13 and np.all(mid>20) and np.all(mid<L-20):
                candidates.append((abs(dist-12),i,int(j),mid,dist))
    _,i,j,mid,dist=min(candidates,key=lambda t:t[0])
    lower=np.floor((mid-5)/h).astype(int)
    upper=np.ceil((mid+5)/h).astype(int)
    bounds=np.column_stack((lower*h,upper*h))
    local=mask[tuple(slice(lo,hi) for lo,hi in zip(lower,upper))]
    local_faces=faces(local,h)+lower*h
    render(local_faces,a.output/'mesh_neck_zoom.png',
        'Actual mesh near a sintered neck (cutout; cut faces included)\nNominal neck diameter 2 mm | element edge 1 mm',bounds,.45)
    metadata={'source':str(a.source.resolve()),'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
        'state':'undeformed','element_type':'C3D8','elements':len(elements),'physical_nodes':len(points),
        'voxel_mm':h,'exposed_quad_faces':len(polygons),'section_voxel_center_z_mm':(k+.5)*h,
        'neck_particle_indices':[i,j],'neck_particle_centers_mm':centers[[i,j]].tolist(),
        'neck_center_distance_mm':dist,'zoom_bounds_mm':bounds.tolist(),
        'note':'Exact saved voxel mesh boundary faces; no smoothing. Mesh convergence is unverified.'}
    (a.output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
