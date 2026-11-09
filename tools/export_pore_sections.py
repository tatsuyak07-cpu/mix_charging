"""Periodic mid-plane cuts of actual tetrahedra; no geometric smoothing."""
import argparse,itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from packing_fem.explicit_material import geometry
from packing_fem.explicit_periodic import deformation


EDGES=list(itertools.combinations(range(4),2))


def cut_tetrahedron(vertices,normal,level,axes,tolerance=1e-10):
    distance=vertices[:,normal]-level;points=[]
    for i,j in EDGES:
        if abs(distance[i])<=tolerance:points.append(vertices[i,list(axes)])
        if abs(distance[j])<=tolerance:points.append(vertices[j,list(axes)])
        if distance[i]*distance[j]<0:
            point=vertices[i]+distance[i]/(distance[i]-distance[j])*(vertices[j]-vertices[i])
            points.append(point[list(axes)])
    if len(points)<3:return None
    points=np.asarray(points);_,keep=np.unique(np.rint(points/tolerance).astype(np.int64),axis=0,return_index=True);points=points[keep]
    if len(points)<3:return None
    center=points.mean(axis=0);order=np.argsort(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]));return points[order]


def section_polygons(state,L,normal):
    H=deformation(state['q'][-6:])
    if np.max(abs(H-np.diag(np.diag(H))))>1e-12:raise ValueError('Section renderer requires a diagonal periodic lattice')
    if np.max(abs(np.diag(H)[:2]-1))>1e-12 or H[2,2]>1+1e-12:raise ValueError('Section comparison requires fixed transverse dimensions and axial compression')
    lengths=L*np.diag(H)
    if np.any(lengths<=0):raise ValueError('Nonpositive cell length')
    x=state['x'];tet=x[state['t']];minimum=tet.min(axis=1);maximum=tet.max(axis=1);axes=tuple(i for i in range(3) if i!=normal);level=lengths[normal]/2
    low=np.ceil(-x.max(axis=0)/lengths).astype(int);high=np.floor((lengths-x.min(axis=0))/lengths).astype(int)
    polygons=[];tolerance=L*1e-10
    for shift in itertools.product(*(range(a,b+1) for a,b in zip(low,high))):
        offset=np.asarray(shift)*lengths;lo=minimum+offset;hi=maximum+offset
        mask=(lo[:,normal]<=level+tolerance)&(hi[:,normal]>=level-tolerance)
        for axis in axes:mask&=(lo[:,axis]<=lengths[axis]+tolerance)&(hi[:,axis]>=-tolerance)
        for vertices in tet[mask]+offset:
            polygon=cut_tetrahedron(vertices,normal,level,axes,tolerance)
            if polygon is not None:polygons.append(polygon)
    return polygons,lengths,axes


def render_sections(initial,loaded,L,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    fig,grid=plt.subplots(2,2,figsize=(11,10),layout='constrained')
    for col,(name,state) in enumerate([('Initial',initial),('Loaded',loaded)]):
        H=deformation(state['q'][-6:]);phi=1-geometry(state['x'],state['t'])[0].sum()/(L**3*np.linalg.det(H))
        for row,normal in enumerate([2,1]):
            polygons,lengths,axes=section_polygons(state,L,normal);ax=grid[row,col]
            collection=PolyCollection(polygons,facecolors='#5b839f',edgecolors='#243b49',linewidths=.13)
            ax.add_collection(collection)
            cell=Rectangle((0,0),lengths[axes[0]],lengths[axes[1]],transform=ax.transData,fill=False,edgecolor='#b32929',linewidth=1.,zorder=3)
            collection.set_clip_path(cell);ax.add_patch(cell)
            if lengths[axes[1]]<L-1e-10:
                ax.add_patch(Rectangle((0,lengths[axes[1]]),L,L-lengths[axes[1]],facecolor='#e5e5e5',edgecolor='#bdbdbd',hatch='///',linewidth=.4,zorder=2))
                ax.text(L/2,(L+lengths[axes[1]])/2,'Outside current cell',ha='center',va='center',fontsize=9,zorder=4)
            ax.set(xlim=(0,L),ylim=(0,L),xlabel='XYZ'[axes[0]]+' (mm)',ylabel='XYZ'[axes[1]]+' (mm)',title=f'{name}: 3D porosity {100*phi:.3f}%\n'+'XYZ'[normal]+f' = {lengths[normal]/2:.3f} mm')
            ax.set_aspect('equal');ax.set_facecolor('white')
    fig.suptitle('Periodic packing: actual FEM mid-plane sections\nFixed transverse cell; displacement scale 1x')
    fig.get_layout_engine().set(rect=(0,.055,1,.95))
    fig.text(.5,.012,'White: section voids. Gray: outside current cell. All panels use the same physical scale.\nTitles use 3D element-volume porosity; mesh / increment convergence unverified.',ha='center',fontsize=10)
    fig.savefig(out,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--initial',required=True);p.add_argument('--loaded',required=True);p.add_argument('--length',required=True,type=float);p.add_argument('--out',required=True);a=p.parse_args()
    with np.load(a.initial) as f:initial=dict(f)
    with np.load(a.loaded) as f:loaded=dict(f)
    render_sections(initial,loaded,a.length,a.out)
