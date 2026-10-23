"""Plot the actual refined contact surface, from saved material mesh arrays."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from packing_fem.adaptive_tet import boundary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig,axes=plt.subplots(1,2,figsize=(10,5));counts=[];old_nodes=None
    for ax,name,title in zip(axes,['contact-before.npz','contact-after.npz'],['Before surface bisection','After periodic surface bisection']):
        with np.load(args.source/name) as z:X=z['reference_points'];x=z['points'];t=z['tets']
        faces=boundary(t);faces=faces[np.all(np.isclose(X[faces,2],.6),axis=1)]
        edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
        ax.add_collection(LineCollection(x[edges][:,:,:2],linewidths=.9,color='steelblue'));ax.autoscale();ax.set_aspect('equal');ax.set(xlabel='Current x [mm]',ylabel='Current y [mm]',title=title)
        if old_nodes is None:old_nodes=len(X)
        else:
            new=np.arange(old_nodes,len(X));new=new[np.isclose(X[new,2],.6)]
            ax.scatter(x[new,0],x[new,1],color='darkred',s=30,label='New periodic nodes',zorder=3);ax.legend(fontsize=8)
        counts.append(dict(file=name,total_nodes=len(X),total_tetrahedra=len(t),shown_contact_triangles=len(faces)))
    fig.suptitle('Actual contact surface (reference z = 0.6 mm), prescribed sticking test');fig.tight_layout();fig.savefig(args.output/'surface-bisection.png',dpi=180);plt.close(fig)
    (args.output/'view.json').write_text(json.dumps(dict(source=str(args.source),scope=__doc__,meshes=counts),indent=2)+'\n');shutil.copy2(Path(__file__),args.output/'plot_surface_bisection.py')


if __name__=='__main__':main()
