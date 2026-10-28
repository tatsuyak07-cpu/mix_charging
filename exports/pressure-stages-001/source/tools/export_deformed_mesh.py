"""Render actual solved C3D8 boundary faces with explicitly labelled display scaling."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import meshio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from export_mesh_images import faces


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    summary=json.loads((args.run/'sintered-summary.json').read_text())
    if not all(summary.get('checks',{}).values()) or summary['status']!='equilibrium_candidate_requires_refinement':
        raise ValueError('Successful checked solution required')
    metadata=json.loads((args.run/'sintered-metadata.json').read_text())
    c=metadata['config'];L=c['length_mm'];h=c['mesh_mm']
    data=np.load(args.run/'sintered-geometry.npz')
    points=data['points'];elements=data['elements'];mask=data['mask'];n=mask.shape[0]
    solved=meshio.read(args.run/'sintered-deformed.vtu')
    u=solved.point_data['displacement_mm']
    assert np.array_equal(solved.cells_dict['hexahedron'],elements)
    assert np.allclose(solved.points,points+u,atol=1e-12,rtol=0)
    lookup=np.full((n+1,)*3,-1,dtype=np.int64)
    lookup[tuple(np.rint(points/h).astype(int).T)]=np.arange(len(points))
    args.output.mkdir(parents=True,exist_ok=False)
    E=np.array(summary['macro_strain'])
    corners=np.array([[x,y,z] for x in [0,L] for y in [0,L] for z in [0,L]])
    frame_edges=[(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(corners[i]!=corners[j])==1]
    for name,scale,cut in [('mesh_actual_1x',1,False),('mesh_deformation_20x',20,False),('mesh_cutaway_20x',20,True)]:
        visible=mask.copy()
        if cut:visible[n//2:,:,:]=False
        quads=faces(visible,h)
        indices=lookup[tuple(np.rint(quads/h).astype(int).transpose(2,0,1))]
        assert np.all(indices>=0)
        deformed=points[indices]+scale*u[indices]
        fig=plt.figure(figsize=(11,10));ax=fig.add_subplot(projection='3d')
        ax.add_collection3d(Poly3DCollection(deformed,facecolors='#86b8d3',edgecolors='#263e4a',linewidths=.12))
        # Gray dashed frame is the original cell; red frame includes only macro deformation.
        moved=corners+scale*(corners@E.T+u[0]-points[0]@E.T)
        for i,j in frame_edges:
            ax.plot(*corners[[i,j]].T,color='gray',ls='--',lw=.6)
            ax.plot(*moved[[i,j]].T,color='#b32929',lw=.8)
        lower=np.minimum(deformed.min(axis=(0,1)),moved.min(axis=0))-2
        upper=np.maximum(deformed.max(axis=(0,1)),moved.max(axis=0))+2
        ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),zlim=(lower[2],upper[2]),
               xlabel='X [mm]',ylabel='Y [mm]',zlabel='Z [mm]')
        ax.set_box_aspect(upper-lower);ax.view_init(elev=24,azim=-55)
        title=f"Solved mesh | E = {c['young_mpa']/1000:g} GPa | compression = {c['target_mpa']:g} MPa\nDisplacement display scale: {scale}x"
        if cut:title+=' | half-cell cutaway'
        ax.set_title(title)
        fig.text(.5,.02,'1 mm C3D8 mesh | dashed: initial cell; red: deformed cell\n'+
                 'Display magnification only; mesh convergence remains unverified.',ha='center',fontsize=10)
        fig.savefig(args.output/f'{name}.png',dpi=200,bbox_inches='tight');plt.close(fig)
    provenance={'run':str(args.run),'geometry_sha256':hashlib.sha256((args.run/'sintered-geometry.npz').read_bytes()).hexdigest(),
                'displacement_source':'sintered-deformed.vtu: displacement_mm','display_scales':[1,20],
                'elements':len(elements),'note':'Exact solved nodal displacements; no smoothing or remeshing. Cutaway includes section faces.'}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':main()
