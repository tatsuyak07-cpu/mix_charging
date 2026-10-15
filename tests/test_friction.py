"""Coulomb sliding benchmark independent of spherical packing geometry."""
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from packing_fem.cli import solver_environment
from packing_fem.results import read_dat

SOLVER='/workspace/environment/calculix/usr/bin/ccx'


@pytest.mark.skipif(shutil.which(SOLVER) is None,reason='CalculiX not installed')
def test_sliding_planes_reach_coulomb_limit(tmp_path):
    lines=['*HEADING','Independent plane sliding: mu=.3','*NODE']
    vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                       [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    for shift in (0,1):
        for i,xyz in enumerate(vertices+[0,0,shift],shift*8+1):
            lines.append(f'{i},'+','.join(map(str,xyz)))
    lines += ['*ELEMENT,TYPE=C3D8,ELSET=BOTTOM','1,1,2,3,4,5,6,7,8',
              '*ELEMENT,TYPE=C3D8,ELSET=TOP','2,9,10,11,12,13,14,15,16',
              '*SOLID SECTION,ELSET=BOTTOM,MATERIAL=MAT','*SOLID SECTION,ELSET=TOP,MATERIAL=MAT',
              '*MATERIAL,NAME=MAT','*ELASTIC','70000,.33',
              '*NSET,NSET=BN,GENERATE','1,8,1','*NSET,NSET=TN,GENERATE','9,16,1',
              '*SURFACE,NAME=MASTER','1,S2','*SURFACE,NAME=SLAVE','2,S1',
              '*SURFACE INTERACTION,NAME=FRIC','*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE=LINEAR',
              '140000','*FRICTION','.3,14000',
              '*CONTACT PAIR,INTERACTION=FRIC,TYPE=SURFACE TO SURFACE','SLAVE,MASTER',
              '*BOUNDARY','BN,1,3,0','TN,1,2,0',
              '*STEP,NLGEOM,INC=100','*STATIC','.1,1,1e-8,.1',
              '*DLOAD','2,P2,.1','*CONTACT PRINT','CDIS,CSTR','*END STEP',
              '*STEP,NLGEOM,INC=100','*STATIC','.1,1,1e-8,.1',
              '*BOUNDARY','TN,1,1,.01','*CONTACT PRINT','CDIS,CSTR','*END STEP']
    (tmp_path/'slide.inp').write_text('\n'.join(lines)+'\n')
    run=subprocess.run([SOLVER,'-i','slide'],cwd=tmp_path,env=solver_environment(),
                       capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stdout[-3000:]
    _,s=list(read_dat(tmp_path/'slide.dat'))[-1]
    stress=next(v for k,v in s.items() if k.startswith('contact stress'))[:,-3:]
    compressed=stress[:,0]>1e-6
    assert compressed.any()
    ratio=np.linalg.norm(stress[compressed,1:],axis=1)/stress[compressed,0]
    np.testing.assert_allclose(ratio,.3,rtol=1e-4,atol=1e-5)
