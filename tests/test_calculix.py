"""Independent continuum benchmark; requires the installed FEM executable."""
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from packing_fem.cli import solver_environment
from packing_fem.results import read_dat,find

SOLVER='/workspace/environment/calculix/usr/bin/ccx'


@pytest.mark.skipif(shutil.which(SOLVER) is None,reason='CalculiX not installed')
def test_uniaxial_elastic_cube_against_analytical_solution(tmp_path):
    (tmp_path/'cube.inp').write_text('''*HEADING
Independent E/nu/pressure test: mm N MPa
*NODE
1,0,0,0
2,1,0,0
3,1,1,0
4,0,1,0
5,0,0,1
6,1,0,1
7,1,1,1
8,0,1,1
*ELEMENT,TYPE=C3D8,ELSET=BODY
1,1,2,3,4,5,6,7,8
*SOLID SECTION,ELSET=BODY,MATERIAL=MAT
*MATERIAL,NAME=MAT
*ELASTIC
70000,0.33
*NSET,NSET=ALL,GENERATE
1,8,1
*NSET,NSET=XZERO
1,4,5,8
*NSET,NSET=YZERO
1,2,5,6
*NSET,NSET=ZZERO
1,2,3,4
*BOUNDARY
XZERO,1,1,0
YZERO,2,2,0
ZZERO,3,3,0
*STEP
*STATIC
*DLOAD
1,P2,0.1
*NODE PRINT,NSET=ALL
U
*EL PRINT,ELSET=BODY
S
*END STEP
''')
    run=subprocess.run([SOLVER,'-i','cube'],cwd=tmp_path,env=solver_environment(),
                       capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stdout[-3000:]
    assert 'Job finished' in run.stdout
    _,s=list(read_dat(tmp_path/'cube.dat'))[-1]
    stress=find(s,'stresses','BODY')[:,2:8]
    np.testing.assert_allclose(stress,np.tile([0,0,-.1,0,0,0],(8,1)),atol=1e-8)
    u=find(s,'displacements','ALL')
    node7=u[u[:,0]==7][0,1:]
    np.testing.assert_allclose(node7,[.33*.1/70000,.33*.1/70000,-.1/70000],rtol=1e-6)
