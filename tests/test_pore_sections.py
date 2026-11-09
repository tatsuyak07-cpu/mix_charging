import numpy as np
from tools.export_pore_sections import cut_tetrahedron
from packing_fem.explicit_material import hex_to_tet


def area(p):
    return .5*abs(p[:,0]@np.roll(p[:,1],-1)-p[:,1]@np.roll(p[:,0],-1))


def test_tetra_midplane_matches_analytic_triangle_area():
    X=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    assert abs(area(cut_tetrahedron(X,2,.5,(0,1)))-.125)<1e-14


def test_cube_tet_sections_preserve_complete_unit_square():
    X=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    t=hex_to_tet(X,np.arange(8)[None,:]);polygons=[cut_tetrahedron(X[element],2,.5,(0,1)) for element in t]
    assert abs(sum(area(p) for p in polygons if p is not None)-1.)<1e-14
