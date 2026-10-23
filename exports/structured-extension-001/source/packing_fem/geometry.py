"""Periodic geometry and explicit tetrahedral sphere meshes (not DEM)."""
from itertools import product
import numpy as np
from scipy.spatial import cKDTree


def sphere_volume(diameter):
    return np.pi * diameter**3 / 6.0


def make_packing(length, diameter, porosity, seed, max_iterations=20000):
    """Relax random overlapping centers at fixed box size, with no loads.

    Counts are integral: exact analytic porosity differs by <= half a sphere.
    This is geometry generation, not a mechanical compression/preload step.
    """
    n = int(np.floor((1.0 - porosity) * length**3 / sphere_volume(diameter) + .5))
    if n < 1 or length <= diameter:
        raise ValueError("Require at least one sphere and box length > diameter")
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, length, (n, 3))
    for iteration in range(max_iterations):
        pairs = cKDTree(x, boxsize=length).query_pairs(diameter, output_type='ndarray')
        if len(pairs) == 0:
            return x
        delta = x[pairs[:, 1]] - x[pairs[:, 0]]
        delta -= length * np.rint(delta / length)
        distance = np.linalg.norm(delta, axis=1)
        if np.any(distance < 1e-12):
            x = (x + rng.normal(0, 1e-6 * diameter, x.shape)) % length
            continue
        overlap = diameter - distance
        if np.max(overlap) < 1e-8 * diameter:
            return x
        correction = overlap[:, None] * delta / distance[:, None]
        force = np.zeros_like(x)
        np.add.at(force, pairs[:, 0], -correction)
        np.add.at(force, pairs[:, 1], correction)
        # Divide by degree to keep simultaneous pair corrections bounded.
        degree = np.bincount(pairs.ravel(), minlength=n)
        x = (x + .8 * force / np.maximum(1, degree)[:, None]) % length
    raise RuntimeError("Nonoverlapping random packing generation did not converge")


def candidate_pairs(centers, lengths, cutoff, layers=1):
    """One representative of each unordered contact, including periodic self.

    (i,j,s) means core i versus image j+s*L.  Other contacts are translations
    of these. All 27 adjacent shifts are searched, not just minimum image.
    """
    lengths = np.asarray(lengths)
    tree = cKDTree(centers)
    result = []
    for shift in product(range(-layers,layers+1), repeat=3):
        neighborhoods = tree.query_ball_point(centers + np.array(shift) * lengths, cutoff)
        for j, neighbors in enumerate(neighborhoods):
            for i in neighbors:
                if i < j or (i == j and shift > (0, 0, 0)):
                    result.append((i, j, tuple(shift)))
    return sorted(result)


def tetra_quadrature(points, tets, exact_volume=False):
    """Signed Jacobian weights for linear or curved quadratic tetrahedra."""
    if tets.shape[1]==4:
        p=points[tets]
        return (np.linalg.det(np.stack((p[:,1]-p[:,0],p[:,2]-p[:,0],
                                      p[:,3]-p[:,0]),axis=2))/6)[:,None]
    if tets.shape[1]!=10:
        raise ValueError('Only C3D4 and C3D10 tetrahedra supported')
    if exact_volume:
        bary=np.array([[.25]*4]+[[.5 if j==i else 1/6 for j in range(4)] for i in range(4)])
        weights=np.array([-2/15]+[3/40]*4)
    else:
        a,b=.5854101966249685,.1381966011250105
        bary=np.array([[a if i==j else b for j in range(4)] for i in range(4)])
        weights=np.full(4,1/24)
    gradL=np.array([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]])
    gradients=np.zeros((len(bary),10,3))
    gradients[:,:4,:]=(4*bary[:,:,None]-1)*gradL
    # CalculiX/Abaqus C3D10 midside order.
    edges=((0,1),(1,2),(0,2),(0,3),(1,3),(2,3))
    for k,(i,j) in enumerate(edges,4):
        gradients[:,k]=4*(bary[:,i,None]*gradL[j]+bary[:,j,None]*gradL[i])
    jac=np.einsum('eni,qnj->eqij',points[tets],gradients)
    return np.linalg.det(jac)*weights[None,:]


def tet_volumes(points, tets, signed=False):
    if tets.shape[1]==10:
        v=tetra_quadrature(points,tets,exact_volume=True).sum(axis=1)
        return v if signed else np.abs(v)
    p = points[tets]
    v = np.linalg.det(np.stack((p[:, 1]-p[:, 0], p[:, 2]-p[:, 0],
                               p[:, 3]-p[:, 0]), axis=2)) / 6.0
    return v if signed else np.abs(v)


def sphere_mesh(diameter, size, order=1):
    import gmsh
    gmsh.initialize(['packing-fem', '-nopopup'])
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', 2)
        gmsh.model.add('sphere')
        gmsh.model.occ.addSphere(0, 0, 0, diameter / 2)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber('Mesh.MeshSizeMin', size)
        gmsh.option.setNumber('Mesh.MeshSizeMax', size)
        gmsh.model.mesh.generate(3)
        if order not in (1,2):
            raise ValueError('Mesh order must be 1 or 2')
        if order==2:
            gmsh.model.mesh.setOrder(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(coords).reshape(-1, 3)
        index = {int(tag): i for i, tag in enumerate(tags)}
        _, connections = gmsh.model.mesh.getElementsByType(4 if order==1 else 11)
        tets = np.array([index[int(tag)] for tag in connections]).reshape(-1, 4 if order==1 else 10)
        if order==2:
            tets[:,[8,9]]=tets[:,[9,8]]  # Gmsh -> CalculiX midside order
        negative = tet_volumes(points, tets, signed=True) < 0
        if np.any(negative):
            permutation=[1,0,2,3] if order==1 else [1,0,2,3,4,6,5,8,7,9]
            tets[negative]=tets[negative][:,permutation]
        # CalculiX tetrahedron local face labels.
        faces = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))
        seen = {}
        for e, tet in enumerate(tets):
            for f, loc in enumerate(faces, 1):
                key = tuple(sorted(tet[list(loc)]))
                if key in seen:
                    seen[key] = None
                else:
                    seen[key] = (e, f)
        surface = [v for v in seen.values() if v is not None]
        return points, tets, surface
    finally:
        gmsh.finalize()
