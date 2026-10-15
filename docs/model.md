# Model specification and limitations

## User-specified quantities

No cylinder. Cubic cell side 100 mm; solid spheres of diameter 5 mm; initial
void fraction 0.60; unbonded Coulomb contact with coefficient 0.3; isotropic
elastic solid with E=70000 MPa and nu=0.33; target compressive Cauchy stress
0.1 MPa; periodic boundaries; no pore pressure; quasistatic final state.

## Discrete geometry

Integer N=round((1-phi)*L^3/(pi*d^3/6)). Seeded random centers are relaxed to
remove initial overlap using periodic distance at fixed cell volume. This is
geometry preparation, not a mechanical preload. No gravity, bond, initial
confining pressure or cohesion is added. Statistical isotropy is an ensemble
property, not a guarantee for any finite seed. Seed variability and contact
fabric require study before interpreting a representative bulk response.

The same tetrahedral sphere template is copied with independent random proper
rotations. Gmsh quadratic surface nodes are projected onto the sphere CAD;
curved C3D10 and linear C3D4 are supported. Gmsh/CalculiX midside ordering is
converted explicitly. Signed curved volumes use a degree-three five-point
tetrahedral quadrature. Mean Cauchy stresses use the solver's integration
points and corresponding deformed Jacobian weights.

## Periodic contact and energy

A canonical interaction (i,j,s) connects sphere i to image j+s*L. The paired
nodes obey u_image - u_core - s_axis*q_axis = 0. q_axis is the change in cell
length. Increments include particle translations, rotations and deformations;
there is no external particle anchoring. These are orthogonal-cell periodic
constraints, not fixed side walls or fixed transverse strain.

An image shares all core displacements apart from the cell shift. If a sphere
has m copies including its core, each copy has E/m and numerical density/m.
Their summed material strain energy equals the single physical grain energy;
using E in every copy would incorrectly stiffen the packing. Poisson ratio
is unchanged. All canonical contact pairs are counted once. Summing scaled
image stress integrals recovers the physical grain stress integral. Image
volumes are never counted in porosity.

Candidate pairs are selected initially within `candidate_mm`. Large grain
rearrangement can create contacts outside this list. Every saved increment is
audited geometrically; an omitted possible contact invalidates the run. This
does not prove no missed contact occurred BETWEEN output frames. Increasing
candidate radius and output frequency are required sensitivity checks. More
than one periodic layer is audited when a cell dimension shrinks below d.

## Loading and numerical relaxation (developer choices)

Initial void fraction 60% need not contain a load-bearing contact network.
A singular initial static solve is avoided with implicit dynamic relaxation:
finite numerical grain masses, mass-proportional Rayleigh damping and a
smooth axial force ramp followed by a hold. Mass, damping and the time scale
are numerical choices; they are not measured density, physical dynamics or
material changes. Three translating controller tetrahedra provide numerical
cell inertia and no elastic strain energy. Controller kinetic energy is
included in the quasistatic acceptance test.

Axial generalized force is initially -0.1*L0^2 N. It is corrected using the
current lateral area in later cycles to target Cauchy rather than nominal
stress. Each rerun includes the complete preceding load history from the same
initial state, in a new job, so frictional history is retained. x/y box DOFs
have zero applied generalized force and are free. Off-diagonal box strains
are fixed; residual shear stresses must be small. A finite anisotropic seed
which fails that test requires a six-component cell formulation, not a claim
of uniaxial stress. No transverse confinement is silently substituted.

Contact is CalculiX face-to-face large-sliding penalty contact, nonbonded,
with linear normal overclosure and regularized Coulomb stick/slip. Penalty
and stick slopes are numerical parameters. Penetration, penalty energy and
Coulomb bounds are checked. In installed CalculiX 2.20 output, face-contact
normal displacement is negative for compression (p=-k*normal); parsing uses
the observed convention, not the older 2.11 manual's opposite wording.

## Porosity and acceptance

phi(t)=1-sum(physical deformed grain volumes)/product(cell lengths).
The deformed FEM volume is used, not a permanently incompressible analytic
sphere volume. Slight overlap introduced by penalty contact makes this a
sum-volume approximation instead of an exact Boolean union; penetration
must be controlled. Ideal-sphere initial porosity and FEM initial porosity
are distinct. Never substitute the former into the deformed-mesh history.

The checks require target axial stress and small lateral/shear stress, a
positive cell, positive element volumes and Jacobians at the solver integration
points, available/current outputs, a completed
solver run and final time, contact coverage, the Coulomb bound, penetration
below 0.005 mm, penalty energy below 10% of bulk strain energy, sphere mesh
volume error below 1%, low TOTAL kinetic/elastic energy over a late window,
and stable cell dimensions. These are initial engineering tolerances, not
proof of mesh independence or material calibration. Failed runs retain their
last accepted increment solely as diagnostics. They are not returned as
physical solutions.

## Current limits

Isotropic elastic grains with geometric nonlinearity, not plasticity, damage,
fracture, cohesion, fluid flow or the later cylinder comparison. Linear elastic
material use requires small grain strains even when rearrangement is large.
Contact patches need their own mesh study; a small volume error alone is not
sufficient contact accuracy. RVE size, seeds, time step, numerical mass,
damping, penalty stiffness and candidate radius all require refinement.
Unconfined frictional packing may not reach a stable equilibrium under the
specified stress. The software reports that outcome without adding walls.

The full-size model has millions of elements/DOFs even on a coarse mesh.
`max_dofs` stops unintended large solves; `--allow-large` explicitly enables
input generation. CalculiX's direct SPOOLES solver is the current validated
backend. A scalable solver and larger compute resources will be necessary
for reliable full-size convergence studies; no full-size readiness claim is
made from the one-particle periodic smoke calculation.

A scale estimate illustrates the contact-resolution issue: if one contact
carries F=0.1 MPa*(5 mm)^2=2.5 N, equal elastic spheres have reduced radius
R*=1.25 mm and E*=70000/[2*(1-.33^2)] MPa. Hertz theory gives contact radius
a=(3 F R*/(4 E*))^(1/3), about 0.039 mm. This is an illustrative force estimate,
not an assumed packing contact force or a Hertz replacement for FEM. A 1 mm
mesh does not resolve such a patch. Local contact refinement and an independent
Hertz benchmark are therefore outstanding even if all runtime checks pass.
