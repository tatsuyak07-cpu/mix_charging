# Development rules

- This project uses the verified documentation conventions of
  https://github.com/tatsuyak07-cpu/four_fluid (reference commit
  f08b61f2f1ff523f39c3a8ba0a757ec90b1f4b8a). Unrecorded chat rules are not
  assumed to have been recovered. Do not change that reference repository.
- Distinguish user specifications, independently verified equations and
  developer-defined numerical choices. Keep units explicit (mm, N, MPa,
  tonne, s). Never describe a particle/DEM approximation as solid FEM.
- Retain failed attempts and original specifications. Result directories
  must be new and exclusive. Do not overwrite or delete archived runs.
- Never return unconverged iterates as solutions, hide failed checks, or
  adjust material, friction, geometry or boundary conditions just to obtain
  convergence. Numerical relaxation/penalty changes require recorded reasons
  and independent sensitivity checks before accepting physical results.
- Verify analytical limits, periodic interactions, force/stress balance,
  Coulomb bounds, mesh and timestep/relaxation dependence. Report each outcome
  separately; one seed or one refinement level does not establish convergence.
- No external communication to other people. The user explicitly authorized
  reflecting completed and future work in this GitHub repository on 2026-10-15.
  Commit and push verified changes and downloadable results within this scope;
  do not publish to unrelated destinations. Keep installation/network operations outside calculation code.
- Use the existing isolated checkout; create no worktree unless requested.
- The user now authorizes creation of the no-cylinder compression program.
  The earlier design-only restriction no longer applies.
- Current development scope (2026-10-29): the sole arrival target is loaded
  porosity 40% to 10%, including hold, zero-load recovery and convergence
  assessment on a small packing. A pressure of 0.1 MPa, a 100 mm cell and a
  rigid inclusion are not completion gates for this scope. Preserve the
  material and existing transverse-stress-free periodic conditions unless
  the user changes them. Report loaded and recovered porosity separately.

- User update (2026-11-07): transverse periodic cell sizes must stay fixed.
  Use fixed transverse lattice vectors (Fxx=Fyy=1, all macro shear zero),
  retaining periodic microscopic fluctuations. Keep these constraints during
  axial unloading; report lateral reactions and distinguish zero axial load
  from a completely traction-free state. Preserve the older stress-free runs.

- User update (2026-11-10): implement stages1-3 of the approved floating rigid
  cylinder plan:6DOF nondeforming inclusion, periodic contact and remesh/history
  integration, then small-packing loaded matrix-region porosity40% to10%.
  The final physical specification remains cylinderD20mm,H20mm in a100mm cell.
  Developer-defined integration prototype: cylinderD4mm,H4mm in the existing
  16-grain D10mm packing, with nonoverlap geometry selected before any loading.
  Distinguish prototype dimensions and matrix-region / total-cell porosity.
  Keep the cylinder free to translate/rotate; adding a fixed support requires
  a separate explicitly labeled physical case. Preserve prior no-cylinder runs.

- Current user scope: carry out stages1-4 of the recovery plan: diagnose the
  stalled25% state, repair the coupled contact/friction solve, complete loaded
  matrix-region porosity10%, then assess increment and mesh dependence.
  Keep the physical parameters, free rigid6DOF and fixed transverse lattice.
  Changes of numerical search must retain an independent original-force audit;
  report any failed convergence study separately. Full100mm/D20mm scaling and
  a no-rigid quantitative comparison are later stages, outside this request.
