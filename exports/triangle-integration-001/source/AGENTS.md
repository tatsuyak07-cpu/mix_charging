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
