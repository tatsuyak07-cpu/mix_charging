# Periodic pressure-compaction development

The 4 mm verification cell starts at **0.25%** porosity. It is a manufactured thin-pore continuum fixture, not the requested sphere packing. Consult `comparison.csv` and each original `summary.json`: failed checks are retained and are not accepted physical solutions.

`barrier-packing40-*` are **25 mm** RVE setup/cost/initialization trials only (actual initial porosity 39.8592%). They do not reach 0.1 MPa or 10% porosity. The requested 100 mm packing has not been solved.

CSV units: mm, N, MPa, tonne, s. Porosity in history is a fraction; comparison uses percent. PNG meshes use actual coordinates and displacement scale 1. VTU and NPZ contain the actual last accepted mesh, including incomplete and deliberately interrupted trials. KeyboardInterrupt trials were stopped by the developer to replace expensive verified-equivalent search/derivative implementations; the isotropic mesh-refinement trial was stopped after its tiny stable timestep exposed the cost.

The code evolved during development: older records lack every-step audit, contact-energy variation and neighbor caching fields. Missing checks must not be interpreted as passed. Existing histories are preserved, not rewritten with newer check outcomes. All models use the stored input configuration.
