# Area-contact correction and verification

New backend: `area_guarded`. Two-pass reference nodal area, one connected nearest target patch per source point. Old `distance_barrier` results remain unchanged. Edge-edge distances and linear-step non-crossing certification remain; unresolved edge-only or disjoint multi-patch contact stops for surface refinement/multi-contact treatment. There is no physical edge-only spring response or surface remeshing.

Final contact-only matrix: area-plane-verification-006 (32 cases). Final friction arc: area-friction-arc-006. Finite sliding: area-sliding-002. Coupled manufactured equilibrium: area-equilibrium-003. Manufactured states test FEM/contact residuals, not an explicitly computed pressure-loading solution.

The development-stage transient area-pressure-layered-002 uses a 1 mm layered cell with initial porosity 20%, reaches 0.02 MPa and FAILS the loaded kinetic screen and quasi-static endpoint strain comparison. Its source snapshot is preserved. It is not a validated compaction curve or the final friction-frame implementation. The 0.1 MPa transient 001 was deliberately stopped as a partial-load cost trial and must not be reported as complete. No 100 mm / 40% to 10% packing solution is supplied.

Failed design/test/output-serialization trials are retained; failure JSON and logs explain them. CSV uses mm, N, MPa, tonne, s; porosity fractions in history, percent on figures. Mesh PNG/VTU/NPZ use actual last accepted coordinates with displacement scale 1. Use the repository tools to reproduce; core-source is a version snapshot, not a standalone installed package.
