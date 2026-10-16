# Flat periodic contact verification: FAILED physical invariance

This is a prescribed **contact-only** test of two flat periodic slab faces, each of area 1 mm². It is not the 100 mm particle packing, a material loading calculation, or a 40% to 10% compaction result. Geometry volume, gap and periodic identification are independently tested. Mu remains 0.3; every mesh phase is initialized with fresh friction history. The phase sweep is not a sliding-friction trajectory.

Changing the in-plane mesh shift must leave the flat physical interface unchanged. The current fixed sum of point-triangle and edge-edge barriers fails this test. Normal force balance and energy/periodic-work derivatives pass; they do not establish physical contact consistency. The area-spring reference columns are diagnostic comparison values, not a claimed exact solution of the implemented fixed constraint sum.

All failed runs and logs are retained. Original matrix 001 precedes component decomposition; matrix 002 adds refinement 8 and force components; the dense phase path records initial friction energy. Missing columns in older records are not inferred as passed checks. Mesh NPZ/VTU files are the actual prescribed interface geometries. Production solver/material/geometry were not changed. See docs/plane-contact-verification.md in the repository for conclusions and required redesign.
