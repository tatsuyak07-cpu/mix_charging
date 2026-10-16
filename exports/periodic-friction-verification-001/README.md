Periodic independent-probe and prescribed known-patch friction verification only. Not integrated with the compression solver. No automatic contact patch tracking, self-contact neighbour exclusion or edge-edge collision detection. No quasi-static compaction solution.

periodic-path.csv: 246 normal-probe samples across x/y/z cuts, two volume-mesh spacings. Mesh files contain reference points, tetrahedra and physical surface faces excluding paired cell caps.
friction-seam.csv: 201/401 path samples on a plane, known contact patch retained across a triangle change.
friction-rotation.csv: prescribed normal rotation and tangential increment, not coupled dynamics.
summary.json: analytic, Coulomb and discrete-work checks. Work includes algorithmic loss separately from friction dissipation.
pytest-final.log: 35 tests passed (8 new + 27 preceding).

Report: https://github.com/tatsuyak07-cpu/mix_charging/blob/main/docs/periodic-friction-verification.md
Reproduce: tools/verify_periodic_contact_probe.py and tools/plot_periodic_contact_probe.py.
