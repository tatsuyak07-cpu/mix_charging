Sequential integration verification for the opt-in tracked_guarded contact backend.

verification.json records each gate. Tracking, static active-contact internal reconnection and noncontact regression passed. Edge-edge RESPONSE and general physical compaction did NOT pass. The original slit runs stop on detected swept edge intersection, before the first node penetration. Stopped histories and last accepted states are diagnostics only.

tracking.csv and active-contact-remesh.npz: manufactured contact and remesh transfer, not equilibrium/dynamics.
solid/ and void/: completed noncontact regression; final stress and porosity exactly match archived legacy results (comparison.json).
impact/ and impact_half_dt/: rejected slit paths, NOT completed solutions.
pytest-full.log: 88 software tests passed. Retained first plots preserve an earlier less readable mesh view.

See ../tracked-node-impact-001/ for the separate completed dynamic node-to-face force-feedback test. No quasi-static compaction validation is claimed.
Report: https://github.com/tatsuyak07-cpu/mix_charging/blob/main/docs/tracked-solver-integration.md

Superseded for the coupled energy ledger by version 003: backward-Euler algorithmic loss must not be added directly to Velocity Verlet work. Raw data retained.
