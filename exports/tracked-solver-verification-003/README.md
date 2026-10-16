Sequential integration verification for the opt-in tracked_guarded contact backend.

verification.json records each gate. Tracking, static active-contact internal reconnection and noncontact regression passed. Edge-edge RESPONSE and general physical compaction did NOT pass. The original slit runs stop on detected swept edge intersection, before the first node penetration. Stopped histories and last accepted states are diagnostics only.

tracking.csv and active-contact-remesh.npz: manufactured contact and remesh transfer, not equilibrium/dynamics.
solid/ and void/: completed noncontact regression; final stress and porosity exactly match archived legacy results (comparison.json).
impact/ and impact_half_dt/: rejected slit paths, NOT completed solutions.
pytest-full.log: 89 software tests passed. Signed trapezoidal contact-work correction is used; the backward-Euler audit versions 001/002 are retained separately.

See ../tracked-node-impact-002/ for the separate completed dynamic node-to-face force-feedback test. No quasi-static compaction validation is claimed.
Report: https://github.com/tatsuyak07-cpu/mix_charging/blob/main/docs/tracked-solver-integration.md
