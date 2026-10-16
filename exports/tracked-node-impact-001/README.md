Manufactured dynamic tip impact inside a cubic pore in a periodic solid, contact_backend=tracked_guarded. Initial velocity 1000 mm/s is numerical test input, not a proposed quasi-static loading condition.

Both base and half_dt paths completed with active node-to-face contact. functional-checks.json audits dynamic functionality; it does not replace or relax the original quasi-static checks in each summary.json. Both cases remain diagnostics_only under those original checks. Edge guard remains enabled. Edge-edge response is not implemented.

Complete inputs, mesh, last accepted state, VTU and CSV histories are included. This is not the user's 10 mm sintered packing or a 0.1 MPa result.

Superseded energy audit by tracked-node-impact-002, which uses the signed trapezoidal work correction. Raw force trajectories are unchanged.
