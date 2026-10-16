Opt-in tracked_edges backend. Interior edge-pair forces are enabled; endpoint, parallel and node-face handoffs remain unsupported.

base/, half_dt/, edge_penalty2/: original slit input, same material, geometry and friction; only numerical CFL/edge penalty varies. All slit paths STOP on an edge-node feature handoff. Only 1 base accepted step has active edge contact (2 for half_dt), so this is not mesh/time/penalty convergence. verification.json marks feature_handoff_validated=false and quasistatic_compaction_validated=false.
node_regression/: separate completed dynamic tip-impact test; does not meet the original quasi-static conditions. Full histories, initial meshes, last accepted states and VTU files are retained.

pytest-full.log: 93 software tests passed. Graph and local mesh image show accepted states only, displacement scale 1. This is not a 0.1 MPa packing result.
Report: https://github.com/tatsuyak07-cpu/mix_charging/blob/main/docs/interior-edge-contact.md
