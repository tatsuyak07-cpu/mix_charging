"""Publish an exclusive, reproducible comparison of two stopped RVE paths."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    current = args.input / 'corrected-run' / 'calculation'
    baseline = args.baseline / 'calculation'
    summaries = [json.loads((p / 'summary.json').read_text()) for p in (baseline, current)]
    configs = [json.loads((p / 'config.json').read_text()) for p in (baseline, current)]
    result = dict(
        scope='Rapid-load diagnostic replay; stopped states are not equilibrated solutions',
        identical_configuration=configs[0] == configs[1],
        baseline=summaries[0], corrected=summaries[1],
        mesh_convergence_verified=False,
        requested_40_to_10_percent_compaction_verified=False,
    )
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copytree(args.input, out / 'validation')
    shutil.copy2(baseline / 'summary.json', out / 'baseline-summary.json')
    shutil.copy2(baseline / 'history.csv', out / 'baseline-history.csv')
    shutil.copy2(Path(__file__), out / 'export_rve_replay.py')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from packing_fem.adaptive_tet import boundary

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for p, label in ((baseline, 'Previous transfer'), (current, 'Exact periodic transfer')):
        h = np.loadtxt(p / 'history.csv', delimiter=',', skiprows=1)
        axes[0].plot(h[:, 24], 100 * h[:, 2], label=label)
        axes[1].plot(h[:, 0] * 1000, h[:, 15], label=label)
    axes[0].set(xlabel='Command pressure [MPa]', ylabel='Porosity [%]')
    axes[1].set(xlabel='Time [ms]', ylabel='Minimum tetrahedron quality')
    axes[1].axhline(configs[1]['minimum_quality'], color='red', linestyle=':', label='Unchanged quality floor')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Rapid transient diagnostics — not quasi-static compaction results')
    fig.tight_layout()
    fig.savefig(out / 'replay-comparison.png', dpi=180)
    plt.close(fig)
    with np.load(current / 'last-state.npz') as state:
        x, t = state['points'], state['tets']
        faces = boundary(t)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.add_collection3d(Poly3DCollection(x[faces], facecolor='#88bddd', edgecolor='#294459', linewidth=.12, alpha=1.))
        for dim, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            setter(x[:, dim].min(), x[:, dim].max())
        ax.set_box_aspect(np.ptp(x, axis=0))
        ax.set(xlabel='x [mm]', ylabel='y [mm]', zlabel='z [mm]', title='Last accepted transient mesh — NOT equilibrium')
        fig.tight_layout()
        fig.savefig(out / 'stopped-mesh.png', dpi=180)
        plt.close(fig)
    hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    with zipfile.ZipFile(out.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                archive.write(p, str(Path(out.name) / p.relative_to(out)))
    print(json.dumps(dict(output=str(out), identical_configuration=result['identical_configuration'], files=len(hashes))))


if __name__ == '__main__':
    main()
