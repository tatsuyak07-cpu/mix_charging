"""Conservative upper bound allowing arbitrary allocation at cut-line samples.

For diagnostic robustness ONLY, permit even fractional allocation of weights
within a parameter-padding band around each sample coordinate. No solver
weights, ownership or acceptance tolerance are changed.
"""
import argparse,json
from pathlib import Path
from tools.cut_balance import merge,intersect


def relaxed_intervals(face,padding=0.):
    if face['apex_mass_fraction']!=0:raise ValueError('This bound requires no samples at the opposite vertex')
    intervals=list(face['intervals']);A=face['area_mm2'];delta=face['area_tolerance_mm2']/A;total=face['total_mass_mm2']/A
    for i,s in enumerate(face['positions']):
        before=face['cumulative_fraction'][i];after=face['cumulative_fraction'][i+1]
        low=max(0.,s-padding,before-delta,1-total+before-delta)
        high=min(1.,s+padding,after+delta,1-total+after+delta)
        if low<=high and high>0 and low<1:intervals.append([low,high])
    return merge(intervals)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--analysis',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    s=json.loads(a.analysis.read_text());rows=[]
    for edge in s['edges']:
        if edge['missing_rule_faces']:continue
        for padding in (0.,1e-10,1e-8):
            common=[[0.,1.]]
            for face in edge['faces']:common=intersect(common,relaxed_intervals(face,padding))
            rows.append(dict(edge=edge['edge'],diagnostic_parameter_padding=padding,relaxed_common_intervals=common))
    summary=dict(scope=__doc__,analysis=str(a.analysis),no_common_interval_even_with_relaxation=all(not r['relaxed_common_intervals'] for r in rows),cases=rows,
                 production_tolerance_changed=False,weights_changed=False,packing_remesh_completed=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
