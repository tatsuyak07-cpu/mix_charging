"""High-precision and rigid-motion checks of the saved near-zero EE pair."""
import argparse,importlib.util,json
from decimal import Decimal,localcontext
from fractions import Fraction
from pathlib import Path
import numpy as np
from packing_fem.area_contact import AreaContact
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.contact_geometry import closest_triangle
from packing_fem.contact_audit import plain


def decimal_pair(points,precision=80):
    """Interior/interior closest pair of the exact binary64 input geometry.

    Decimal is an independent reference for this branch, not a replacement
    contact solver or a statement about uncertain physical coordinates.
    """
    with localcontext() as context:
        context.prec=precision
        vec=lambda p:tuple(Decimal.from_float(float(v)) for v in p)
        sub=lambda a,b:tuple(x-y for x,y in zip(a,b))
        add=lambda a,b:tuple(x+y for x,y in zip(a,b))
        mul=lambda a,b:tuple(x*b for x in a)
        dot=lambda a,b:sum(x*y for x,y in zip(a,b))
        a,b,c,d=map(vec,points);u=sub(b,a);v=sub(d,c);r=sub(a,c)
        A=dot(u,u);B=dot(u,v);C=dot(v,v);ru=dot(r,u);rv=dot(r,v);det=A*C-B*B
        if det<=0:raise ValueError('Independent reference requires nonparallel edges')
        s=(B*rv-C*ru)/det;t=(A*rv-B*ru)/det
        if not 0<s<1 or not 0<t<1:raise ValueError('Independent reference requires interior closest points')
        q=add(a,mul(u,s));p=add(c,mul(v,t));toward=sub(p,q);distance=dot(toward,toward).sqrt()
        return dict(s=str(s),t=str(t),distance=str(distance),q=q,p=p,toward=toward)


def decimal_cosine(toward,point,vertex,precision=80):
    with localcontext() as context:
        context.prec=precision
        v=[Decimal.from_float(float(x))-y for x,y in zip(vertex,point)]
        dot=sum(a*b for a,b in zip(toward,v))
        return float(dot/(sum(a*a for a in toward).sqrt()*sum(a*a for a in v).sqrt()))


def exact_separating_projection(first,second,point,direction):
    """Exact rational projection of binary64 vertices onto a chosen plane.

    The direction need not be an exact closest normal. Strictly disjoint
    projection intervals prove these two convex hulls do not intersect.
    """
    q=tuple(Fraction(v) for v in point);n=tuple(Fraction(v) for v in direction)
    project=lambda v:sum(w*(Fraction.from_float(float(x))-y) for x,y,w in zip(v,q,n))
    maximum=max(map(project,first));minimum=min(map(project,second))
    return dict(strict_separation=maximum<minimum,maximum_first=float(maximum),minimum_second=float(minimum),projection_gap=float(minimum-maximum))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads((a.source/'config.json').read_text());cap=json.loads((a.source/'contact-audit.json').read_text())
    with np.load(a.source/'last-state.npz') as z:X=z['reference_points'];x=z['points'];t=z['tets'];H=deformation(z['generalized_position'][-6:])*c['length_mm']
    ids=np.array([175,384,385,401]);shift=np.zeros(3,int)
    contact=AreaContact(X,t,Periodic(X,c['length_mm']),c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],thickness=c['barrier_thickness_mm'],multiple_patches=True,edge_stationarity_filter=True,closest_tie_rule='clarke')
    stationarity=contact.surface_stationarity
    archived=Path('exports/closest-selection-001/validation/pressure-run/source/surface_stationarity.py')
    spec=importlib.util.spec_from_file_location('packing_fem._near_edge_reference',archived);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    legacy=old.SurfaceStationarity(X,contact.faces,contact.classes,c['length_mm'])
    original=legacy.classify(ids,shift,x,H);corrected=stationarity.classify(ids,shift,x,H)
    reference=decimal_pair(x[ids]);reference100=decimal_pair(x[ids],100)
    derivatives=[];stars=[[],[]]
    for side,part,q,toward in ((0,ids[:2],reference['q'],reference['toward']),(1,ids[2:],reference['p'],tuple(v.copy_negate() for v in reference['toward']))):
        for face,lift in stationarity.star(part,shift):
            for node in face:
                stars[side].append(x[node]+H@lift)
                derivatives.append(dict(side=side,face=face.tolist(),vertex=int(node),on_original_edge=bool(node in part),cosine=decimal_cosine(toward,q,x[node]+H@lift)))
    separation=exact_separating_projection(stars[0],stars[1],reference['q'],reference['toward'])
    variants=[]
    for angle in (0.,.37,1.19):
        R=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0,0,1.]])
        for offset in ([0.,0,0],[.31,-.27,.19],[100.,-50.,23.]):
            y=x@R.T+offset
            variants.append(dict(kind='rigid_motion',angle=angle,translation=offset,legacy=legacy.classify(ids,shift,y,R@H),corrected=stationarity.classify(ids,shift,y,R@H)))
    for node in ids:
        for axis in range(3):
            for delta in (-1e-12,1e-12):
                y=x.copy();y[node,axis]+=delta
                variants.append(dict(kind='coordinate_noise',node=int(node),axis=axis,delta_mm=delta,legacy=legacy.classify(ids,shift,y,H),corrected=stationarity.classify(ids,shift,y,H)))
    contact.history={tuple(item['key']):{k:np.array(v) if isinstance(v,list) else v for k,v in item['state'].items()} for item in cap['committed_history']}
    endpoint_contacts=[]
    for key,state in contact.history.items():
        if key[1] not in ids:continue
        face=contact.faces[state['face']];tri=x[face]+H@np.array(state['shift']);distance=float(np.linalg.norm(x[key[1]]-closest_triangle(x[key[1]],tri)[0]))
        endpoint_contacts.append(dict(node=int(key[1]),target_face=face.tolist(),target_shift=state['shift'].tolist(),point_triangle_gap_mm=distance,source_distance_to_EE_closest_point_mm=float(np.linalg.norm(x[key[1]]-np.array([float(v) for v in reference['p']])))))
    error=None;report=None
    try:report=contact.evaluate(x,x*0,H,H*0,0.)[-1]
    except ValueError as exc:error=str(exc)
    checks=dict(legacy_false_tangent_witness_reproduced=original['dominated'] and original['witness']['vertex'] in ids,
                high_precision_tangent_derivatives_zero=all(abs(r['cosine'])<1e-60 for r in derivatives if r['on_original_edge']),
                high_precision_offedge_derivatives_negative=all(r['cosine']<-.01 for r in derivatives if not r['on_original_edge']),
                reference_precision_agreement=abs(Decimal(reference['distance'])-Decimal(reference100['distance']))<Decimal('1e-70'),
                corrected_pair_not_screened=not corrected['dominated'],
                all_33_variants_not_screened=len(variants)==33 and all(not r['corrected']['dominated'] for r in variants),
                endpoint_contact_is_not_at_EE_closest_point=bool(endpoint_contacts) and all(r['point_triangle_gap_mm']>1e-3 for r in endpoint_contacts),
                endpoint_guard_rejects_unresolved_pair=error is not None and 'Edge-only' in error)
    checks['exact_local_star_convex_hulls_separated']=separation['strict_separation']
    result=plain(dict(scope=__doc__,checks=checks,passed=all(checks.values()),legacy=original,corrected=corrected,
                      decimal_distance_mm=reference['distance'],decimal_s=reference['s'],decimal_t=reference['t'],derivatives=derivatives,variants=variants,
                      exact_local_star_projection=separation,
                      endpoint_contacts=endpoint_contacts,accepted_geometry_evaluation_error=error,accepted_geometry_report=report,
                      coverage_limitation='The corrected guard rejects this near-zero pair despite another endpoint contact; general quadrature/history convergence remains unverified.',
                      time_advanced=False,physical_solution_verified=False))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'verify_near_edge.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(checks=checks,passed=result['passed'],endpoint_contacts=endpoint_contacts),indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
