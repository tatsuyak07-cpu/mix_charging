"""Full prescribed-path replay, never interpolation of new active histories."""
from copy import deepcopy
import numpy as np
import pytest
from packing_fem.embedded_contact import EmbeddedContact
from packing_fem.explicit_periodic import Periodic
from packing_fem.area_contact import SurfaceResolutionError
from packing_fem.barrier_contact import barrier
from tools.verify_plane_contact import fixture


def trajectory(tolerance):
    X,t,_,_,upper=fixture(1,.015,.25);rules={};requests=0
    for attempt in range(20):
        contact=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,quadrature_rules=rules,
                                quadrature_tolerance=tolerance,error_budget='normalized',rule_strategy='pose_atlas')
        dissipation=0.;forward=None
        try:
            path=[(float(gap),0.) for gap in np.linspace(.025,.015,9)]
            path +=[(.015,float(slip)) for slip in np.linspace(0,.01,9)[1:]]
            path +=[(.015,float(slip)) for slip in np.linspace(.01,0,9)[1:]]
            for step,(gap,slip) in enumerate(path):
                _,_,x,H,_=fixture(1,gap,.25);x[upper,0]+=slip;H[0,2]+=slip
                response=contact.evaluate(x,x*0,H,H*0,.000125)
                assert response[-1]['friction_increment']>=-1e-14
                assert response[-1]['max_coulomb_ratio']<=.3+1e-12
                dissipation+=response[-1]['friction_increment'];contact.history=response[2];contact.accept_geometry(x,H)
                if step==16:forward=response
            old=deepcopy(contact.history);replayed=contact.replay_force_state(x,x*0,H)
            np.testing.assert_allclose(response[0],replayed[0],rtol=1e-12,atol=1e-13)
            for key in old:
                for field in ('spring','plastic','director','bary','anchor_ids'):np.testing.assert_array_equal(old[key][field],contact.history[key][field])
            return forward,response,dissipation,requests,rules
        except SurfaceResolutionError as error:
            feature=getattr(error,'feature',None)
            assert feature and feature.get('rules'),str(error)
            rules.update(feature['rules']);requests+=1
    raise AssertionError('Atlas prescribed-history replay budget exhausted')


def test_atlas_full_friction_path_matches_analytic_limit_and_tolerance_refinement():
    results=[];E,F,_=barrier(.015,20.,.02);spring=min(.01,.3*F/10.)
    for tolerance in (.001,.0005,.00025):
        forward,last,dissipation,requests,rules=trajectory(tolerance)
        assert requests>0 and any(r.get('atlas') for r in rules.values())
        assert forward[-1]['normal_energy']==pytest.approx(E,rel=.001)
        assert forward[-1]['tangent_energy']==pytest.approx(5*spring**2,rel=.002)
        assert dissipation>0
        X=fixture(1,.015,.25)[0]
        # Work-conjugate periodic DOFs, including image/macro reactions. Raw
        # forces on equivalent cell-cut representatives need not coincide.
        results.append((forward[-1]['tangent_energy'],dissipation,Periodic(X,1.).force(last[0],last[1])))
    for a,b in zip(results,results[1:]):
        assert abs(a[0]-b[0])/max(a[0],b[0])<.002
        assert abs(a[1]-b[1])/max(a[1],b[1])<.002
        assert np.max(abs(a[2]-b[2]))/F<.002
