import numpy as np
from packing_fem.line_search import bounded_wolfe


def test_rosenbrock_search_satisfies_energy_and_curvature_conditions():
    def values(x):
        a,b=x;e=100*(b-a*a)**2+(1-a)**2
        g=np.array([-400*a*(b-a*a)+2*(a-1),200*(b-a*a)])
        return e,g,None
    x=np.array([-1.2,1.]);e,g,_=values(x);d=-g
    result=bounded_wolfe(x,d,e,g,values,1.)
    assert result is not None and result['wolfe']
    assert result['energy']<e
    assert abs(result['gradient']@d)<=.9*abs(g@d)


def test_search_rejects_invalid_domain_before_accepting_a_point():
    def values(x):
        if x[0]<=0:raise ValueError('Outside admissible domain')
        return float((x[0]-.2)**2),np.array([2*(x[0]-.2)]),x.copy()
    x=np.array([1.]);e,g,_=values(x);d=np.array([-3.])
    result=bounded_wolfe(x,d,e,g,values,1.)
    assert result is not None
    assert result['payload'][0]>0
    assert result['energy']<e
