"""Optional prebuilt CPU kernels; never install or compile during calculation."""
import ctypes,hashlib
from pathlib import Path
import numpy as np
from .explicit_material import constants

_CACHE={}
D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
I=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
N=ctypes.c_int64;F=ctypes.c_double


class NativeKernel:
    def __init__(self,path):
        self.path=Path(path).resolve();self.lib=ctypes.CDLL(str(self.path));self.lib.pf_geometry.argtypes=[N,D,I,D,D,D];self.lib.pf_advance.argtypes=[N,D,D,D,F,F,F,F,D,D]
        self.lib.pf_internal.argtypes=[N,N,D,I,D,D,F,F,F,F,D,D,D,D,D,D];self.lib.pf_quality.argtypes=[N,D,I,D];self.lib.pf_bound.argtypes=[N,N,D,I,D,F,F,D,D,ctypes.c_int,ctypes.c_int,D]
        for name in ('pf_geometry','pf_advance','pf_internal','pf_quality','pf_bound'):getattr(self.lib,name).restype=ctypes.c_int

    def identity(self):return dict(library=str(self.path),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),backend='optional_native_CPU_C++17_no_fast_math')
    @staticmethod
    def array(a,shape=None):
        a=np.ascontiguousarray(a,dtype=np.float64)
        if shape is not None and a.shape!=shape:raise ValueError('Native kernel shape mismatch')
        return a
    @staticmethod
    def mesh(x,t):
        x=np.ascontiguousarray(x,dtype=np.float64);t=np.ascontiguousarray(t,dtype=np.int64)
        if x.ndim!=2 or x.shape[1]!=3 or t.ndim!=2 or t.shape[1]!=4 or not len(t) or t.min()<0 or t.max()>=len(x):raise ValueError('Native mesh indices/shape invalid')
        return x,t
    @staticmethod
    def check(code):
        if code:raise ValueError(f'Native element {code-1}: invalid geometry/tensor or unconverged Jacobi eigensolve')
    @staticmethod
    def parameters(c):
        constants(c);return [float(c[k]) for k in ('young_mpa','poisson','yield_mpa','hardening_mpa')]

    def geometry(self,x,t):
        x,t=self.mesh(x,t);n=len(t);V=np.empty(n);g=np.empty((n,4,3));D=np.empty((n,3,3));self.check(self.lib.pf_geometry(n,x,t,V,g,D));return V,g,D

    def advance(self,b,ep,increment,c):
        b=self.array(b);n=len(b);b=self.array(b,(n,3,3));ep=self.array(ep,(n,));inc=self.array(increment,(n,3,3));out=np.empty_like(b);pe=np.empty(n)
        self.check(self.lib.pf_advance(n,b,ep,inc,*self.parameters(c),out,pe));return out,pe

    def internal_force(self,x,t,b,ep,c):
        x,t=self.mesh(x,t);n=len(t);b=self.array(b,(n,3,3));ep=self.array(ep,(n,));force=np.empty_like(x);V=np.empty(n);energy=np.empty(n);J=np.empty(n);D=np.empty((n,3,3));sigma=np.empty_like(b)
        self.check(self.lib.pf_internal(n,len(x),x,t,b,ep,*self.parameters(c),force,V,energy,J,D,sigma));return force,V,energy,J,D,sigma

    def quality(self,x,t):
        x,t=self.mesh(x,t);q=np.empty(len(t));self.check(self.lib.pf_quality(len(t),x,t,q));return q

    def bound(self,x,t,mass,c,stress=None,elastic_b=None):
        x,t=self.mesh(x,t);n=len(t);mass=self.array(mass,(len(x),));self.parameters(c);rows=np.empty_like(x);sigma=np.empty(1) if stress is None else self.array(stress,(n,3,3));b=np.empty(1) if elastic_b is None else self.array(elastic_b,(n,3,3))
        self.check(self.lib.pf_bound(n,len(x),x,t,mass,float(c['young_mpa']),float(c['poisson']),sigma,b,int(stress is not None),int(elastic_b is not None),rows));return float(rows.max())


def load(c):
    if not c.get('native_library'):raise ValueError('Explicit prebuild and native_library path required')
    path=str(Path(c['native_library']).resolve())
    if path not in _CACHE:_CACHE[path]=NativeKernel(path)
    return _CACHE[path]
