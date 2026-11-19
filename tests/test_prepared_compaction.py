import json
import numpy as np
import pytest
from tools.prepare_refined_compaction import prepare
from tools.run_implicit_compaction import run


def test_prepared_mesh_preserves_volume_and_boundary_and_binds_actual_input(tmp_path):
    X=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    mesh=tmp_path/'original.npz';np.savez_compressed(mesh,points=X,tets=np.array([[0,1,2,3]]))
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(length_mm=1.,young_mpa=10.,poisson=.33,friction=.3)))
    result=prepare(mesh,config,tmp_path/'prepared');c=json.loads((tmp_path/'prepared/config.json').read_text())
    assert result['tetrahedra']==8 and result['volume_mm3']==pytest.approx(1/6)
    assert result['boundary_area_mm2']==pytest.approx(result['initial']['boundary_area_mm2'])
    assert c['young_mpa']==10 and c['poisson']==.33 and c['friction']==.3
    assert c['required_mesh_sha256']==result['prepared_mesh_sha256'] and c['implicit_require_stress_free_initial_contact']
    assert c['require_prepared_mesh']


def test_compression_rejects_unprepared_input_before_loading(tmp_path):
    c=json.loads(open('configs/rigid-stage4-small-load-002.json').read())
    c.update(required_mesh_sha256='0'*64,implicit_sparse_solver='superlu')
    config=tmp_path/'config.json';config.write_text(json.dumps(c))
    X=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    mesh=tmp_path/'mesh.npz';np.savez_compressed(mesh,points=X,tets=np.array([[0,1,2,3]]))
    out=tmp_path/'rejected'
    with pytest.raises(ValueError,match='explicitly prepared mesh'):run(config,mesh,out)
    assert not (out/'state-0000.npz').exists() and not (out/'history.csv').exists()


def test_coarse_restart_cannot_override_the_prepared_refinement(tmp_path):
    c=json.loads(open('configs/rigid-stage4-small-load-002.json').read());c['implicit_sparse_solver']='superlu'
    config=tmp_path/'config.json';config.write_text(json.dumps(c))
    X=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]);t=np.array([[0,1,2,3]])
    mesh=tmp_path/'mesh.npz';np.savez_compressed(mesh,points=X,tets=t)
    prepare(mesh,config,tmp_path/'prepared')
    old=tmp_path/'old';old.mkdir();restart=old/'state.npz';np.savez_compressed(restart,X=X,t=t)
    with pytest.raises(ValueError,match='does not belong to the prepared mesh'):
        run(tmp_path/'prepared/config.json',tmp_path/'prepared/mesh.npz',tmp_path/'rejected',restart=restart)


def test_virgin_remesh_preload_is_rejected_before_material_history_changes(tmp_path,monkeypatch):
    import tools.run_implicit_compaction as solver
    import packing_fem.implicit_remesh as remesh
    c=json.loads(open('configs/rigid-stage4-small-load-002.json').read())
    c.update(length_mm=2.,rigid_inclusion=False,implicit_sparse_solver='superlu',implicit_remesh=True,
             implicit_remesh_quality=.99,implicit_require_stress_free_initial_contact=True)
    c.pop('initial_porosity_tolerance',None)
    config=tmp_path/'config.json';config.write_text(json.dumps(c))
    X=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]);t=np.array([[0,1,2,3]])
    mesh=tmp_path/'mesh.npz';np.savez_compressed(mesh,points=X,tets=t)
    contacts=[]
    class Contact:
        def __init__(self,p,preload):self.p=p;self.X=p.X;self.preload=preload
        def evaluate_coupled(self,q,old):
            g=np.zeros(self.p.ndof)
            if self.preload:g[0]=.1
            return (1e-4 if self.preload else 0.),g,None,int(self.preload)
    def make_contact(X,t,p,c):
        obj=Contact(p,bool(contacts));contacts.append(obj);return obj
    def propose(x,t,b,ep,V0,c,parent,X,L):
        return x.copy(),t.copy(),b.copy(),ep.copy(),V0.copy(),parent.copy(),{'_reference_points':X.copy()}
    class StopAfterGuard(RuntimeError):pass
    def stop(p,t,q,b,ep,inverse,V0,c,contact,*args,**kwargs):
        assert contact is contacts[0]
        np.testing.assert_array_equal(ep,0.);np.testing.assert_array_equal(b,np.eye(3)[None])
        raise StopAfterGuard()
    monkeypatch.setattr(solver,'contact_model',make_contact)
    monkeypatch.setattr(remesh,'refine_deformed_edges',propose)
    monkeypatch.setattr(solver,'equilibrium',stop)
    out=tmp_path/'rejected-remesh'
    with pytest.raises(StopAfterGuard):solver.run(config,mesh,out)
    report=json.loads((out/'remesh-virgin-contact-preflight-0000.json').read_text())
    assert not report['passed'] and not (out/'state-0001.npz').exists()
