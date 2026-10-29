"""Checksummed, pickle-free full explicit-state checkpoint containers."""
import hashlib,json,math
from pathlib import Path
import numpy as np


def save(path,state,metadata):
    path=Path(path);path.mkdir(parents=True,exist_ok=False);arrays={};groups={};identities={}
    def pack(value):
        if isinstance(value,np.ndarray):
            if value.dtype.hasobject:raise TypeError('Object arrays are not pickle-free checkpoint fields')
            if id(value) in identities:return identities[id(value)]
            if value.size<=256:
                signature=(value.dtype,value.shape)
                if signature not in groups:groups[signature]=(f'b{len(groups)}',[])
                key,items=groups[signature];tag={'array':key,'row':len(items)};items.append(value)
            else:
                key=f'a{len(arrays)}';arrays[key]=value;tag={'array':key}
            identities[id(value)]=tag
            return tag
        if isinstance(value,np.generic):return pack(value.item())
        if isinstance(value,dict):return {'dict':[[pack(k),pack(v)] for k,v in value.items()]}
        if isinstance(value,tuple):return {'tuple':[pack(v) for v in value]}
        if isinstance(value,list):return [pack(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value):return {'float':str(value)}
        if value is None or isinstance(value,(str,int,float,bool)):return value
        raise TypeError(f'Unsupported checkpoint field: {type(value).__name__}')
    tree=pack(state)
    for key,items in groups.values():arrays[key]=np.stack(items)
    np.savez_compressed(path/'arrays.npz',**arrays)
    payload=dict(schema=2,metadata=metadata,state=tree,array_sha256=hashlib.sha256((path/'arrays.npz').read_bytes()).hexdigest())
    raw=json.dumps(payload,allow_nan=False,separators=(',',':')).encode();(path/'state.json').write_bytes(raw)
    (path/'manifest.json').write_text(json.dumps({'state_sha256':hashlib.sha256(raw).hexdigest()},indent=2)+'\n')


def load(path):
    path=Path(path);raw=(path/'state.json').read_bytes();manifest=json.loads((path/'manifest.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=manifest['state_sha256']:raise ValueError('Checkpoint state checksum mismatch')
    payload=json.loads(raw)
    if payload['schema'] not in (1,2) or hashlib.sha256((path/'arrays.npz').read_bytes()).hexdigest()!=payload['array_sha256']:raise ValueError('Checkpoint array checksum/schema mismatch')
    with np.load(path/'arrays.npz',allow_pickle=False) as archive:arrays={key:archive[key].copy() for key in archive.files}
    views={}
    def unpack(value):
        if isinstance(value,list):return [unpack(v) for v in value]
        if not isinstance(value,dict):return value
        if 'array' in value:
            if 'row' not in value:return arrays[value['array']]
            key=(value['array'],value['row'])
            if key not in views:views[key]=np.asarray(arrays[key[0]][key[1]])
            return views[key]
        if 'dict' in value:return {unpack(k):unpack(v) for k,v in value['dict']}
        if 'tuple' in value:return tuple(unpack(v) for v in value['tuple'])
        if 'float' in value:return float(value['float'])
        raise ValueError('Unknown checkpoint value tag')
    return unpack(payload['state']),payload['metadata']


def contact_state(contact):
    inner=getattr(contact,'inner',contact);guard=getattr(inner,'guard',inner)
    attributes=('neighbor_geometry','neighbor_bounds','neighbors','neighbor_builds','accepted_geometry','accepted_shifts','minimum_separation','diagonal','trial_diagonal','trial_separation','trial_shifts')
    state=dict(history=inner.history,guard={name:getattr(guard,name) for name in attributes if hasattr(guard,name)})
    for name in ('accepted_force_history','trial_force_history','accepted_audit','trial_audit','accepted_history'):
        if name in inner.__dict__:state[name]=getattr(inner,name)
    if hasattr(contact,'inner'):
        state['embedded']={name:getattr(contact,name) for name in ('_candidate_build','_face_candidates') if name in contact.__dict__}
    return state


def restore_contact(contact,state):
    inner=getattr(contact,'inner',contact);guard=getattr(inner,'guard',inner)
    inner.history=state['history']
    for name,value in state['guard'].items():setattr(guard,name,value)
    for name,value in state.items():
        if name not in ('history','guard','embedded'):setattr(inner,name,value)
    for name,value in state.get('embedded',{}).items():setattr(contact,name,value)
