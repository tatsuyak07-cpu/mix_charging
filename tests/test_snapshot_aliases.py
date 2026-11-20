import hashlib,json,tarfile
from pathlib import Path
import pytest
from tools.archive_rigid_snapshot import archive
from tools.restore_snapshot_aliases import restore


def prepare(tmp_path,monkeypatch):
    helper=(Path(__file__).resolve().parents[1]/'tools/restore_snapshot_aliases.py').read_bytes()
    monkeypatch.chdir(tmp_path)
    Path('tools').mkdir();Path('tools/restore_snapshot_aliases.py').write_bytes(helper)
    root=Path('calculation');root.mkdir()
    (root/'a.npz').write_bytes(b'original saved state')
    (root/'b.npz').write_bytes(b'original saved state')
    (root/'helper-copy.py').write_bytes(helper)
    return root


def unpack(out,destination):
    for item in json.loads((out/'archives.json').read_text()):
        with tarfile.open(out.parent/item['file'],'r:xz') as t:t.extractall(destination,filter='data')


def test_deduplicated_archive_restores_every_original_and_keeps_bootstrap_tool(tmp_path,monkeypatch):
    root=prepare(tmp_path,monkeypatch);out=Path('exports/one');out.mkdir(parents=True)
    archive(root,out,[],deduplicate_content=True)
    manifest=out/'raw-archive-files.json';report=json.loads(manifest.read_text())
    assert len(report['content_aliases'])==1
    extracted=Path('extracted');extracted.mkdir();unpack(out,extracted)
    assert (extracted/'source/tools/restore_snapshot_aliases.py').is_file()
    assert not (extracted/'raw/b.npz').exists()
    assert restore(manifest,extracted)==dict(passed=True,aliases_verified=1,files_created=1)
    for path in root.iterdir():assert (extracted/'raw'/path.name).read_bytes()==path.read_bytes()
    assert restore(manifest,extracted)['files_created']==0
    assert json.loads((out/'archive-verification.json').read_text())['passed']


def test_incremental_snapshot_preserves_old_alias_when_its_original_target_changes(tmp_path,monkeypatch):
    root=prepare(tmp_path,monkeypatch);first=Path('exports/first');first.mkdir(parents=True)
    archive(root,first,[],deduplicate_content=True)
    extracted=Path('extracted');extracted.mkdir();unpack(first,extracted)
    # Do not restore the first snapshot's aliases before applying the update.
    (root/'a.npz').write_bytes(b'new saved state')
    second=Path('exports/second');second.mkdir()
    archive(root,second,[],first/'raw-archive-files.json',True)
    unpack(second,extracted);restore(second/'raw-archive-files.json',extracted)
    assert (extracted/'raw/a.npz').read_bytes()==b'new saved state'
    assert (extracted/'raw/b.npz').read_bytes()==b'original saved state'


@pytest.mark.parametrize('failure',['corrupt_source','different_destination','cycle','escape'])
def test_alias_restore_rejects_invalid_inputs_without_overwriting_existing_files(tmp_path,failure):
    data=b'checkpoint';sha=hashlib.sha256(data).hexdigest()
    (tmp_path/'original').write_bytes(data)
    alias=dict(file='copy',target='original',bytes=len(data),sha256=sha)
    if failure=='corrupt_source':(tmp_path/'original').write_bytes(b'corrupt')
    elif failure=='different_destination':(tmp_path/'copy').write_bytes(b'existing different state')
    elif failure=='cycle':alias['target']='copy'
    elif failure=='escape':alias['file']='../outside'
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(content_aliases=[alias])))
    with pytest.raises(ValueError):restore(manifest,tmp_path)
    if failure=='different_destination':assert (tmp_path/'copy').read_bytes()==b'existing different state'
    else:assert not (tmp_path/'copy').exists()
