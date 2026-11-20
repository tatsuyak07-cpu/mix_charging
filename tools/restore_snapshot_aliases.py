"""Restore identical archived files to their original names, verifying SHA-256."""
import argparse,hashlib,json,shutil
from pathlib import Path,PurePosixPath


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def restore(manifest,directory):
    root=Path(directory).resolve();report=json.loads(Path(manifest).read_text())
    aliases={item['file']:item for item in report.get('content_aliases',[])}
    # Also accept a manifest carrying aliases inherited from an older format.
    aliases.update({item['file']:item for item in report.get('reused_previous_files',[]) if 'target' in item})
    def safe_path(name):
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:raise ValueError('Invalid archive path: '+name)
        path=root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(root):raise ValueError('Archive path escapes extraction directory: '+name)
        return path
    restored=set();visiting=set();created=0
    def materialize(name):
        nonlocal created
        if name in restored:return
        if name in visiting:raise ValueError('Cyclic snapshot aliases')
        item=aliases[name];visiting.add(name);target=item['target']
        if target in aliases:materialize(target)
        source=safe_path(target);destination=safe_path(name)
        if not source.is_file() or source.stat().st_size!=item['bytes'] or digest(source)!=item['sha256']:raise ValueError('Alias source is missing or corrupt: '+target)
        if destination.exists():
            if not destination.is_file() or destination.stat().st_size!=item['bytes'] or digest(destination)!=item['sha256']:raise ValueError('Refusing to overwrite different existing file: '+name)
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            with source.open('rb') as src,destination.open('xb') as dst:shutil.copyfileobj(src,dst)
            if digest(destination)!=item['sha256']:raise ValueError('Restored alias failed SHA-256: '+name)
            created+=1
        visiting.remove(name);restored.add(name)
    for name in aliases:materialize(name)
    return dict(passed=True,aliases_verified=len(restored),files_created=created)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--directory',required=True);a=p.parse_args()
    print(json.dumps(restore(a.manifest,a.directory)),flush=True)
