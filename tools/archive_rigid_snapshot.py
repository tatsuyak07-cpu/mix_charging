"""Freeze inactive calculations and sources in small, individually extractable archives."""
import argparse,concurrent.futures,hashlib,json,lzma,tarfile
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def archive(root,out,excluded):
    root=Path(root);out=Path(out)
    if (out/'archives.json').exists():raise FileExistsError('Snapshot already archived')
    items=[]
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root)
        if any(name in relative.parts or relative.name==name+'.log' for name in excluded):continue
        if path.is_file():items.append((path,'raw/'+relative.as_posix()))
    for directory in ['packing_fem','tools','tests']:
        items.extend((p,'source/'+p.as_posix()) for p in sorted(Path(directory).glob('*.py')))
    items.extend((p,'source/'+p.as_posix()) for p in sorted(Path('configs').glob('rigid-cylinder*.json')))
    for name in ['AGENTS.md','README.md','docs/rigid-inclusion-development.md','requirements.txt','requirements-implicit.txt','requirements-lock.txt','pyproject.toml']:
        path=Path(name)
        if path.exists():items.append((path,'source/'+path.as_posix()))
    items.extend((p,'source/'+p.as_posix()) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['README.md','archives.json','raw-archive-files.json','archive-verification.json'])
    groups=[];current=[];size=0;limit=48*1024**2
    for path,name in items:
        if path.is_symlink():raise ValueError('Snapshot source must be a regular file: '+str(path))
        length=path.stat().st_size
        if length>limit:raise ValueError('File exceeds the independently extractable part budget: '+str(path))
        if current and size+length>limit:groups.append(current);current=[];size=0
        current.append(dict(path=path,name=name,bytes=length,sha256=digest(path)));size+=length
    if current:groups.append(current)
    def compress(index_group):
        index,group=index_group;path=out.parent/f'{out.name}-part-{index:02d}.tar.xz'
        options=[dict(id=lzma.FILTER_LZMA2,dict_size=64*1024**2,mode=lzma.MODE_FAST,mf=lzma.MF_HC3,nice_len=32)]
        with path.open('xb') as stream,lzma.LZMAFile(stream,'w',filters=options) as compressed,tarfile.open(fileobj=compressed,mode='w|',dereference=True) as tar:
            for item in group:tar.add(item['path'],arcname=item['name'],recursive=False)
        expected={item['name']:item['sha256'] for item in group};seen=set()
        with tarfile.open(path,'r:xz') as tar:
            for member in tar:
                if not member.isfile() or member.name not in expected:raise ValueError('Unexpected archive member')
                with tar.extractfile(member) as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
                if actual!=expected[member.name]:raise ValueError('Snapshot changed or archive is corrupt: '+member.name)
                seen.add(member.name)
        if seen!=set(expected):raise ValueError('Archive members missing')
        if path.stat().st_size>=100*1024**2:raise ValueError('Archive exceeds GitHub hard file limit')
        record=dict(file=path.name,bytes=path.stat().st_size,sha256=digest(path),members=len(group),uncompressed_bytes=sum(i['bytes'] for i in group),verified=True)
        print(json.dumps(record),flush=True);return index,record
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:parts=sorted(executor.map(compress,enumerate(groups,1)))
    records=[part for _,part in parts]
    with (out/'archives.json').open('x') as stream:json.dump(records,stream,indent=2);stream.write('\n')
    manifest=[dict(archive=records[index]['file'],file=item['name'],bytes=item['bytes'],sha256=item['sha256']) for index,group in enumerate(groups) for item in group]
    with (out/'raw-archive-files.json').open('x') as stream:json.dump(dict(excluded_live_runs=excluded,files=manifest),stream,indent=2);stream.write('\n')
    with (out/'archive-verification.json').open('x') as stream:json.dump(dict(passed=True,archives=len(records),files=len(manifest),all_member_sha256_checked=True,live_calculation_excluded=excluded),stream,indent=2);stream.write('\n')
    with (out/'README.md').open('a') as stream:
        stream.write('\n## 実行ソースと終了済み試行の全保存データ\n\n各tar.xzは単独で展開できます。全パートを同じ保存先へ展開します。計算中の'+', '.join(excluded)+'は除外しています。\n[全ファイルSHA-256](raw-archive-files.json)・[アーカイブ検証](archive-verification.json)。\n\n')
        for part in records:stream.write(f"- [{part['file']}](../{part['file']})\n")
        stream.write('\n展開後の`source/`に実行コードと画像・CSV、`raw/`に終了済み計算の入力・保存状態・不採用試行が入っています。元の相対パスを復元する場合は、`source/results/rigid-inclusion-001/`を作成し`raw/`の内容をコピーしてから`source/`で実行します。\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--out',required=True);parser.add_argument('--exclude',action='append',default=[]);args=parser.parse_args();archive(args.root,args.out,args.exclude)
