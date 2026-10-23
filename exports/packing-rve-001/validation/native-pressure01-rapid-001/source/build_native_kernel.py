"""Explicit build step, separate from calculation and without downloads."""
import argparse,hashlib,json,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
    source=Path('native/element_kernel.cpp');digest=hashlib.sha256(source.read_bytes()).hexdigest();target=a.directory/f'element-{digest}.so'
    if not target.exists():subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared',str(source),'-o',str(target)],check=True)
    print(json.dumps(dict(library=str(target.resolve()),source_sha256=digest,flags=['-O3','-std=c++17','-fPIC','-shared'],fast_math=False)),flush=True)


if __name__=='__main__':main()
