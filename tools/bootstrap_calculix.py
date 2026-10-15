"""Install verified Debian CalculiX binaries locally without root access.

SHA256 values below were read from Debian bookworm Packages after APT
verified the signed InRelease. Do not change them to bypass a failed check.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import urllib.request

PACKAGES=(
    ('pool/main/c/calculix-ccx/calculix-ccx_2.20-1_amd64.deb',
     'a448239f3caf5a324f9589a636e595e3e676153489f75843e509decb6809f346'),
    ('pool/main/s/spooles/libspooles2.2_2.2-14_amd64.deb',
     '51faa6a5aad9aba20beadffba5d4d9e28fa7a5ef426a1d4fa16c825fd80034a0'),
    ('pool/main/c/calculix-ccx-doc/calculix-ccx-doc_2.11-1.1_all.deb',
     '163bb023e144a313b9a8fe5a1920934a0438977dba78f48851af625d038621ae'),
)


def main():
    root=Path('/workspace/environment/calculix');root.mkdir(parents=True,exist_ok=True)
    for source,digest in PACKAGES:
        archive=root/Path(source).name
        data=archive.read_bytes() if archive.exists() else urllib.request.urlopen(
            'https://deb.debian.org/debian/'+source,timeout=60).read()
        if hashlib.sha256(data).hexdigest()!=digest:
            raise RuntimeError(f'Checksum verification failed: {archive.name}')
        if not archive.exists():
            archive.write_bytes(data)
        subprocess.run(['dpkg-deb','-x',str(archive),str(root)],check=True)
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']=str(root/'usr/lib/x86_64-linux-gnu')+(
        os.pathsep+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    executable=root/'usr/bin/ccx'
    dependencies=subprocess.run(['ldd',str(executable)],env=env,capture_output=True,text=True,check=True)
    if 'not found' in dependencies.stdout:
        raise RuntimeError('Missing system libraries: '+dependencies.stdout)
    version=subprocess.run([str(executable),'-v'],env=env,capture_output=True,text=True)
    # ccx -v deliberately exits 201 even when successfully printing version.
    if 'Version 2.20' not in version.stdout:
        raise RuntimeError('Unexpected CalculiX version: '+version.stdout+version.stderr)
    print('CalculiX 2.20 and local dependencies verified')


if __name__=='__main__':
    main()
