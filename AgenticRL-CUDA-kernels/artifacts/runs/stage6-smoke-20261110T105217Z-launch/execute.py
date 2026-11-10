import hashlib,json,os,subprocess
from pathlib import Path
project=Path(__file__).resolve().parents[3]
request_path=project/'artifacts/runs/multi-lora-local-20261110T105217Z/h200-smoke-request.json'
request=json.loads(request_path.read_text())
source=Path(request['cwd'])
manifest=json.loads((source/'task-source-manifest.json').read_text())
for name,digest in manifest['files'].items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
record={'authorized_by':'User: execute the prepared Modal H200 smoke request',
        'request':request,'verified_source_files':len(manifest['files'])}
Path(__file__).with_name('execution-request.json').write_text(json.dumps(record,indent=2)+'\n')
print('Verified frozen source:',len(manifest['files']),'files',flush=True)
raise SystemExit(subprocess.call(request['command'],cwd=source,env={**os.environ,**request['environment'],'PYTHONDONTWRITEBYTECODE':'1'}))
