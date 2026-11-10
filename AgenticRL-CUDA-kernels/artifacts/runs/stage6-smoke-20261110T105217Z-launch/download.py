import concurrent.futures,json
from pathlib import Path,PurePosixPath
import modal
from modal.types import FileEntryType
project=Path(__file__).resolve().parents[3]
run='stage6-smoke-20261110T105217Z'
root=project/'artifacts/runs'/run
partial=root/run
if partial.exists():
    partial.rename(Path(__file__).with_name('partial-cli-download'))
volume=modal.Volume.from_name('agenticrl-cuda-artifacts').hydrate()
entries=[entry for entry in volume.iterdir('/'+run,recursive=True) if entry.type==FileEntryType.FILE]
print('Remote files:',len(entries),flush=True)
def download(entry):
    relative=PurePosixPath(entry.path.lstrip('/')).relative_to(run)
    assert '..' not in relative.parts
    target=root/str(relative)
    target.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with target.open('wb') as stream:
        for chunk in volume.read_file(entry.path):
            count+=stream.write(chunk)
    return {'path':str(relative),'bytes':count}
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for result in pool.map(download,entries):
        results.append(result)
        if len(results)%100==0:print('Downloaded',len(results),flush=True)
Path(__file__).with_name('download-manifest.json').write_text(json.dumps({'run_id':run,'files':results},indent=2)+'\n')
print('DOWNLOAD_COMPLETE',len(results),sum(r['bytes'] for r in results),flush=True)
