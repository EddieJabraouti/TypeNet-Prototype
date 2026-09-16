"""Outcome-blind census of distinct available typing observations."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / 'runs'

def inspect_file(path):
    raw = path.read_bytes()
    lines = raw.splitlines()
    if not lines:
        return dict(id=path.stem.split('_')[0], sessions=0,
                    nonoverlapping_50key_windows=0, full_50key_windows=0,
                    keystrokes=0, sha256=hashlib.sha256(raw).hexdigest())
    fields = lines[0].split(b'\t')
    si, pi, ri, ki, ei = [fields.index(x) for x in (b'TEST_SECTION_ID', b'PRESS_TIME', b'RELEASE_TIME', b'KEYCODE', b'KEYSTROKE_ID')]
    sessions = Counter()
    seen = set()
    for line in lines[1:]:
        f = line.split(b'\t')
        try:
            event = tuple(int(f[j]) for j in (pi,ri,ki,ei))
            key = (f[si], event)
            if key in seen: continue
            seen.add(key)
            sessions[f[si]] += 1
        except (ValueError, IndexError): continue
    windows = sum(n // 50 + (n % 50 >= 6) for n in sessions.values())
    return dict(id=path.stem.split('_')[0], sessions=sum(n>=6 for n in sessions.values()),
                nonoverlapping_50key_windows=windows, full_50key_windows=sum(n//50 for n in sessions.values()),
                keystrokes=sum(sessions.values()), sha256=hashlib.sha256(raw).hexdigest())

def main():
    OUT.mkdir(exist_ok=True)
    start = time.monotonic()
    directories = [ROOT/'data/Keystrokes/files', ROOT/'data/keystrokes_f/files']
    reports = {}
    for directory in directories:
        cache = OUT / (directory.parent.name + '_inventory.json')
        if cache.exists():
            rows=json.loads(cache.read_text())
        else:
            paths=sorted(directory.glob('[0-9]*_keystrokes.txt'))
            rows=[]
            with ProcessPoolExecutor(max_workers=4) as pool:
                for i,row in enumerate(pool.map(inspect_file, paths, chunksize=128)):
                    rows.append(row)
                    if (i+1)%20000==0: print(directory.parent.name, i+1, round(time.monotonic()-start), flush=True)
            cache.write_text(json.dumps(rows))
        report=dict(participants=len(rows), session_histogram=dict(sorted(Counter(r['sessions'] for r in rows).items())),
                    maximum_windows=max(r['nonoverlapping_50key_windows'] for r in rows),
                    maximum_full_windows=max(r['full_50key_windows'] for r in rows),
                    eligible_equal_sides={str(n):sum(r['nonoverlapping_50key_windows']>=2*n for r in rows) for n in [5,7,10,20,50,100,200]},
                    eligible_full_windows_equal_sides={str(n):sum(r['full_50key_windows']>=2*n for r in rows) for n in [5,7,10,20,50,100,200]})
        reports[directory.parent.name]=report
        print(json.dumps({directory.parent.name:report}), flush=True)
    a,b=[json.loads((OUT/(p.parent.name+'_inventory.json')).read_text()) for p in directories]
    index={r['id']:r['sha256'] for r in a}
    reports['overlap']=dict(shared_ids=sum(r['id'] in index for r in b), identical_files=sum(index.get(r['id'])==r['sha256'] for r in b), second_only=sum(r['id'] not in index for r in b))
    (OUT/'availability.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports['overlap']),flush=True)

if __name__=='__main__': main()
