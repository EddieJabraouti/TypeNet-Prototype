"""Read-only local availability audit, not a clinical or model evaluation."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import zipfile

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/clinical/tappy'
users={}
with zipfile.ZipFile(BASE/'Archived-users.zip') as archive:
    for name in archive.namelist():
        if not name.endswith('.txt'): continue
        uid=Path(name).stem.removeprefix('User_')
        users[uid]=dict(line.split(': ',1) for line in archive.read(name).decode('utf8',errors='replace').splitlines() if ': ' in line)
counts=Counter(); days=defaultdict(set); bad=0; total=0
with zipfile.ZipFile(BASE/'Archived-Data.zip') as archive:
    for name in archive.namelist():
        if not name.endswith('.txt'): continue
        with archive.open(name) as f:
            for line in f:
                total+=1
                fields=line.decode('utf8',errors='replace').strip().split('\t')
                try:
                    uid,date=fields[:2]
                    hold,latency,flight=[float(fields[j]) for j in (4,6,7)]
                    if not (len(uid)==10 and len(date)==6 and date.isdigit() and fields[3] in ('L','R','S') and
                            all(math.isfinite(x) for x in (hold,latency,flight)) and hold>0 and latency>0):
                        raise ValueError()
                except (ValueError,IndexError): bad+=1; continue
                counts[uid]+=1; days[uid].add(date)
controls={u:n for u,n in counts.items() if users.get(u,{}).get('Parkinsons')=='False'}
result=dict(total_rows=total,structurally_valid_timing_rows=sum(counts.values()),invalid_rows=bad,
            participants_with_valid_rows=len(counts),participants_with_metadata=len(users),
            self_reported_non_PD_with_valid_rows=len(controls),
            all_record_count_upper_bounds={str(n):sum(k>=2*n*50 for k in counts.values()) for n in (20,50,100,200)},
            non_PD_record_count_upper_bounds={str(n):sum(k>=2*n*50 for k in controls.values()) for n in (20,50,100,200)},
            non_PD_with_at_least_two_observed_dates=sum(len(days[u])>=2 for u in controls),
            caveat='Counts are only availability upper bounds. No adjacency, duplication, chronological split, session-gap, or day-gap validation. Non-PD is self-reported and is not proven cognitive health. No keys were reconstructed; no modeling or clinical evaluation was performed.')
path=Path(__file__).with_name('local_tappy_availability.json')
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
