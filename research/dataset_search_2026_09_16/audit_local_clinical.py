"""Audit existing clinical sources without fitting or evaluating a classifier."""
from collections import Counter,defaultdict
from pathlib import Path
import csv,io,json,zipfile,datetime,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/clinical'
OUT=Path(__file__).parent
BUDGETS=(5,10,20,50,100,200)

def nq_audit():
    results={}; people=[]; keys=Counter(); mismatches=[]; archive=BASE/'neuroqwerty/neuroQWERTY.zip'
    with zipfile.ZipFile(archive) as z:
        for study in ('MIT-CS1PD','MIT-CS2PD'):
            meta=list(csv.DictReader(io.StringIO(z.read(f'{study}/GT_DataPD_{study}.csv').decode())))
            for person in meta:
                sessions=[]
                for col,name in person.items():
                    if not col.startswith('file_') or not name: continue
                    rows=list(csv.reader(io.StringIO(z.read(f'{study}/data_{study}/{name}').decode())))
                    valid=[]; bad=0
                    for row in rows:
                        try:
                            key=row[0]; hold,release,press=map(float,row[1:])
                            if key.startswith('mouse'): continue
                            if not all(np.isfinite([hold,release,press])) or not (0<hold<5 and release>=press>=0):raise ValueError()
                            if abs(hold-(release-press))>.001: mismatches.append(name)
                            valid.append((press,release,key)); keys[key]+=1
                        except (ValueError,IndexError):bad+=1
                    valid=sorted(set(valid)); run=0; windows=0; last=None
                    for press,release,key in valid:
                        if last is not None and (press-last>30 or press<=last): windows+=run//50;run=0
                        run+=1;last=press
                    windows+=run//50
                    sessions.append(dict(file=name,valid_keys=len(valid),bad_rows=bad,windows_50=windows,
                                         duration_seconds=valid[-1][1]-valid[0][0] if valid else 0))
                people.append(dict(study=study,id=person['pID'],pd=person['gt']=='True',sessions=sessions))
            subset=[r for r in people if r['study']==study]
            results[study]=dict(participants=len(subset),controls=sum(not r['pd'] for r in subset),
                pd=sum(r['pd'] for r in subset),recordings=sum(len(r['sessions']) for r in subset),
                usable_keys=sum(s['valid_keys'] for r in subset for s in r['sessions']))
    controls=[r for r in people if not r['pd']]
    results['control_total_window_upper_bounds']={str(n):sum(sum(s['windows_50'] for s in r['sessions'])>=2*n for r in controls) for n in BUDGETS}
    results['controls_with_separate_recordings_each_supporting_N_windows']={str(n):sum(len(r['sessions'])>=2 and min(s['windows_50'] for s in r['sessions'][:2])>=n for r in controls) for n in BUDGETS}
    results['per_recording_key_quantiles']=np.quantile([s['valid_keys'] for r in people for s in r['sessions']],[0,.25,.5,.75,1]).tolist()
    results['hold_release_minus_press_mismatch_rows']=len(mismatches)
    results['key_labels']=dict(keys)
    results['counting_rule']='Keyboard records with finite consistent timestamps and 0<hold<5s; exact tuples deduplicated; non-overlapping full 50-key windows; break at nonpositive press differences or >30-second gaps. Sessions never stitched.'
    (OUT/'neuroqwerty_participant_audit.json').write_text(json.dumps(people,indent=2)+'\n')
    return results

def writing_audit():
    result={}
    for path,idkey,groupkey in [(BASE/'ad_mci_writing/Participant_level.csv','participant_code','group1'),(BASE/'ad_mci_writing/Linguistic_analysis_word_level.csv','Participant','Group1')]:
        with path.open(encoding='latin1',newline='') as f:rows=list(csv.DictReader(f,delimiter=';'))
        groups=defaultdict(set)
        for r in rows:groups[r[groupkey]].add(r[idkey])
        result[path.name]=dict(rows=len(rows),participants=len({r[idkey] for r in rows}),
                              groups={k:len(v) for k,v in groups.items()},columns=list(rows[0]))
    return result

def tappy_audit():
    users={}
    with zipfile.ZipFile(BASE/'tappy/Archived-users.zip') as z:
        for name in z.namelist():
            if name.endswith('.txt'):users[Path(name).stem.removeprefix('User_')]=dict(line.split(': ',1) for line in z.read(name).decode(errors='replace').splitlines() if ': ' in line)
    stats=Counter(); eligible={t:{str(n):0 for n in BUDGETS} for t in (5,20,50)}; loose={str(n):0 for n in BUDGETS}; people=[]
    datecache={}; byuser=defaultdict(list)
    with zipfile.ZipFile(BASE/'tappy/Archived-Data.zip') as z:
        for name in z.namelist():
            if name.endswith('.txt'):byuser[Path(name).stem.split('_')[0]].append(name)
        for ix,(uid,names) in enumerate(sorted(byuser.items())):
            records=[]
            for name in names:
                with z.open(name) as f:
                    for line in f:
                        stats['raw_rows']+=1
                        try:
                            r=line.decode(errors='replace').strip().split('\t'); h,p,flight=[float(r[j]) for j in (4,6,7)]
                            if r[0]!=uid or r[3] not in ('L','R','S') or not all(np.isfinite([h,p,flight])) or not (h>0 and p>0):raise ValueError()
                            day=r[1]
                            if day not in datecache:datecache[day]=datetime.datetime.strptime(day,'%y%m%d').toordinal()
                            hh,mm,ss=r[2].split(':'); hh=int(hh);mm=int(mm);ss=float(ss)
                            if not (0<=hh<24 and 0<=mm<60 and 0<=ss<60):raise ValueError()
                            ts=datecache[day]*86400000+(hh*3600+mm*60+ss)*1000
                            records.append((ts,day,h,p,flight,r[3],r[5]))
                        except (ValueError,IndexError):stats['invalid_rows']+=1
            distinct=sorted(set(records),key=lambda r:(r[0]-r[2],r[0],r[5]));stats['exact_duplicate_rows']+=len(records)-len(distinct)
            stats['unique_valid_rows']+=len(distinct)
            days=sorted({r[1] for r in distinct}); cut=len(days)//2
            first=set(days[:cut]); rest=set(days[cut:]); control=users.get(uid,{}).get('Parkinsons')=='False'
            daycounts=Counter(r[1] for r in distinct)
            loose_sides=[sum(daycounts[d]//50 for d in group) for group in (first,rest)]
            if control:
                for n in BUDGETS:loose[str(n)]+=int(min(loose_sides)>=n)
            windows={}
            for tolerance in (5,20,50):
                byday=Counter();run=0;prev=None
                for r in distinct:
                    connected=False
                    if prev is not None and r[1]==prev[1]:
                        dt=r[0]-prev[0]
                        closure=abs(r[3]-r[4]-prev[2])
                        # Empirical test of stamp-as-release alignment, not an assumed published guarantee.
                        err_release=abs((dt-r[2]+prev[2])-r[3])
                        err_press=abs(dt-r[3])
                        if tolerance==20:
                            stats['within_day_pairs']+=1
                            stats['hold_flight_identity_within_1ms']+=int(closure<=1)
                            stats['stamp_as_release_within_20ms']+=int(err_release<=20)
                            stats['stamp_as_press_within_20ms']+=int(err_press<=20)
                        connected=(0<dt-r[2]+prev[2]<=30000 and closure<=1 and err_release<=tolerance and r[6]==prev[5]+r[5])
                    if not connected:
                        if prev is not None:byday[prev[1]]+=run//50
                        run=0
                    run+=1;prev=r
                if prev is not None:byday[prev[1]]+=run//50
                sides=[sum(byday[d] for d in group) for group in (first,rest)]
                windows[str(tolerance)]=sides
                if control:
                    for n in BUDGETS:eligible[tolerance][str(n)]+=int(min(sides)>=n)
            people.append(dict(id=uid,non_PD=control,metadata_present=uid in users,unique_valid_records=len(distinct),
                               observed_dates=len(days),date_split_record_count_windows=loose_sides,consistent_run_windows=windows))
            if (ix+1)%50==0:print('TAPPY',ix+1,len(byuser),flush=True)
    (OUT/'tappy_participant_audit.json').write_text(json.dumps(people,indent=2)+'\n')
    return dict(counts=dict(stats),participants=len(people),non_PD=sum(r['non_PD'] for r in people),
                non_PD_date_split_upper_bounds=loose,non_PD_consistent_run_eligibility_by_timestamp_tolerance_ms=eligible,
                rule='Deduplicate; order by inferred press time (record timestamp minus hold); split earliest/later halves of observed dates; full nonoverlapping 50-record windows within day. Strict runs require hand-transition agreement, previous hold = latency-flight within 1ms, timestamp-as-release alignment within 5/20/50ms, and positive <=30s inferred press gaps.',
                caveat='Conservative continuity diagnostic, not a definitive clean-key event reconstruction. Timestamp interpretation is tested empirically. Recording omissions and unavailable exact key identities remain. Non-PD is self-reported, not cognitive-health ground truth.')

if __name__=='__main__':
    result={'neuroqwerty':nq_audit(),'ad_mci_writing':writing_audit(),'tappy':tappy_audit()}
    result['archive_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.rglob('*.zip')}
    (OUT/'local_clinical_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='archive_sha256'},indent=2))
