"""Read two real adjacent MAT segments through the playback traversal, source read-only."""
import json,time
from pathlib import Path
from analysis import load_result,inspect_mat,make_plan,analyze,IQCache,Plan
from navigation import next_play_view

def main():
    root=Path(__file__).resolve().parent.parent;out=root/'output/signal_observer'
    overview=load_result(out/'demo_full_recording.npz');info=inspect_mat(overview['source_path'])
    full=make_plan(info,info.variables[0],True);parts=full.parts[:2];total=parts[-1].offset+parts[-1].length
    plan=Plan(full.title+' / 前两段播放验证',parts,total,'recording',full.source_path)
    bounds=(parts[0].offset,total);width=262144;view=(bounds[0],bounds[0]+width);cache=IQCache();records=[];begin=time.perf_counter()
    while view is not None:
        result=analyze(plan,view[0],view[1]-view[0],2048,hop=512,max_frequency_bins=1024,allow_empty=True,cache=cache)
        assert (result['start'],result['stop'])==view
        records.append({'view':view,'frames':result['frames'],'source_count':len(result['sources'])})
        view=next_play_view(bounds,view,.5)
    assert records[0]['view'][0]==bounds[0] and records[-1]['view'][1]==bounds[1]
    assert all(b['view'][0]<=a['view'][1] and b['view'][0]>a['view'][0] for a,b in zip(records,records[1:]))
    assert any(r['source_count']==2 for r in records)
    data=dict(passed=True,bounds=bounds,view_samples=width,step_fraction=.5,views=len(records),cross_file_views=sum(r['source_count']==2 for r in records),elapsed_seconds=time.perf_counter()-begin,iq_cache_loads=cache.loads,iq_cache_hits=cache.hits,source_paths=[p.path for p in parts],first_view=records[0],last_view=records[-1],note='Numerical traversal across two adjacent real MAT segments, not a native GUI frame-rate measurement.')
    (out/'real_playback_verification.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(data,ensure_ascii=False))
if __name__=='__main__':main()
