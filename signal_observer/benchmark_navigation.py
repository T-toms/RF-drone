"""Real-source navigation timings; no Tk control or source writes."""
import argparse,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import analysis,plots

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent;out=root/'output/signal_observer'
    overview=analysis.load_result(out/'demo_full_recording.npz')
    info=analysis.inspect_mat(overview['source_path']);plan=analysis.make_plan(info,info.variables[0],True)
    cache=analysis.IQCache() if hasattr(analysis,'IQCache') else None
    fig=Figure(figsize=(12,6.2),dpi=100);canvas=FigureCanvasAgg(fig);rows=[]
    for start in [0,131072,262144,131072,0]:
        begin=time.perf_counter();kw={'cache':cache} if cache is not None else {}
        result=analysis.analyze(plan,start,262144,2048,hop=512,max_frequency_bins=1024,**kw)
        read=time.perf_counter();plots.draw_result(fig,result,overview=overview);canvas.draw();end=time.perf_counter()
        rows.append(dict(start=start,analysis_ms=(read-begin)*1000,draw_ms=(end-read)*1000,total_ms=(end-begin)*1000))
    fig.savefig(out/(args.label+'_navigation.png'),dpi=120)
    data={'label':args.label,'runs':rows,'warm_median_ms':float(np.median([r['total_ms'] for r in rows[1:]]))}
    if cache is not None:data['iq_cache']={'loads':cache.loads,'hits':cache.hits,'bytes':cache.bytes}
    (out/(args.label+'_navigation_timings.json')).write_text(json.dumps(data,indent=2),encoding='utf8');print(json.dumps(data))
if __name__=='__main__':main()
