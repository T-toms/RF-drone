"""Exercise the actual playback event flow with a deterministic timer, no native UI."""
import heapq,tempfile,threading,queue,unittest
from collections import OrderedDict,deque
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
from analysis import inspect_mat,make_plan,analyze,IQCache,Part,Plan
from local_observer import Observer
from navigation import next_play_view

class Value:
    def __init__(self,value=None):self.value=value
    def get(self):return self.value
    def set(self,value):self.value=value
class Widget(dict):
    def configure(self,**kwargs):self.update(kwargs)
class Timer:
    def __init__(self):self.now=0;self.serial=0;self.tasks=[];self.cancelled=set()
    def after(self,delay,callback):
        self.serial+=1;heapq.heappush(self.tasks,(self.now+delay,self.serial,callback));return self.serial
    def after_cancel(self,key):self.cancelled.add(key)
    def step(self):
        when,key,callback=heapq.heappop(self.tasks);self.now=when
        if key not in self.cancelled:callback()
class Future:
    def __init__(self,worker):self.worker=worker;self.cancelled=False
    def cancel(self):self.cancelled=True
class Executor:
    def __init__(self):self.tasks=deque()
    def submit(self,worker):
        future=Future(worker);self.tasks.append(future);return future
    def work(self):
        if self.tasks:
            future=self.tasks.popleft()
            if not future.cancelled:future.worker()

class PlaybackChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'iq.mat'
        savemat(self.path,{'data':np.exp(2j*np.pi*.125*np.arange(32768)).astype(np.complex64)})
        self.plan=make_plan(inspect_mat(self.path),'data',audit=Path(self.tmp.name))
        self.o=Observer.__new__(Observer);o=self.o;o.root=Timer();o.executor=Executor();o.events=queue.Queue();o.cancel=threading.Event()
        o.overview=analyze(self.plan,nfft=512);o.plan=self.plan;o.result=analyze(self.plan,0,4096,512,hop=128);o.view=(0,4096)
        o.busy=False;o.active_mode=None;o.pending_view=None;o.pending_auto=False;o.view_job=None;o.view_guard=False;o.slide_preview=False
        o.job_id=0;o.suppressed_job=None;o.compute_key=None;o.iq_cache=IQCache();o.detail_cache=OrderedDict()
        o.playing=False;o.play_job=None;o.play_target=None;o.play_key=None;o.play_due=0;o.play_frame_job=None
        o.prefetch_cancel=threading.Event();o.prefetch_future=None;o.prefetch_generation=0
        for name in ['run_button','browse_button','cancel_button','progress','play_button']:setattr(o,name,Widget())
        for name,value in dict(fft='512',play_rate='5步/秒',play_step='1/2视窗',fs='',fc='',view_width='4096',position=0,status='',play_state='',view_label='').items():setattr(o,name,Value(value))
        o.file_fs=None;self.frames=[];o.render=lambda:self.frames.append((o.result['start'],o.result['stop'],o.result['frames']))
        o.root.after(25,o.poll);self.clock=patch('local_observer.time.perf_counter',lambda:o.root.now/1000);self.clock.start()
    def tearDown(self):self.clock.stop();self.tmp.cleanup()
    def drive(self,predicate,work=True,limit=10000):
        for _ in range(limit):
            if predicate():return
            if work:self.o.executor.work()
            self.o.root.step()
        self.fail('Playback did not reach its expected state')
    def test_complete_playback_covers_entire_range_and_stops_at_tail(self):
        o=self.o;o.toggle_play();self.drive(lambda:not o.playing)
        views=[(0,4096)]+[(a,b) for a,b,_ in self.frames]
        self.assertEqual(views[-1][1],32768);self.assertEqual(len(views),15)
        for before,after in zip(views,views[1:]):self.assertGreater(after[0],before[0]);self.assertLessEqual(after[0],before[1])
        self.assertIn('末尾',o.play_state.get());self.assertEqual(o.play_button['text'],'▶ 播放')
    def test_pause_discards_a_play_frame_already_queued_for_display(self):
        o=self.o;o.toggle_play();self.drive(lambda:o.busy and o.play_frame_job==o.job_id)
        o.pause_playback();now=o.root.now;self.drive(lambda:o.root.now>now+1000)
        self.assertFalse(o.playing);self.assertEqual(self.frames,[]);self.assertEqual(o.view,(0,4096))
    def test_slow_prefetch_waits_instead_of_skipping_windows(self):
        o=self.o;o.toggle_play();self.drive(lambda:o.root.now>=500,work=False)
        self.assertEqual(self.frames,[]);self.assertIn('缓冲',o.play_state.get())
        self.drive(lambda:len(self.frames)>0);o.pause_playback()
        self.assertEqual(self.frames[0][:2],(2048,6144))
    def test_manual_seek_pauses_and_rejects_stale_prefetch(self):
        o=self.o;o.toggle_play();o.executor.work();o.request_view(16384,4096)
        self.drive(lambda:len(self.frames)>0);now=o.root.now;self.drive(lambda:o.root.now>now+500)
        self.assertFalse(o.playing);self.assertEqual([(a,b) for a,b,_ in self.frames],[(16384,20480)])
    def test_stop_returns_to_beginning(self):
        o=self.o;o.toggle_play();self.drive(lambda:len(self.frames)>=3);o.stop_playback()
        self.drive(lambda:not o.busy and o.pending_view is None and o.result['start']==0)
        self.assertEqual(self.frames[-1][:2],(0,4096));self.assertFalse(o.playing)
    def test_play_again_after_end_restarts_from_beginning(self):
        o=self.o;o.toggle_play();self.drive(lambda:not o.playing);self.frames.clear();o.toggle_play()
        self.drive(lambda:len(self.frames)>0);o.pause_playback();self.assertEqual(self.frames[0][:2],(0,4096))
    def test_missing_source_interval_is_shown_as_blank_and_not_skipped(self):
        o=self.o;p=self.plan.parts[0]
        o.plan=Plan('gap',[Part(p.path,'data',p.length,0,0,8192),Part(p.path,'data',p.length,16384,0,8192)],24576,'test',p.path)
        o.overview=analyze(o.plan,nfft=512);o.toggle_play();self.drive(lambda:not o.playing)
        self.assertTrue(any(a>=8192 and b<=16384 and frames==0 for a,b,frames in self.frames));self.assertEqual(self.frames[-1][1],24576)
    def test_prefetch_error_pauses_instead_of_leaving_timer_running(self):
        o=self.o
        with patch('local_observer.analyze',side_effect=OSError('read failed')):
            o.toggle_play();self.drive(lambda:not o.playing)
        self.assertIn('read failed',o.status.get());self.assertEqual(self.frames,[]);self.assertIsNone(o.play_job)
    def test_play_options_change_keeps_order(self):
        o=self.o;o.toggle_play();self.drive(lambda:len(self.frames)>=2)
        last=o.view;o.play_rate.set('10步/秒');o.play_step.set('1/4视窗');o.play_options_changed()
        self.drive(lambda:len(self.frames)>=3);o.pause_playback();self.assertEqual(self.frames[2][0],last[0]+1024)
    def test_nondivisible_tail_keeps_width_and_reaches_exact_end(self):
        bounds=(37,100000);view=(37,37+4096);last=view
        while (target:=next_play_view(bounds,last,.5)) is not None:
            self.assertEqual(target[1]-target[0],4096);self.assertLessEqual(target[0],last[1]);last=target
        self.assertEqual(last[1],100000)
        with self.assertRaises(ValueError):next_play_view(bounds,view,1.1)

if __name__=='__main__':unittest.main(verbosity=2)
