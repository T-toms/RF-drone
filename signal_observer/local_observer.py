"""Local Python/Tk MAT observer. No web server, installer, or source mutation."""
from pathlib import Path
import argparse,datetime,json,math,os,queue,sys,threading,traceback,time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from analysis import inspect_mat,make_plan,analyze,save_result,load_result,Cancelled,WORKSPACE,IQCache
from plots import draw_result,time_scale,configure_font
from navigation import viewport,zoom_view,next_play_view

OUTPUT=WORKSPACE/'output/signal_observer'

class Observer:
    def __init__(self,root,args):
        configure_font()
        self.root=root;self.info=None;self.result=None;self.axes=[];self.busy=False
        self.events=queue.Queue();self.cancel=threading.Event();self.fs_source='unknown';self.file_fs=None
        self.overview=None;self.plan=None;self.pending_view=None;self.view_job=None;self.view_guard=False
        self.view=None;self.job_id=0;self.active_mode=None;self.iq_cache=IQCache();self.detail_cache=OrderedDict();self.compute_key=None;self.slide_preview=False
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='iq-observer')
        self.playing=False;self.play_job=None;self.play_target=None;self.play_key=None;self.play_due=0.;self.play_frame_job=None;self.suppressed_job=None;self.pending_auto=False
        self.prefetch_cancel=threading.Event();self.prefetch_future=None;self.prefetch_generation=0
        root.title('本地 MAT 信号观察脚本 · 连续播放版');root.geometry('1280x940');root.minsize(1000,720)
        style=ttk.Style(root);style.configure('Navigation.TButton',font=('Microsoft YaHei',10,'bold'),padding=(9,5))
        self.path=tk.StringVar();self.variable=tk.StringVar();self.scope=tk.StringVar(value='仅当前MAT');self.fs=tk.StringVar();self.fc=tk.StringVar()
        self.start=tk.StringVar(value='0');self.count=tk.StringVar();self.fft=tk.StringVar(value='2048');self.exclude=tk.BooleanVar(value=True)
        self.view_width=tk.StringVar(value='262144');self.position=tk.DoubleVar(value=0)
        self.detail_kind=tk.StringVar(value='平均谱')
        self.play_rate=tk.StringVar(value='5步/秒');self.play_step=tk.StringVar(value='1/2视窗');self.play_state=tk.StringVar(value='已暂停；从当前位置开始播放')
        self.info_text=tk.StringVar(value='选择一个MAT。整文件可浏览，主图默认显示清晰局部；精确中心频率可留空。')
        self.status=tk.StringVar(value='就绪。用“向前/向后”按钮或时间滑块浏览；拖动滑块松开后更新。')
        top=ttk.Frame(root,padding=8);top.pack(fill='x')
        row=ttk.Frame(top);row.pack(fill='x',pady=2)
        ttk.Label(row,text='MAT文件').pack(side='left');ttk.Entry(row,textvariable=self.path).pack(side='left',fill='x',expand=True,padx=5)
        self.browse_button=ttk.Button(row,text='选择文件…',command=self.browse);self.browse_button.pack(side='left')
        row=ttk.Frame(top);row.pack(fill='x',pady=2)
        ttk.Label(row,text='IQ变量').pack(side='left');self.variables=ttk.Combobox(row,textvariable=self.variable,width=13,state='readonly');self.variables.pack(side='left',padx=5)
        ttk.Label(row,text='观察范围').pack(side='left');ttk.Combobox(row,textvariable=self.scope,values=['仅当前MAT','整次录制（按已审计索引）'],width=28,state='readonly').pack(side='left',padx=5)
        ttk.Checkbutton(row,text='应用已核实的无效零区间标记',variable=self.exclude).pack(side='left',padx=8)
        row=ttk.Frame(top);row.pack(fill='x',pady=2)
        ttk.Label(row,text='采样率 MS/s（可空）').pack(side='left');ttk.Entry(row,textvariable=self.fs,width=12).pack(side='left',padx=4)
        ttk.Button(row,text='填候选122.88',command=self.candidate).pack(side='left')
        ttk.Label(row,text='接收中心 MHz（可空）').pack(side='left',padx=(14,0));ttk.Entry(row,textvariable=self.fc,width=12).pack(side='left',padx=4)
        ttk.Label(row,text='FFT点数').pack(side='left',padx=(14,0));ttk.Combobox(row,textvariable=self.fft,values=['512','1024','2048','4096','8192','16384'],width=8,state='readonly').pack(side='left',padx=4)
        ttk.Label(row,text='细节显示').pack(side='left',padx=(12,0));kindbox=ttk.Combobox(row,textvariable=self.detail_kind,values=['平均谱','峰值谱'],width=8,state='readonly');kindbox.pack(side='left',padx=4);kindbox.bind('<<ComboboxSelected>>',lambda e:self.render())
        row=ttk.Frame(top);row.pack(fill='x',pady=2)
        ttk.Label(row,text='起始采样点').pack(side='left');ttk.Entry(row,textvariable=self.start,width=15).pack(side='left',padx=4)
        ttk.Label(row,text='样点数（空=到末尾）').pack(side='left');ttk.Entry(row,textvariable=self.count,width=15).pack(side='left',padx=4)
        ttk.Button(row,text='全部',command=self.full).pack(side='left',padx=3)
        for label,seconds in [('20 ms',.02),('100 ms',.1),('1秒',1)]:ttk.Button(row,text=label,command=lambda s=seconds:self.duration(s),width=7).pack(side='left',padx=2)
        row=ttk.Frame(top);row.pack(fill='x',pady=3)
        self.run_button=ttk.Button(row,text='生成 / 更新图',command=self.run);self.run_button.pack(side='left')
        self.cancel_button=ttk.Button(row,text='取消计算',command=self.cancel_work,state='disabled');self.cancel_button.pack(side='left',padx=4)
        ttk.Button(row,text='只更新坐标',command=self.render).pack(side='left',padx=3)
        ttk.Button(row,text='保存PNG与参数',command=self.export).pack(side='left',padx=3)
        ttk.Button(row,text='打开输出目录',command=self.open_output).pack(side='left',padx=3)
        self.progress=ttk.Progressbar(row,maximum=100,length=180);self.progress.pack(side='left',fill='x',expand=True,padx=10)
        ttk.Label(top,textvariable=self.info_text,foreground='#235789',wraplength=1200).pack(anchor='w',pady=2)
        row=ttk.Frame(top);row.pack(fill='x',pady=3)
        ttk.Label(row,text='局部视窗样点数').pack(side='left')
        widthbox=ttk.Combobox(row,textvariable=self.view_width,values=['262144','1048576','2457600','6144000'],width=13)
        widthbox.pack(side='left',padx=4);widthbox.bind('<<ComboboxSelected>>',lambda e:self.change_width());widthbox.bind('<Return>',lambda e:self.change_width())
        ttk.Button(row,text='应用宽度',command=self.change_width).pack(side='left')
        ttk.Button(row,text='20ms视窗',command=lambda:self.change_width(seconds=.02)).pack(side='left',padx=3)
        ttk.Button(row,text='放大 ×2',command=lambda:self.scale_view(.5)).pack(side='left',padx=3)
        ttk.Button(row,text='缩小 ×2',command=lambda:self.scale_view(2)).pack(side='left',padx=3)
        ttk.Button(row,text='查看全段',command=self.show_overview).pack(side='left',padx=3)
        row=ttk.Frame(top);row.pack(fill='x',pady=2)
        self.play_button=ttk.Button(row,text='▶ 播放',style='Navigation.TButton',command=self.toggle_play);self.play_button.pack(side='left',padx=3)
        ttk.Button(row,text='■ 停止并回开头',command=self.stop_playback).pack(side='left',padx=4)
        ttk.Label(row,text='浏览速度上限').pack(side='left',padx=(10,3));ratebox=ttk.Combobox(row,textvariable=self.play_rate,values=['2步/秒','5步/秒','10步/秒'],state='readonly',width=9);ratebox.pack(side='left');ratebox.bind('<<ComboboxSelected>>',self.play_options_changed)
        ttk.Label(row,text='每步前进').pack(side='left',padx=(10,3));stepbox=ttk.Combobox(row,textvariable=self.play_step,values=['1/4视窗','1/2视窗','1视窗'],state='readonly',width=9);stepbox.pack(side='left');stepbox.bind('<<ComboboxSelected>>',self.play_options_changed)
        ttk.Label(row,textvariable=self.play_state,foreground='#235789').pack(side='left',padx=12)
        row=ttk.Frame(top);row.pack(fill='x',pady=2)
        ttk.Button(row,text='|◀ 开头',style='Navigation.TButton',command=lambda:self.edge_view(False)).pack(side='left',padx=3)
        ttk.Button(row,text='◀ 向前移动',style='Navigation.TButton',command=lambda:self.step_view(-.5)).pack(side='left',padx=3)
        ttk.Button(row,text='向后移动 ▶',style='Navigation.TButton',command=lambda:self.step_view(.5)).pack(side='left',padx=3)
        ttk.Button(row,text='末尾 ▶|',style='Navigation.TButton',command=lambda:self.edge_view(True)).pack(side='left',padx=3)
        ttk.Label(row,text='时间位置（拖此滑块，松开更新）').pack(side='left',padx=(10,2))
        self.slider=tk.Scale(row,from_=0,to=1,resolution=.000001,showvalue=False,orient='horizontal',variable=self.position,command=self.slide,sliderlength=44,width=17,highlightthickness=1,troughcolor='#cbddea',activebackground='#377eaf')
        self.slider.pack(side='left',fill='x',expand=True,padx=6)
        self.slider.bind('<ButtonPress-1>',self.begin_slide)
        self.slider.bind('<ButtonRelease-1>',self.commit_slide);self.slider.bind('<KeyRelease>',self.commit_slide)
        self.view_label=tk.StringVar(value='主图不需要拖动；使用上方四个移动按钮或时间位置滑块。')
        ttk.Label(top,textvariable=self.view_label,foreground='#235789').pack(anchor='w')
        ttk.Label(root,textvariable=self.status,anchor='w',padding=(8,2),wraplength=1200).pack(fill='x',side='bottom')
        self.figure=Figure(figsize=(11,7),dpi=100);self.canvas=FigureCanvasTkAgg(self.figure,master=root)
        self.canvas.get_tk_widget().pack(fill='both',expand=True)
        self.figure.text(.5,.55,'先选择MAT，再点击“生成 / 更新图”\n主图从清晰局部开始；用移动按钮/时间滑块浏览整段。',ha='center',fontsize=13)
        configure_font();self.canvas.draw_idle();root.after(100,self.poll);root.protocol('WM_DELETE_WINDOW',self.close)
        if args.file:root.after(200,lambda:self.set_file(args.file))
        if args.scope=='recording':self.scope.set('整次录制（按已审计索引）')
        if args.load:
            self.result=load_result(args.load);self.path.set(self.result['source_path'])
            self.info=inspect_mat(self.path.get());self.variables['values']=self.info.variables;self.variable.set(self.info.variables[0])
            self.scope.set('整次录制（按已审计索引）' if self.result['scope']=='recording' else '仅当前MAT')
            if self.result.get('fs_hz'):self.fs.set(f"{self.result['fs_hz']/1e6:g}")
            if self.result.get('fc_hz') is not None:self.fc.set(f"{self.result['fc_hz']/1e6:g}")
            self.fs_source=self.result.get('fs_source','unknown');self.file_fs=self.info.fs_hz
            self.overview=self.result
            self.plan=make_plan(self.info,self.variable.get(),self.scope.get().startswith('整次'),self.exclude.get())
            root.after(250,self.initial_view)

    def fail(self,exc):
        self.status.set(str(exc));messagebox.showerror('暂时无法绘图',str(exc),parent=self.root)
    def candidate(self):
        self.fs.set('122.88');self.fs_source='assumed';self.info_text.set('已填入候选122.88 MS/s，时间坐标按推定值换算；尚未确认为整批数据的采集参数。')
    def browse(self):
        path=filedialog.askopenfilename(title='选择MAT文件',filetypes=[('MATLAB文件','*.mat'),('所有文件','*.*')],initialdir=str(Path(self.path.get()).parent) if self.path.get() else 'G:/drone/data')
        if path:self.set_file(path)
    def set_file(self,path):
        if self.busy:return
        self.pause_playback(cancel_frame=False)
        self.info=None
        try:
            self.info=inspect_mat(path);self.path.set(self.info.path);self.variables['values']=self.info.variables;self.variable.set(self.info.variables[0])
            self.overview=None;self.plan=None;self.pending_view=None;self.view=None;self.result=None
            self.iq_cache=IQCache();self.detail_cache.clear();self.figure._observer_live=None
            if self.view_job:self.root.after_cancel(self.view_job);self.view_job=None
            self.fs.set('' if self.info.fs_hz is None else f'{self.info.fs_hz/1e6:g}');self.fc.set('' if self.info.fc_hz is None else f'{self.info.fc_hz/1e6:g}')
            self.file_fs=self.info.fs_hz;self.fs_source='file' if self.file_fs else 'unknown';self.start.set('0');self.count.set('')
            n=self.info.lengths[self.variable.get()];self.info_text.set(f'当前MAT：{n:,}个样点。'+('采样率来自文件。' if self.file_fs else '文件未记录Fs；可留空用样点观察，或明确填入估计值。'))
            self.status.set('文件已选择；点击“生成 / 更新图”。')
            return True
        except Exception as exc:self.fail(exc);return False
    def parameters(self):
        fs=float(self.fs.get())*1e6 if self.fs.get().strip() else None
        fc=float(self.fc.get())*1e6 if self.fc.get().strip() else None
        if fs is not None and (not math.isfinite(fs) or fs<=0):raise ValueError('采样率必须为正数，单位MS/s；未知时留空。')
        if fc is not None and (not math.isfinite(fc) or fc<=0):raise ValueError('接收中心必须为正数，单位MHz；未知时留空。')
        if fc is not None and fs is None:raise ValueError('换算绝对频率需要Fs；请先填Fs，或暂时将接收中心留空。')
        source='file' if fs is not None and self.file_fs is not None and math.isclose(fs,self.file_fs,rel_tol=1e-12) else 'assumed' if fs else 'unknown'
        return fs,fc,source
    def full(self):self.start.set('0');self.count.set('')
    def duration(self,seconds):
        try:
            fs,_,_=self.parameters()
            if fs is None:raise ValueError('先填写采样率，才能按毫秒选范围；未知时可直接输入样点数。')
            self.count.set(str(round(fs*seconds)))
        except Exception as exc:self.fail(exc)
    def run(self):
        if self.busy:return
        self.pause_playback(cancel_frame=False)
        try:
            self.parameters()
            if self.info is None or str(Path(self.path.get()).resolve())!=self.info.path:
                if not self.set_file(self.path.get()):return
            if self.info is None:raise ValueError('请先选择MAT文件。')
            plan=make_plan(self.info,self.variable.get(),self.scope.get().startswith('整次'),self.exclude.get());self.plan=plan;self.pending_view=None;self.detail_cache.clear()
            start=int(self.start.get());count=int(self.count.get()) if self.count.get().strip() else None;nfft=int(self.fft.get())
            self.compute(plan,start,count,nfft,'overview')
        except Exception as exc:self.fail(exc)
    def compute(self,plan,start,count,nfft,mode):
        self.busy=True;self.active_mode=mode;self.cancel.clear();self.job_id+=1;job=self.job_id
        self.run_button['state']='disabled';self.browse_button['state']='disabled';self.cancel_button['state']='normal';self.progress['value']=0
        self.status.set('正在读取IQ并计算'+('整段总览' if mode=='overview' else '局部细节')+'；输入文件只读。')
        self.compute_key=None
        self.play_frame_job=job if self.pending_auto and mode=='detail' else None
        self.pending_auto=False
        if mode=='detail':
            try:self.compute_key=self.result_key(plan,start,count,nfft)
            except Exception as exc:self.events.put(('error',job,mode,str(exc),traceback.format_exc()));return
            if self.compute_key in self.detail_cache:
                result=self.detail_cache[self.compute_key];self.detail_cache.move_to_end(self.compute_key);self.events.put(('result',job,mode,result));return
        cache=self.iq_cache
        def worker():
            try:
                result=analyze(plan,start,count,nfft,progress=lambda a,b,msg:self.events.put(('progress',job,a,b,msg)),cancel=self.cancel,allow_empty=mode=='detail',hop=nfft//4 if mode=='detail' else nfft,max_frequency_bins=1024 if mode=='detail' else 512,cache=cache)
                self.events.put(('result',job,mode,result))
            except Cancelled:self.events.put(('cancelled',job,mode))
            except Exception as exc:self.events.put(('error',job,mode,str(exc),traceback.format_exc()))
        self.executor.submit(worker)
    def poll(self):
        try:
            while True:
                item=self.events.get_nowait()
                if item[0].startswith('prefetch'):
                    self.receive_prefetch(item);continue
                if item[1]!=self.job_id:continue
                if item[0]=='progress':
                    _,_,a,b,msg=item;self.progress['value']=a/max(b,1)*100;self.status.set(f'读取/计算 {a}/{b}：{msg}')
                else:
                    self.busy=False;self.run_button['state']='normal';self.browse_button['state']='normal';self.cancel_button['state']='disabled'
                    if item[0]=='result' and item[2]=='detail' and self.compute_key is not None:
                        self.remember_result(self.compute_key,item[3])
                    if self.pending_view is not None:
                        self.flush_view();continue
                    if item[1]==self.suppressed_job:continue
                    if item[0]=='result':
                        self.result=item[3]
                        if item[2]=='overview':self.overview=self.result;self.initial_view()
                        else:
                            self.view=(self.result['start'],self.result['stop']);self.render()
                            if self.playing:self.play_due=time.perf_counter()+self.play_interval();self.prepare_playback()
                    elif item[0]=='cancelled':self.status.set('已取消；原始IQ未更改。')
                    elif item[0]=='error':
                        self.pause_playback(cancel_frame=False)
                        OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/'last_error.txt').write_text(item[4],encoding='utf8');self.status.set(item[3]+'；可继续移动到其他区间。')
        except queue.Empty:pass
        self.root.after(25,self.poll)
    def render(self):
        if self.result is None or self.slide_preview:return
        try:
            fs,fc,source=self.parameters()
            self.axes,self.scale=draw_result(self.figure,self.result,fs,fc,source,overview=self.overview,detail_kind='mean_db' if self.detail_kind.get()=='平均谱' else 'peak_db')
            self.canvas.draw_idle();r=self.result;n=r['stop']-r['start'];clock=f'约{n/fs:.6g}秒（'+('文件Fs' if source=='file' else '输入/推定Fs')+'）' if fs else 'Fs未知，时间以样点表示'
            clipped='；请求范围超出源末尾，已取到末尾，未补零' if r['requested_count'] and r['start']+r['requested_count']>r['total_samples'] else ''
            valid=r.get('analyzed_samples',r['frames']*r['nfft'])
            self.info_text.set(f'当前视窗 [{r["start"]:,}, {r["stop"]:,})，跨度{n:,}点；覆盖IQ {valid:,}点；{clock}；使用{len(r["sources"])}个源文件。'+clipped)
            self.sync_navigation();self.status.set('用“向前/向后”移动，或拖时间滑块后松开；放大/缩小请点按钮。当前附近IQ和最近视窗已缓存。')
        except Exception as exc:self.fail(exc)
    def bounds(self):return (self.overview['start'],self.overview['stop'])
    def initial_view(self):
        # Never flash a compressed whole-recording spectrogram on first display.
        width=max(int(self.fft.get())*2,int(self.view_width.get()))
        self.result=None;self.axes=[];self.figure.clear();self.figure._observer_live=None
        self.figure.text(.5,.55,'正在准备清晰的局部视窗…\n整段时间可在图生成后拖动浏览。',ha='center',fontsize=13);self.canvas.draw_idle()
        self.request_view(self.overview['start'],width)
    def sync_navigation(self):
        if self.overview is None:return
        a,b=self.view or (self.result['start'],self.result['stop']);lo,hi=self.bounds();self.view_guard=True
        self.position.set((a-lo)/max(1,hi-lo-(b-a)));self.view_width.set(str(b-a));self.view_guard=False
        fs,_,_=self.parameters()
        self.view_label.set((f'{a/fs*1000:.3f}–{b/fs*1000:.3f} ms（输入Fs）' if fs else f'{a:,}–{b:,} 点')+f' / 全段{hi-lo:,}点')
    def request_view(self,start,width,automatic=False):
        if self.overview is None or self.plan is None:return
        if self.busy and self.active_mode=='overview':return
        if not automatic:self.pause_playback(cancel_frame=False)
        self.view=viewport(self.bounds(),start,width,max(32,int(self.fft.get())*2));self.pending_view=self.view
        self.pending_auto=automatic
        self.sync_navigation()
        if self.view_job:self.root.after_cancel(self.view_job)
        self.view_job=self.root.after(30,self.flush_view)
        if self.busy:self.cancel.set()
    def flush_view(self):
        self.view_job=None
        if self.busy or self.pending_view is None:return
        a,b=self.pending_view;self.pending_view=None
        if (a,b)==self.bounds() and int(self.fft.get())==self.overview['nfft']:
            self.result=self.overview;self.render()
            if self.playing:self.finish_playback()
            return
        self.compute(self.plan,a,b-a,int(self.fft.get()),'detail')
    def slide(self,value):
        if self.view_guard or not self.slide_preview or self.overview is None or self.view is None:return
        lo,hi=self.bounds();width=self.view[1]-self.view[0]
        a=round(lo+float(value)*max(0,hi-lo-width));b=a+width
        self.view_label.set(f'待定位：{a:,}–{b:,} 点；松开滑块更新（拖动过程不计算、不重画）。')
    def begin_slide(self,event=None):
        self.pause_playback();self.slide_preview=True
    def commit_slide(self,event=None):
        self.slide_preview=False
        if self.overview is None or self.view is None:return
        lo,hi=self.bounds();width=self.view[1]-self.view[0]
        self.request_view(lo+self.position.get()*max(0,hi-lo-width),width)
    def edge_view(self,last):
        if self.view:
            lo,hi=self.bounds();width=self.view[1]-self.view[0];self.request_view(hi-width if last else lo,width)
    def change_width(self,seconds=None):
        if self.overview is None:return
        try:
            fs,_,_=self.parameters()
            if seconds and fs is None:raise ValueError('按毫秒设视窗需要Fs；未知时请填写视窗样点数。')
            width=round(fs*seconds) if seconds else int(self.view_width.get())
            if width<=0:raise ValueError('视窗样点数必须为正数。')
            self.request_view((self.view or self.bounds())[0],width)
        except Exception as exc:self.fail(exc)
    def step_view(self,factor):
        if self.view:self.request_view(self.view[0]+(self.view[1]-self.view[0])*factor,self.view[1]-self.view[0])
    def scale_view(self,factor,anchor=None):
        if self.view:
            a,b=zoom_view(self.bounds(),self.view,factor,anchor,int(self.fft.get())*2);self.request_view(a,b-a)
    def show_overview(self):
        if self.overview is not None:self.request_view(self.overview['start'],self.overview['stop']-self.overview['start'])
    def export(self):
        if self.result is None:return
        if self.busy or self.pending_view is not None or self.slide_preview:self.status.set('请等当前视窗计算完成再保存，以保证图片与参数一致。');return
        try:
            fs,fc,source=self.parameters();OUTPUT.mkdir(parents=True,exist_ok=True)
            stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');p=OUTPUT/('signal_'+stamp)
            self.figure.savefig(p.with_suffix('.png'),dpi=150)
            save_result(self.result,p.with_suffix('.npz'),fs,fc,source)
            self.status.set('已保存PNG、参数JSON和数值统计：'+str(p))
        except Exception as exc:self.fail(exc)
    def open_output(self):OUTPUT.mkdir(parents=True,exist_ok=True);os.startfile(OUTPUT)
    def cancel_work(self):
        self.pause_playback(cancel_frame=False)
        self.pending_view=None;self.cancel.set()
        if self.view_job:self.root.after_cancel(self.view_job);self.view_job=None
        if self.result is not None:self.view=(self.result['start'],self.result['stop']);self.sync_navigation()
    def result_key(self,plan,start,count,nfft):
        files=[]
        for p in plan.parts:
            if p.offset+p.valid_stop>start and p.offset+p.valid_start<start+count:
                stat=Path(p.path).stat();files.append((p.path,p.variable,p.valid_start,p.valid_stop,stat.st_size,stat.st_mtime_ns))
        return (start,count,nfft,tuple(files))
    def remember_result(self,key,result):
        self.detail_cache[key]=result;self.detail_cache.move_to_end(key)
        while len(self.detail_cache)>8:self.detail_cache.popitem(last=False)
    def play_interval(self):return 1/{'2步/秒':2,'5步/秒':5,'10步/秒':10}[self.play_rate.get()]
    def play_fraction(self):return {'1/4视窗':.25,'1/2视窗':.5,'1视窗':1}[self.play_step.get()]
    def cancel_prefetch(self):
        self.prefetch_generation+=1;self.prefetch_cancel.set()
        if self.prefetch_future:self.prefetch_future.cancel()
        self.prefetch_future=None
    def pause_playback(self,cancel_frame=True):
        self.playing=False;self.play_button.configure(text='▶ 播放');self.play_state.set('已暂停；从当前位置开始播放')
        if self.play_job:self.root.after_cancel(self.play_job);self.play_job=None
        self.cancel_prefetch();self.play_target=None;self.play_key=None
        if cancel_frame and self.pending_auto:
            self.pending_view=None;self.pending_auto=False
            if self.view_job:self.root.after_cancel(self.view_job);self.view_job=None
        if cancel_frame and self.busy and self.play_frame_job==self.job_id:
            self.suppressed_job=self.job_id;self.cancel.set()
        if cancel_frame and self.result is not None:
            self.view=(self.result['start'],self.result['stop']);self.sync_navigation()
    def toggle_play(self):
        if self.playing:self.pause_playback();return
        if self.result is None or self.busy or self.pending_view is not None:
            self.status.set('先等待图生成完成，再点击播放。');return
        self.playing=True;self.play_button.configure(text='Ⅱ 暂停');self.play_state.set('播放中：按视窗顺序浏览，不跳过区间')
        self.play_due=time.perf_counter()+self.play_interval()
        if self.view[1]>=self.bounds()[1]:self.request_view(self.bounds()[0],self.view[1]-self.view[0],automatic=True)
        else:self.prepare_playback()
    def finish_playback(self):
        self.pause_playback(cancel_frame=False);self.play_state.set('已播放到末尾；再次播放将从开头开始')
    def stop_playback(self):
        self.pause_playback()
        if self.view:self.edge_view(False)
        self.play_state.set('已停止，回到开头')
    def play_options_changed(self,event=None):
        if not self.playing:return
        if self.play_job:self.root.after_cancel(self.play_job);self.play_job=None
        self.cancel_prefetch();self.play_target=None;self.play_key=None
        self.play_due=time.perf_counter()+self.play_interval()
        if not self.busy and self.pending_view is None:self.prepare_playback()
    def prepare_playback(self):
        if not self.playing or self.busy or self.pending_view is not None:return
        target=next_play_view(self.bounds(),self.view,self.play_fraction())
        if target is None:self.finish_playback();return
        self.play_target=target
        try:self.play_key=self.result_key(self.plan,target[0],target[1]-target[0],int(self.fft.get()))
        except Exception as exc:self.pause_playback();self.status.set(str(exc));return
        if self.play_key not in self.detail_cache:
            self.cancel_prefetch();generation=self.prefetch_generation;self.prefetch_cancel=threading.Event();cancel=self.prefetch_cancel
            plan=self.plan;key=self.play_key;cache=self.iq_cache;nfft=int(self.fft.get())
            def worker():
                try:
                    result=analyze(plan,target[0],target[1]-target[0],nfft,hop=nfft//4,max_frequency_bins=1024,allow_empty=True,cancel=cancel,cache=cache)
                    self.events.put(('prefetch_result',generation,key,result))
                except Cancelled:pass
                except Exception as exc:self.events.put(('prefetch_error',generation,str(exc)))
            self.prefetch_future=self.executor.submit(worker)
        if self.play_job:self.root.after_cancel(self.play_job)
        self.play_job=self.root.after(max(1,round((self.play_due-time.perf_counter())*1000)),self.play_tick)
    def receive_prefetch(self,item):
        if item[1]!=self.prefetch_generation:return
        self.prefetch_future=None
        if item[0]=='prefetch_error':self.pause_playback();self.status.set('播放读取失败：'+item[2]);return
        self.remember_result(item[2],item[3])
        if self.playing and item[2]==self.play_key and time.perf_counter()>=self.play_due:self.play_tick()
    def play_tick(self):
        if self.play_job:self.root.after_cancel(self.play_job);self.play_job=None
        if not self.playing:return
        if self.busy or self.pending_view is not None:
            self.play_job=self.root.after(25,self.play_tick);return
        if self.play_target is None:self.prepare_playback();return
        if self.play_key not in self.detail_cache:
            self.play_state.set('播放缓冲中…保留下一视窗，不跳过信号');return
        target=self.play_target;self.play_target=None;self.play_key=None
        self.play_state.set('播放中：按视窗顺序浏览，不跳过区间')
        self.request_view(target[0],target[1]-target[0],automatic=True)
    def close(self):
        self.pause_playback(cancel_frame=False);self.cancel.set();self.executor.shutdown(wait=False,cancel_futures=True);self.root.destroy()

def main():
    p=argparse.ArgumentParser();p.add_argument('--file');p.add_argument('--scope',choices=['single','recording'],default='single');p.add_argument('--load');args=p.parse_args()
    root=tk.Tk();Observer(root,args);root.mainloop()
if __name__=='__main__':main()
