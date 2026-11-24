"""Meaningful checks for clock provenance, skipped gaps, and MAT decoding."""
import tempfile,threading,unittest,os
from pathlib import Path
import numpy as np
import h5py
from scipy.io import savemat
from analysis import MatInfo,Part,Plan,inspect_mat,make_plan,analyze,save_result,load_result,Cancelled,IQCache
from plots import time_scale
from navigation import viewport,zoom_view

class ObserverChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.n=8192
        self.t=np.arange(self.n);self.x=np.exp(2j*np.pi*.125*self.t).astype(np.complex64)
        self.file=self.root/'signal.mat';savemat(self.file,{'data':self.x.reshape(1,-1)})
    def tearDown(self):self.tmp.cleanup()
    def plan(self):return make_plan(inspect_mat(self.file),'data',audit=self.root)
    def test_unknown_clock_and_center_remain_unknown(self):
        info=inspect_mat(self.file);self.assertIsNone(info.fs_hz);self.assertIsNone(info.fc_hz)
        result=analyze(self.plan(),nfft=512)
        self.assertAlmostEqual(np.argmax(result['spectrum'])/512-.5,.125)
        self.assertIn('采样点',time_scale(result,None)[1])
    def test_known_metadata_has_explicit_units(self):
        f=self.root/'known.mat';savemat(f,{'IQ':self.x,'fs':250000000,'centfreq':2440000000})
        info=inspect_mat(f);self.assertEqual(info.fs_hz,250000000);self.assertEqual(info.fc_hz,2440000000)
    def test_gap_not_filled_and_mean_not_diluted(self):
        p=Plan('gap',[Part(str(self.file),'data',self.n,0,0,self.n),Part(str(self.file),'data',self.n,2*self.n,0,self.n)],3*self.n,'test',str(self.file))
        r=analyze(p,nfft=512,max_bins=48)
        self.assertEqual(r['frames']*512,2*self.n);self.assertTrue(np.isnan(r['mean_db'][16:32]).all());self.assertTrue((r['weights'][16:32]==0).all())
        a=analyze(self.plan(),nfft=512);np.testing.assert_allclose(r['spectrum'],a['spectrum'])
    def test_invalid_prefix_excluded_but_coordinates_kept(self):
        p=Plan('prefix',[Part(str(self.file),'data',self.n,0,2048,self.n)],self.n,'test',str(self.file))
        r=analyze(p,nfft=512,max_bins=16)
        self.assertEqual(r['sources'][0]['source_interval'],[2048,self.n]);self.assertTrue(np.isnan(r['mean_db'][:4]).all())
    def test_range_is_clipped_not_zero_padded(self):
        r=analyze(self.plan(),start=4096,count=100000,nfft=512)
        self.assertEqual(r['stop'],self.n);self.assertEqual(r['frames']*512,4096)
    def test_subrange_boundaries_do_not_pull_outside_samples(self):
        r=analyze(self.plan(),start=100,count=3000,nfft=512)
        self.assertEqual(r['sources'][0]['source_interval'],[512,3072])
    def test_real_array_not_silently_treated_as_iq(self):
        f=self.root/'real.mat';savemat(f,{'data':self.x.real})
        with self.assertRaisesRegex(ValueError,'复数'):analyze(make_plan(inspect_mat(f),'data',audit=self.root),nfft=512)
    def test_hdf5_matlab_compound_iq(self):
        f=self.root/'hdf.mat';d=np.empty((self.n,1),dtype=[('real','<f8'),('imag','<f8')]);d['real'][:,0]=self.x.real;d['imag'][:,0]=self.x.imag
        with h5py.File(f,'w') as h:h['IQ']=d;h['fs']=np.array([[125000000.]])
        info=inspect_mat(f);self.assertEqual(info.fs_hz,125000000)
        r=analyze(make_plan(info,'IQ',audit=self.root),nfft=512)
        a=analyze(self.plan(),nfft=512);np.testing.assert_allclose(r['spectrum'],a['spectrum'])
    def test_cancellation(self):
        e=threading.Event();e.set()
        with self.assertRaises(Cancelled):analyze(self.plan(),nfft=512,cancel=e)
    def test_export_preserves_assumption_and_numeric_statistics(self):
        r=analyze(self.plan(),nfft=512);p=self.root/'result.npz';save_result(r,p,122880000,None,'assumed');loaded=load_result(p)
        self.assertEqual(loaded['clock_status'],'user assumption');self.assertIsNone(loaded['fc_hz']);np.testing.assert_array_equal(loaded['weights'],r['weights'])
    def test_too_short_or_empty_interval_rejected(self):
        with self.assertRaises(ValueError):analyze(self.plan(),count=50,nfft=512)
        with self.assertRaises(ValueError):analyze(self.plan(),count=0,nfft=512)
        with self.assertRaises(ValueError):analyze(self.plan(),start=self.n,nfft=512)
    def test_overlapping_local_view_matches_direct_fft(self):
        r=analyze(self.plan(),start=100,count=7000,nfft=512,hop=128,max_bins=2000)
        lo,hi=r['sources'][0]['source_interval'];frames=np.lib.stride_tricks.sliding_window_view(self.x[lo:hi],512)[::128]
        win=np.hanning(512).astype(np.float32)
        expected=np.mean(np.abs(np.fft.fftshift(np.fft.fft(frames*win,axis=1),axes=1))**2/np.sum(win*win),axis=0)
        np.testing.assert_allclose(r['spectrum'],expected,rtol=2e-5,atol=1e-5)
        self.assertEqual(r['frames'],len(frames));self.assertEqual(r['analyzed_samples'],hi-lo)
        self.assertGreater(r['frames']*512,r['analyzed_samples']);self.assertGreaterEqual(lo,100);self.assertLessEqual(hi,7100)
    def test_view_entirely_inside_gap_is_blank_not_previous_data(self):
        p=Plan('gap',[Part(str(self.file),'data',self.n,0,0,self.n),Part(str(self.file),'data',self.n,2*self.n,0,self.n)],3*self.n,'test',str(self.file))
        r=analyze(p,start=self.n,count=self.n,nfft=512,hop=128,allow_empty=True)
        self.assertEqual(r['frames'],0);self.assertEqual(r['analyzed_samples'],0);self.assertEqual(r['sources'],[])
        self.assertTrue(np.isnan(r['mean_db']).all())
    def test_drag_reaches_both_ends_without_changing_view_width(self):
        bounds=(1000,1001000)
        self.assertEqual(viewport(bounds,-50000,262144),(1000,263144))
        self.assertEqual(viewport(bounds,999999999,262144),(738856,1001000))
        self.assertEqual(viewport(bounds,5000,2000000),bounds)
    def test_scroll_zoom_keeps_mouse_anchor_and_limits(self):
        self.assertEqual(zoom_view((0,10000),(2000,6000),.5,3000,128),(2500,4500))
        a,b=zoom_view((0,10000),(0,4000),100,1000,128);self.assertEqual((a,b),(0,10000))
        a,b=zoom_view((0,10000),(2000,6000),.0001,3000,128);self.assertEqual(b-a,128);self.assertLessEqual(a,3000);self.assertGreater(b,3000)
    def test_iq_cache_reuses_read_only_vector_and_respects_memory_bound(self):
        cache=IQCache(max_bytes=self.x.nbytes)
        first=analyze(self.plan(),nfft=512,cache=cache);second=analyze(self.plan(),nfft=512,cache=cache)
        self.assertEqual((cache.loads,cache.hits),(1,1));np.testing.assert_array_equal(first['spectrum'],second['spectrum'])
        self.assertFalse(next(iter(cache.entries.values())).flags.writeable)
        other=self.root/'other.mat';savemat(other,{'data':self.x})
        analyze(make_plan(inspect_mat(other),'data',audit=self.root),nfft=512,cache=cache)
        self.assertEqual(len(cache.entries),1);self.assertLessEqual(cache.bytes,cache.max_bytes)
        cache.clear();self.assertEqual(cache.bytes,0)
    def test_iq_cache_invalidates_changed_source(self):
        cache=IQCache();before=analyze(self.plan(),nfft=512,cache=cache);stamp=self.file.stat()
        x=np.exp(2j*np.pi*.25*self.t).astype(np.complex64);savemat(self.file,{'data':x.reshape(1,-1)})
        os.utime(self.file,ns=(stamp.st_atime_ns,stamp.st_mtime_ns+1000000))
        after=analyze(self.plan(),nfft=512,cache=cache)
        self.assertEqual(cache.loads,2);self.assertNotEqual(int(np.argmax(before['spectrum'])),int(np.argmax(after['spectrum'])))
    def test_live_plot_reuses_axes_and_updates_exact_coordinates(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from plots import draw_result
        overview=analyze(self.plan(),nfft=512)
        first=analyze(self.plan(),start=0,count=3000,nfft=512,hop=128)
        second=analyze(self.plan(),start=3000,count=3000,nfft=512,hop=128)
        fig=Figure();FigureCanvasAgg(fig);axes,scale=draw_result(fig,first,overview=overview);image=fig._observer_live['image'];count=len(fig.axes)
        moved,_=draw_result(fig,second,overview=overview,detail_kind='peak_db')
        self.assertIs(axes[1],moved[1]);self.assertIs(image,fig._observer_live['image']);self.assertEqual(len(fig.axes),count)
        np.testing.assert_allclose(moved[1].get_xlim(),np.array([3000,6000])*scale)
        self.assertAlmostEqual(image.get_extent()[0],3000*scale)
        np.testing.assert_allclose(image.get_array(),second['peak_db'].T-fig._observer_live['ref'])
        fig.canvas.draw()

if __name__=='__main__':unittest.main(verbosity=2)
