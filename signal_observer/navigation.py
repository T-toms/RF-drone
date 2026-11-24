"""Sample-coordinate viewport bounds, independent of any assumed clock."""
def viewport(bounds, start, width, minimum=32):
    lo, hi = bounds
    width = min(hi-lo, max(int(minimum), int(round(width))))
    start = max(lo, min(hi-width, int(round(start))))
    return start, start+width

def zoom_view(bounds, view, factor, anchor=None, minimum=32):
    a, b = view
    anchor = (a+b)/2 if anchor is None else max(a, min(b, anchor))
    ratio = (anchor-a)/(b-a)
    width = min(bounds[1]-bounds[0], max(minimum, round((b-a)*factor)))
    return viewport(bounds, anchor-ratio*width, width, minimum)

def next_play_view(bounds,view,fraction=.5):
    """Advance without skipping any part of the recording, including its tail."""
    if not 0<fraction<=1:raise ValueError('播放步长必须大于0且不超过一个视窗。')
    a,b=view
    if b>=bounds[1]:return None
    width=b-a
    return viewport(bounds,a+max(1,round(width*fraction)),width,1)
