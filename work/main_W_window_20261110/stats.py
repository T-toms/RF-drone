import functools
import math

@functools.lru_cache(None)
def cp(k,n):
    if k==n:return 1.
    # Invert regularized beta(k+1,n-k), using the continued fraction (no scipy).
    a=k+1.;b=n-k
    def cf(a,b,x):
        qab=a+b;qap=a+1;qam=a-1;c=1.;d=1-qab*x/qap;d=1/max(d,1e-300);h=d
        for m in range(1,501):
            aa=m*(b-m)*x/((qam+2*m)*(a+2*m));d=1+aa*d;d=d if abs(d)>1e-300 else 1e-300;c=1+aa/c;c=c if abs(c)>1e-300 else 1e-300;d=1/d;h*=d*c
            aa=-(a+m)*(qab+m)*x/((a+2*m)*(qap+2*m));d=1+aa*d;d=d if abs(d)>1e-300 else 1e-300;c=1+aa/c;c=c if abs(c)>1e-300 else 1e-300;d=1/d;delta=d*c;h*=delta
            if abs(delta-1)<3e-14:break
        return h
    def beta(x):
        bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
        return bt*cf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-bt*cf(b,a,1-x)/b
    lo=0.;hi=1.
    for _ in range(60):
        mid=(lo+hi)/2
        if beta(mid)<.95:lo=mid
        else:hi=mid
    return (lo+hi)/2
