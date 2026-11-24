"""Forward to the existing Anaconda SciPy; installs and copies no packages."""
import sys as _sys
from importlib.machinery import PathFinder as _Finder
from importlib.util import module_from_spec as _make
_existing='D:/anaconda/Lib/site-packages'
if _existing not in _sys.path:_sys.path.append(_existing)
_spec=_Finder.find_spec(__name__,[_existing])
if _spec is None:raise ImportError('Existing system SciPy is unavailable')
_real=_make(_spec);_sys.modules[__name__]=_real;_spec.loader.exec_module(_real)
globals().update(vars(_real))
