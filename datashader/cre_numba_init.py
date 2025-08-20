# cre_numba_init.py
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/Users/kanu/Library/Caches/cre/numba_cache")
os.environ.setdefault("CRE_CACHE_DIR",  "/Users/kanu/Library/Caches/cre/cre_cache")
os.environ.setdefault("MPLCONFIGDIR",   "/Users/kanu/Library/Caches/matplotlib") 

from numba.core.dispatcher import Dispatcher
from .cre_cache_helpers import enable_precise_caching
Dispatcher.enable_caching = enable_precise_caching
print("^^^^^hello")
