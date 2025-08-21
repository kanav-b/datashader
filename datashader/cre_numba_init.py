# cre_numba_init.py
import os

home = os.path.expanduser("~")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(home, "Library", "Caches", "cre", "numba_cache"))
os.environ.setdefault("CRE_CACHE_DIR",  os.path.join(home, "Library", "Caches", "cre", "cre_cache"))
os.environ.setdefault("MPLCONFIGDIR",   os.path.join(home, "Library", "Caches", "matplotlib"))

from numba.core.dispatcher import Dispatcher
from .cre_cache_helpers import enable_precise_caching
Dispatcher.enable_caching = enable_precise_caching
print("^^^^^hello")