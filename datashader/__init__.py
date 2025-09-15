from __future__ import annotations

# Ensure Numba cache debug logs and a persistent cache directory are enabled
# as early as possible, before importing any submodules that may import numba.
import os
try:
    # Always enable detailed Numba cache logging so compile/load behavior is visible.
    os.environ.setdefault("NUMBA_DEBUG_CACHE", "1")
    # Persist Numba's cache inside the package so compiled artifacts are reused
    # across kernel restarts by default.
    _cache_dir = os.path.join(os.path.dirname(__file__), "__pycache__", "numba_cache_v2")
    os.makedirs(_cache_dir, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", _cache_dir)
except Exception:
    # Non-fatal if the environment cannot be set
    pass

from packaging.version import Version

from .__version import __version__  # noqa: F401

from .core import Canvas                                 # noqa (API import)
from .reductions import *                                # noqa (API import)
from .glyphs import Point                                # noqa (API import)
from .pipeline import Pipeline                           # noqa (API import)
from . import transfer_functions as tf                   # noqa (API import)
from . import data_libraries                             # noqa (API import)

# Make RaggedArray pandas extension array available for
# pandas >= 0.24.0 is installed
from pandas import __version__ as pandas_version
if Version(pandas_version) >= Version('0.24.0'):
    from . import datatypes  # noqa (API import)

# make pyct's example/data commands available if possible
from functools import partial
try:
    from pyct.cmd import copy_examples as _copy, fetch_data as _fetch, examples as _examples
    copy_examples = partial(_copy,'datashader')
    fetch_data = partial(_fetch,'datashader')
    examples = partial(_examples,'datashader')
except ImportError:
    def _missing_cmd(*args,**kw):
        return("install pyct to enable this command (e.g. `conda install pyct or "
               "`pip install pyct[cmd]`)")
    _copy = _fetch = _examples = _missing_cmd
    def err():
        raise ValueError(_missing_cmd())
    fetch_data = copy_examples = examples = err
del partial, _examples, _copy, _fetch
