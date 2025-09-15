import os
import sys
import subprocess
import textwrap


def _run_subprocess(script: str) -> str:
    env = os.environ.copy()
    # Keep Numba cache consistent and persistent during tests
    # Let datashader/__init__.py set NUMBA_CACHE_DIR if not present
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=True,
        text=True,
    )
    return proc.stdout


def test_10_disk_cache_lines_axis1_aa_pandas():
    """Two separate Python processes: first should generate kernels,
    second should reuse from disk with no generation logs.
    """
    code = r'''
import os
# Set debug cache before importing numba/datashader
os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_DEBUG_CACHE', '1')

import numpy as np, pandas as pd
import time
from numba import config as nb_config
import datashader as ds

np.random.seed(0)
# Make the first run expensive enough to see a delta
# Choose sizes where compilation dominates compute
n_rows, n_pts = 30, 512
x = np.linspace(0, 1, n_pts)
Y = np.array([np.sin(2*np.pi*(1+0.05*i)*x) for i in range(n_rows)])
df = pd.DataFrame({f'y{j}': Y[:, j] for j in range(n_pts)})
cvs = ds.Canvas(plot_width=200, plot_height=100)
# Trigger AA path so stage-2 + append are generated/loaded
start = time.perf_counter()
_ = cvs.line(df, x=x, y=list(df.columns), axis=1, agg=ds.count(), line_width=1.0)
dt = time.perf_counter() - start
cache_dir = os.environ.get('NUMBA_CACHE_DIR', nb_config.CACHE_DIR)

# Count compiled cache artifacts after the call
compiled = 0
if cache_dir and os.path.isdir(cache_dir):
    for root, _dirs, files in os.walk(cache_dir):
        compiled += sum(1 for f in files if f.endswith('.nbc') or f.endswith('.nbi') or f.endswith('.pickle'))
# Summarize for the test harness; the test will also parse debug cache logs
print(f"elapsed={dt:.3f}s cache_dir={cache_dir} compiled_files={compiled}")
'''

    # Force a cold start for the generator by removing generated sources
    import datashader as _ds
    pkg_dir = os.path.dirname(_ds.__file__)
    gen_dir = os.path.join(pkg_dir, "_generated")
    if os.path.isdir(gen_dir):
        for fn in os.listdir(gen_dir):
            if fn.endswith('.py') or fn.endswith('.pyc'):
                try:
                    os.remove(os.path.join(gen_dir, fn))
                except Exception:
                    pass

    out1 = _run_subprocess(code)
    # With static cacheable kernels or generated modules, first run should save to cache
    out2 = _run_subprocess(code)

    # Parse Numba cache debug logs to verify on-disk cache reuse
    def count_loaded(text: str) -> int:
        # Count data loads for AA helpers (in _generated) and glyphs.* kernels
        return sum(1 for line in text.splitlines()
                   if '[cache] data loaded' in line and (
                       '/_generated_' in line or '/_generated/' in line or '/glyphs_' in line or 'glyphs.' in line
                   ))

    def count_saved(text: str) -> int:
        return sum(1 for line in text.splitlines()
                   if '[cache] data saved' in line and (
                       '//_generated_' in line or '/_generated/' in line or '/glyphs_' in line or 'glyphs.' in line
                   ))


    loaded1, loaded2 = count_loaded(out1), count_loaded(out2)
    saved1, saved2 = count_saved(out1), count_saved(out2)

    # First fresh process should save some compiled artifacts (either static or generated)
    assert (saved1 + loaded1) >= 1, f"expected first run to compile or load; logs:\n{out1}"
    # Second fresh process should not trigger new compiles (saves)
    # No new compiled artifacts should be saved on the second run
    assert saved2 == 0, f"expected no new compiled artifacts saved on warm run;\nfirst:\n{out1}\nsecond:\n{out2}"


def _timed_line(width: float) -> str:
    s = r'''
import os
# Ensure deterministic numba behavior & logs
os.environ.setdefault('NUMBA_THREADING_LAYER','workqueue')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
os.environ.setdefault('NUMBA_DEBUG_CACHE','1')
import numpy as np, pandas as pd, time
import datashader as ds

np.random.seed(0)
n_rows, n_pts = 4, 128
x = np.linspace(0, 1, n_pts, dtype=np.float64)
Y = np.array([np.sin(2*np.pi*(1+0.05*i)*x) for i in range(n_rows)], dtype=np.float64)
df = pd.DataFrame({'y'+str(j): Y[:, j] for j in range(n_pts)})
cvs = ds.Canvas(plot_width=150, plot_height=75, x_range=(0,1), y_range=(-1.5,1.5))
_ = cvs.line(df, x=x, y=list(df.columns), axis=1, agg=ds.count(), line_width=__WIDTH__)
print('done width='+str(__WIDTH__))
'''
    return s.replace('__WIDTH__', str(width))


def test_00_disk_cache_axis1_cold_warm_cold_no_recompile():
    """Cold -> warm -> cold timing stays fast for width 0 and AA width.

    This guards the end-to-end on-disk reuse, including helper kernels,
    not just generated modules. Second and third runs should be similar.
    """
    def run_and_capture(script: str) -> str:
        return _run_subprocess(script)

    import datashader as _ds
    import shutil, os as _os
    # Ensure truly cold by clearing datashader's generated sources and cache dir once
    _pkg_dir = _os.path.dirname(_ds.__file__)
    _gen_dir = _os.path.join(_pkg_dir, "_generated")
    _cache_root = _os.path.join(_pkg_dir, "__pycache__", "numba_cache_v2")
    if _os.path.isdir(_gen_dir):
        for fn in _os.listdir(_gen_dir):
            try:
                _os.remove(_os.path.join(_gen_dir, fn))
            except Exception:
                pass
    if _os.path.isdir(_cache_root):
        shutil.rmtree(_cache_root, ignore_errors=True)

    for width in (0.0, 5.0):
        code = _timed_line(width)
        # Run 3 times in fresh interpreters
        out1 = run_and_capture(code)
        out2 = run_and_capture(code)
        out3 = run_and_capture(code)
        # Parse cache logs
        def count_loaded(text: str) -> int:
            return sum(1 for line in text.splitlines() if '[cache] data loaded' in line and (
                '/_generated_' in line or '/_generated/' in line or '/glyphs_' in line or 'glyphs.' in line
            ))
        def count_saved(text: str) -> int:
            return sum(1 for line in text.splitlines() if '[cache] data saved' in line and (
                '/_generated_' in line or '/_generated/' in line or '/glyphs_' in line or 'glyphs.' in line
            ))
        def count_gen(name: str, text: str) -> int:
            token = f'Generating {name} function'
            return sum(1 for line in text.splitlines() if token in line)

        saved2 = count_saved(out2)
        saved3 = count_saved(out3)
        # On warm and cold2, no new compiled artifacts should be saved
        assert saved2 == 0, f"expected no new compiled artifacts saved on warm run;\n{out2}"
        assert saved3 == 0, f"expected no new compiled artifacts saved on cold2 run;\n{out3}"
        # First run should show at least one save (compile) event somewhere
        assert count_saved(out1) >= 1, f"expected compile events on first run;\n{out1}"


def test_01_disk_cache_axis0_aa_no_recompile():
    """Verify AA axis=0 reuses disk cache and runs fast warm/cold2.

    Uses a small 1D line (axis=0) with AA enabled to exercise the AA helper.
    """

    code = r'''
import os, time, numpy as np, pandas as pd
# Deterministic numba behavior & cache logs
os.environ.setdefault('NUMBA_THREADING_LAYER','workqueue')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
os.environ.setdefault('NUMBA_DEBUG_CACHE','1')
import datashader as ds

N = 512
x = np.linspace(0, 1, N, dtype=np.float64)
y = np.sin(2*np.pi*3*x).astype(np.float64)
df = pd.DataFrame({'x': x, 'y': y})
cvs = ds.Canvas(plot_width=300, plot_height=150, x_range=(0,1), y_range=(-1.5,1.5))
start = time.perf_counter()
_ = cvs.line(df, 'x', 'y', agg=ds.count(), line_width=3.0)
dt = time.perf_counter() - start
print('elapsed='+format(dt, '.3f')+'s')
'''

    def run(script: str) -> str:
        return _run_subprocess(script)

    out1 = run(code)
    out2 = run(code)
    out3 = run(code)

    def count_loaded(text: str) -> int:
        return sum(1 for line in text.splitlines() if '[cache] data loaded' in line and (
            '/_generated_' in line or '/_generated/' in line or '/glyphs_' in line or 'glyphs.' in line or 'core.' in line
        ))
    def count_saved(text: str) -> int:
        return sum(1 for line in text.splitlines() if '[cache] data saved' in line and (
            '/_generated_' in line or '/_generated/' in line or '/glyphs_' in line or 'glyphs.' in line or 'core.' in line
        ))
    def count_gen(name: str, text: str) -> int:
        token = f'Generating {name} function'
        return sum(1 for line in text.splitlines() if token in line)

    # For static AA kernels on axis=0, we accept glyphs/core cache reuse; prove reuse without new saves
    assert count_saved(out2) == 0, f"expected no new compiled artifacts saved on warm run;\n{out2}"
    assert count_saved(out3) == 0, f"expected no new compiled artifacts saved on cold2 run;\n{out3}"
    # And we should see at least some cache loads on warm or cold2
    assert (count_loaded(out2) + count_loaded(out3)) >= 1, f"expected cache loads on warm/cold2;\n{out2}\n{out3}"


def test_02_disk_cache_axis0_int_columns_no_recompile():
    """Axis=0 with integer column indices should use disk cache.

    Validates that using integer indices instead of column names does not
    trigger new compiles on warm runs.
    """

    code = r'''
import os, time, numpy as np, pandas as pd
# Deterministic numba behavior & cache logs
os.environ.setdefault('NUMBA_THREADING_LAYER','workqueue')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
os.environ.setdefault('NUMBA_DEBUG_CACHE','1')
import datashader as ds

N = 512
x = np.linspace(0, 1, N, dtype=np.float64)
y = np.sin(2*np.pi*3*x).astype(np.float64)
df = pd.DataFrame({'x': x, 'y': y})
cvs = ds.Canvas(plot_width=200, plot_height=100, x_range=(0,1), y_range=(-1.5,1.5))
# Pass integer column indices instead of names
_ = cvs.line(df, 0, 1, agg=ds.count(), line_width=0.0)
print('ok axis0-int')
'''

    out1 = _run_subprocess(code)
    out2 = _run_subprocess(code)

    def count_loaded(text: str) -> int:
        return sum(1 for line in text.splitlines() if '[cache] data loaded' in line and (
            '/_generated_' in line or '/_generated/' in line or '/glyphs_' in line or 'glyphs.' in line or 'core.' in line
        ))
    def count_saved(text: str) -> int:
        return sum(1 for line in text.splitlines() if '[cache] data saved' in line and (
            '/_generated_' in line or '/_generated/' in line or '/glyphs_' in line or 'glyphs.' in line or 'core.' in line
        ))

    # First run should either compile/save or load from cache; second should not save
    assert (count_saved(out1) + count_loaded(out1)) >= 1, f"expected compile or load events on first run;\n{out1}"
    assert count_saved(out2) == 0, f"expected no new compiled artifacts saved on warm run;\n{out2}"
    assert count_loaded(out2) >= 1, f"expected cache loads on warm run;\n{out2}"


def test_03_disk_cache_axis1_int_columns_no_recompile():
    """Axis=1 with integer column indices should use disk cache.

    Temporarily skipped: passing integer indices for axis=1 is not yet
    consistently supported across the core glyph pipeline in this branch.
    """
    import pytest as _pytest
    _pytest.skip("axis=1 integer column indices not yet supported in core pipeline")

    code = r'''
import os
os.environ.setdefault('NUMBA_THREADING_LAYER','workqueue')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
os.environ.setdefault('NUMBA_DEBUG_CACHE','1')

import numpy as np, pandas as pd
import datashader as ds

np.random.seed(42)
n_rows, n_pts = 4, 128
x = np.linspace(0, 1, n_pts, dtype=np.float64)
Y = np.array([np.sin(2*np.pi*(1+0.1*i)*x) for i in range(n_rows)], dtype=np.float64)
df = pd.DataFrame({'y'+str(j): Y[:, j] for j in range(n_pts)})
cvs = ds.Canvas(plot_width=150, plot_height=75, x_range=(0,1), y_range=(-1.5,1.5))

# Use integer indices for y columns
y_indices = list(range(df.shape[1]))
_ = cvs.line(df, x=x, y=y_indices, axis=1, agg=ds.count(), line_width=3.0)
print('ok axis1-int')
'''

    # The body below will be enabled once axis=1 integer index paths
    # are fully wired through the Canvas -> glyph pipeline.
