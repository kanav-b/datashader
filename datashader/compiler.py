from __future__ import annotations
from itertools import count
import logging
from typing import TYPE_CHECKING

from toolz import unique, concat, pluck, get, memoize
from numba import literal_unroll
import numpy as np
import xarray as xr

import numba as nb, datashader as ds


# --- disk-backed emission for generated @njit functions ---
# For AA kernel caching
from .cre_cache_helpers import unique_hash, source_to_cache, import_from_cached, import_module_from_cached

from .antialias import AntialiasCombination
from .reductions import SpecialColumn, UsesCudaMutex, by, category_codes, summary
from .utils import (isnull, ngjit,
    nanmax_in_place, nanmin_in_place, nansum_in_place, nanfirst_in_place, nanlast_in_place,
    nanmax_n_in_place_3d, nanmax_n_in_place_4d, nanmin_n_in_place_3d, nanmin_n_in_place_4d,
    nanfirst_n_in_place_3d, nanfirst_n_in_place_4d, nanlast_n_in_place_3d, nanlast_n_in_place_4d,
    row_min_in_place, row_min_n_in_place_3d, row_min_n_in_place_4d,
    row_max_in_place, row_max_n_in_place_3d, row_max_n_in_place_4d,
)

try:
    from datashader.transfer_functions._cuda_utils import cuda_mutex_lock, cuda_mutex_unlock
except ImportError:
    cuda_mutex_lock, cuda_mutex_unlock = None, None

if TYPE_CHECKING:
    from datashader.antialias import UnzippedAntialiasStage2


__all__ = ['compile_components']


logger = logging.getLogger(__name__)

# --- disk-backed emission for generated @njit functions ---
# Writes the generated function to a stable on-disk module keyed by a hash of
# the spec + function body, then imports it and returns the callable.
# This enables reliable reuse across processes and runs, and plays nicely with
# Numba's own cache because the function has a real filename.

def _cache_emit_njit(func_name: str, lines: list[str], cache_key_parts: list, *,
                    extra_imports: list[str] | None = None, bindings: dict[str, object] | None = None):
    # Build a deterministic source file containing a single @njit function.
    # `lines` must start with `def {func_name}(...):`.
    header = [
        "import numpy as np",
        "from numba import njit, literal_unroll",
        # Import the in-place combine kernels that the generated code may call
        "from datashader.utils import (",
        "    nanmax_in_place, nanmin_in_place, nansum_in_place,",
        "    nanfirst_in_place, nanlast_in_place,",
        "    nanmax_n_in_place_3d, nanmax_n_in_place_4d,",
        "    nanmin_n_in_place_3d, nanmin_n_in_place_4d,",
        "    nanfirst_n_in_place_3d, nanfirst_n_in_place_4d,",
        "    nanlast_n_in_place_3d, nanlast_n_in_place_4d,",
        "    row_min_in_place, row_min_n_in_place_3d, row_min_n_in_place_4d,",
        "    row_max_in_place, row_max_n_in_place_3d, row_max_n_in_place_4d,",
        ")",
        "",
    ]
    if extra_imports:
        header.extend(extra_imports)
        header.append("")

    body = ["@njit(nopython=True, nogil=True, cache=True)"] + lines
    source = "\n".join(header + body) + "\n"

    # Create a stable hash from the spec and the function body.
    # The module group name is kept constant so all DS kernels live under one path.
    h = unique_hash(tuple(cache_key_parts) + ("\n".join(lines),))
    name = "datashader_kernels"

    # Write and import the cached module
    source_to_cache(name, h, source)
    if bindings:
        mod = import_module_from_cached(name, h)
        for k, v in bindings.items():
            setattr(mod, k, v)
        return getattr(mod, func_name)   # already @njit(cache=True)
    mod_dict = import_from_cached(name, h, [func_name])
    return mod_dict[func_name]


    # mod_dict = import_from_cached(name, h, [func_name])
    # return mod_dict[func_name]


@memoize
def compile_components(agg, schema, glyph, *, antialias=False, cuda=False, partitioned=False):
    """Given an ``Aggregation`` object and a schema, return 5 sub-functions
    and information on how to perform the second stage aggregation if
    antialiasing is requested,

    Parameters
    ----------
    agg : Aggregation
        The expression describing the aggregation(s) to be computed.

    schema : DataShape
        Columns and dtypes in the source dataset.

    glyph : Glyph
        The glyph to render.

    antialias : bool
        Whether to render using antialiasing.

    cuda : bool
        Whether to render using CUDA (on the GPU) or CPU.

    partitioned : bool
        Whether the source dataset is partitioned using dask.

    Returns
    -------
    A tuple of the following:

    ``create(shape)``
        Function that takes the aggregate shape, and returns a tuple of
        initialized numpy arrays.

    ``info(df, canvas_shape)``
        Function that takes a dataframe, and returns preprocessed 1D numpy
        arrays of the needed columns.

    ``append(i, x, y, *aggs_and_cols)``
        Function that appends the ``i``th row of the table to the ``(x, y)``
        bin, given the base arrays and columns in ``aggs_and_cols``. This does
        the bulk of the work.

    ``combine(base_tuples)``
        Function that combines a list of base tuples into a single base tuple.
        This forms the reducing step in a reduction tree.

    ``finalize(aggs, cuda)``
        Function that is given a tuple of base numpy arrays and returns the
        finalized ``DataArray`` or ``Dataset``.

    ``antialias_stage_2``
        If using antialiased lines this is a tuple of the ``AntialiasCombination``
        values corresponding to the aggs. If not using antialiased lines then
        this is ``False``.

    ``antialias_stage_2_funcs``
        If using antialiased lines which require a second stage combine, this
        is a tuple of the three combine functions which are the accumulate,
        clear and copy_back functions. If not using antialiased lines then this
        is ``None``.

    ``column_names``
        Names of DataFrame columns or DataArray variables that are used by the
        agg.
    """
    reds = list(traverse_aggregation(agg))

    # List of base reductions (actually computed)
    bases = list(unique(concat(r._build_bases(cuda, partitioned) for r in reds)))
    dshapes = [b.out_dshape(schema, antialias, cuda, partitioned) for b in bases]

    # Information on how to perform second stage aggregation of antialiased lines,
    # including whether antialiased lines self-intersect or not as we need a single
    # value for this even for a compound reduction. This is by default True, but
    # is False if a single constituent reduction requests it.
    if antialias:
        self_intersect, antialias_stage_2 = make_antialias_stage_2(reds, bases)
        if cuda:
            import cupy
            array_module = cupy
        else:
            array_module = np
        antialias_stage_2 = antialias_stage_2(array_module)
        antialias_stage_2_funcs = make_antialias_stage_2_functions(antialias_stage_2, bases, cuda,
                                                                   partitioned)
    else:
        self_intersect = False
        antialias_stage_2 = False
        antialias_stage_2_funcs = None

    # List of tuples of
    # (append, base, input columns, temps, combine temps, uses cuda mutex, is_categorical)
    calls = [_get_call_tuples(b, d, schema, cuda, antialias, self_intersect, partitioned)
             for (b, d) in zip(bases, dshapes)]

    # List of unique column names needed, including nan_check_columns
    cols = list(concat(pluck(2, calls)))
    nan_check_cols = list(c[3] for c in calls if c[3] is not None)
    cols = list(unique(cols + nan_check_cols))

    # List of temps needed
    temps = list(pluck(4, calls))
    combine_temps = list(pluck(5, calls))

    create = make_create(bases, dshapes, cuda)
    append, any_uses_cuda_mutex = make_append(bases, cols, calls, glyph, antialias)
    info = make_info(cols, cuda, any_uses_cuda_mutex)
    combine = make_combine(bases, dshapes, temps, combine_temps, antialias, cuda, partitioned)
    finalize = make_finalize(bases, agg, schema, cuda, partitioned)

    column_names = [c.column for c in cols if c.column != SpecialColumn.RowIndex]

    return create, info, append, combine, finalize, antialias_stage_2, antialias_stage_2_funcs, \
        column_names


def _get_antialias_stage_2_combine_func(combination: AntialiasCombination, zero: float,
                                        n_reduction: bool, categorical: bool):
    if n_reduction:
        if zero == -1:
            if combination in (AntialiasCombination.MAX, AntialiasCombination.LAST):
                return row_max_n_in_place_4d if categorical else row_max_n_in_place_3d
            elif combination in (AntialiasCombination.MIN, AntialiasCombination.FIRST):
                return row_min_n_in_place_4d if categorical else row_min_n_in_place_3d
            else:
                raise NotImplementedError
        else:
            if combination == AntialiasCombination.MAX:
                return nanmax_n_in_place_4d if categorical else nanmax_n_in_place_3d
            elif combination == AntialiasCombination.MIN:
                return nanmin_n_in_place_4d if categorical else nanmin_n_in_place_3d
            elif combination == AntialiasCombination.FIRST:
                return nanfirst_n_in_place_4d if categorical else nanfirst_n_in_place_3d
            elif combination == AntialiasCombination.LAST:
                return nanlast_n_in_place_4d if categorical else nanlast_n_in_place_3d
            else:
                raise NotImplementedError
    else:
        # The aggs to combine here are either 3D (ny, nx, ncat) if categorical is True or
        # 2D (ny, nx) if categorical is False. The same combination functions can be for both
        # as all elements are independent.
        if zero == -1:
            if combination in (AntialiasCombination.MAX, AntialiasCombination.LAST):
                return row_max_in_place
            elif combination in (AntialiasCombination.MIN, AntialiasCombination.FIRST):
                return row_min_in_place
            else:
                raise NotImplementedError
        else:
            if combination == AntialiasCombination.MAX:
                return nanmax_in_place
            elif combination == AntialiasCombination.MIN:
                return nanmin_in_place
            elif combination == AntialiasCombination.FIRST:
                return nanfirst_in_place
            elif combination == AntialiasCombination.LAST:
                return nanlast_in_place
            else:
                return nansum_in_place


def make_antialias_stage_2_functions(antialias_stage_2, bases, cuda, partitioned):
    aa_combinations, aa_zeroes, aa_n_reductions, aa_categorical = antialias_stage_2

    # Accumulate functions.
    funcs = [_get_antialias_stage_2_combine_func(comb, zero, n_red, cat) for comb, zero, n_red, cat
             in zip(aa_combinations, aa_zeroes, aa_n_reductions, aa_categorical)]

    base_is_where = [b.is_where() for b in bases]
    next_base_is_where = base_is_where[1:] + [False]

    namespace = {}
    namespace["literal_unroll"] = literal_unroll
    for func in set(funcs):
        namespace[func.__name__] = func

    # Generator of unique names for combine functions
    names = (f"combine{i}" for i in count())

    # aa_stage_2_accumulate
    lines = [
        "def aa_stage_2_accumulate(aggs_and_copies, first_pass):",
        #    Don't need to accumulate if first_pass, just copy (opposite of aa_stage_2_copy_back)
        "    if first_pass:",
        "        for a in literal_unroll(aggs_and_copies):",
        "            a[1][:] = a[0][:]",
        "    else:",
    ]
    for i, (func, is_where, next_is_where) in enumerate(zip(funcs, base_is_where,
                                                            next_base_is_where)):
        if is_where:
            where_reduction = bases[i]
            if isinstance(where_reduction, by):
                where_reduction = where_reduction.reduction

            combine = where_reduction._combine_callback(cuda, partitioned, aa_categorical[i])
            name = next(names)  # Unique name
            namespace[name] = combine

            lines.append(
                f"        {name}(aggs_and_copies[{i}][::-1], aggs_and_copies[{i-1}][::-1])")
        elif next_is_where:
            # This is dealt with as part of the following base which is a where reduction.
            pass
        else:
            lines.append(
                f"        {func.__name__}(aggs_and_copies[{i}][1], aggs_and_copies[{i}][0])")
    code = "\n".join(lines)
    logger.debug(code)
    # Build a spec key sensitive to AA settings and the chosen combine funcs
    spec_parts = [
        "aa_stage_2_accumulate",
        tuple(str(f.__name__) for f in funcs),
        tuple(bool(b) for b in base_is_where),
        tuple(bool(n) for n in next_base_is_where),
        tuple(bool(c) for c in aa_categorical),
    ]
    aa_stage_2_accumulate = _cache_emit_njit(
        "aa_stage_2_accumulate",
        lines,
        spec_parts,
        extra_imports=["# namespace helpers are resolved via imports above"],
    )

    # aa_stage_2_clear
    if np.any(np.isnan(aa_zeroes)):
        namespace["nan"] = np.nan

    lines = ["def aa_stage_2_clear(aggs_and_copies):"]
    for i, aa_zero in enumerate(aa_zeroes):
        lines.append(f"    aggs_and_copies[{i}][0].fill({aa_zero})")
    code = "\n".join(lines)
    logger.debug(code)
    spec_parts = [
        "aa_stage_2_clear",
        tuple(float(z) for z in aa_zeroes.ravel()) if hasattr(aa_zeroes, "ravel") else tuple(aa_zeroes),
    ]
    aa_stage_2_clear = _cache_emit_njit(
        "aa_stage_2_clear",
        lines,
        spec_parts,
        extra_imports=["nan = np.nan" if "nan" in code else ""],
    )

    # aa_stage_2_copy_back
    @ngjit
    def aa_stage_2_copy_back(aggs_and_copies):
        # Numba access to heterogeneous tuples is only permitted using literal_unroll.
        for agg_and_copy in literal_unroll(aggs_and_copies):
            agg_and_copy[0][:] = agg_and_copy[1][:]

    return aa_stage_2_accumulate, aa_stage_2_clear, aa_stage_2_copy_back


def traverse_aggregation(agg):
    """Yield a left->right traversal of an aggregation"""
    if isinstance(agg, summary):
        for a in agg.values:
            yield from traverse_aggregation(a)
    else:
        yield agg


def _get_call_tuples(base, dshape, schema, cuda, antialias, self_intersect, partitioned):
    # Comments refer to usage in make_append()
    return (
        base._build_append(dshape, schema, cuda, antialias, self_intersect),  # func
        (base,),  # bases
        base.inputs,  # cols, arrays of these are passed to reduction append functions
        base.nan_check_column,  # column used to check for NaNs in some where reductions
        base._build_temps(cuda),  # temps
        base._build_combine_temps(cuda, partitioned),  # combine temps
        base.uses_cuda_mutex() if cuda else UsesCudaMutex.No,  # uses cuda mutex
        base.is_categorical(),
    )


def make_create(bases, dshapes, cuda):
    creators = [b._build_create(d) for (b, d) in zip(bases, dshapes)]
    if cuda:
        import cupy
        array_module = cupy
    else:
        array_module = np
    return lambda shape: tuple(c(shape, array_module) for c in creators)


def make_info(cols, cuda, uses_cuda_mutex: bool):
    def info(df, canvas_shape):
        ret = tuple(c.apply(df, cuda) for c in cols)
        if uses_cuda_mutex:
            import cupy  # Guaranteed to be available if uses_cuda_mutex is True
            import numba
            from packaging.version import Version
            if Version(numba.__version__) >= Version("0.57"):
                mutex_array = cupy.zeros(canvas_shape, dtype=np.uint32)
            else:
                mutex_array = cupy.zeros((1,), dtype=np.uint32)
            ret += (mutex_array,)
        return ret

    return info


def make_append(bases, cols, calls, glyph, antialias):
    names = ('_{0}'.format(i) for i in count())
    inputs = list(bases) + list(cols)
    namespace = {}
    need_isnull = any(call[3] for call in calls)
    if need_isnull:
        namespace["isnull"] = isnull
    global_cuda_mutex = any(call[6] == UsesCudaMutex.Global for call in calls)
    any_uses_cuda_mutex = any(call[6] != UsesCudaMutex.No for call in calls)
    if any_uses_cuda_mutex:
        # This adds an argument to the append() function that is the cuda mutex
        # generated in make_info.
        inputs += ["_cuda_mutex"]
        namespace["cuda_mutex_lock"] = cuda_mutex_lock
        namespace["cuda_mutex_unlock"] = cuda_mutex_unlock
    signature = [next(names) for i in inputs]
    arg_lk = dict(zip(inputs, signature))
    local_lk = {}
    head = []
    body = []
    ndims = glyph.ndims
    if ndims is not None:
        subscript = ', '.join(['i' + str(n) for n in range(ndims)])
    else:
        subscript = None
    prev_local_cuda_mutex = False
    categorical_args = {}  # Reuse categorical arguments if used in more than one reduction
    where_selectors = {}  # Reuse where.selector if used more than once in a summary reduction

    if logger.isEnabledFor(logging.DEBUG):   # mostly does nothing...
        logger.debug(f"global_cuda_mutex {global_cuda_mutex}")
        logger.debug(f"any_uses_cuda_mutex {any_uses_cuda_mutex}")
        for k, v in arg_lk.items():
            logger.debug(f"arg_lk {v} {k} {getattr(k, 'column', None)}")

    def get_cuda_mutex_call(lock: bool) -> str:
        func = "cuda_mutex_lock" if lock else "cuda_mutex_unlock"
        return f'{func}({arg_lk["_cuda_mutex"]}, (y, x))'

    for index, (func, bases, cols, nan_check_column, temps, _, uses_cuda_mutex, categorical) \
            in enumerate(calls):
        local_cuda_mutex = not global_cuda_mutex and uses_cuda_mutex == UsesCudaMutex.Local
        local_lk.update(zip(temps, (next(names) for i in temps)))
        func_name = next(names)
        logger.debug(f"func {func_name} {func}")
        namespace[func_name] = func
        args = [arg_lk[i] for i in bases]
        if categorical and isinstance(cols[0], category_codes):
            args.extend('{0}[{1}]'.format(arg_lk[col], subscript) for col in cols[1:])
        elif ndims is None:
            args.extend('{0}'.format(arg_lk[i]) for i in cols)
        elif categorical:
            args.extend('{0}[{1}][1]'.format(arg_lk[i], subscript)
                        for i in cols)
        else:
            args.extend('{0}[{1}]'.format(arg_lk[i], subscript)
                        for i in cols)

        if categorical:
            # Categorical aggregate arrays need to be unpacked
            categorical_arg = arg_lk[cols[0]]
            cat_name = categorical_args.get(categorical_arg, None)
            if cat_name is None:
                # Each categorical column only needs to be unpacked once
                col_index = '' if isinstance(cols[0], category_codes) else '[0]'
                cat_name = f'cat{next(names)}'
                categorical_args[categorical_arg] = cat_name
                head.append(f'{cat_name} = int({categorical_arg}[{subscript}]{col_index})')
            arg = signature[index]
            head.append(f'{arg} = {arg}[:, :, {cat_name}]')

        args.extend([local_lk[i] for i in temps])
        if antialias:
            args += ["aa_factor", "prev_aa_factor"]

        if local_cuda_mutex and prev_local_cuda_mutex:
            # Avoid unnecessary mutex unlock and lock cycle
            body.pop()

        is_where = len(bases) == 1 and bases[0].is_where()
        if is_where:
            where_reduction = bases[0]
            if isinstance(where_reduction, by):
                where_reduction = where_reduction.reduction

            selector_hash = hash(where_reduction.selector)
            update_index_arg_name = where_selectors.get(selector_hash, None)
            new_selector = update_index_arg_name is None
            if new_selector:
                update_index_arg_name = next(names)
                where_selectors[selector_hash] = update_index_arg_name
            args.append(update_index_arg_name)

            # where reduction needs access to the return of the contained
            # reduction, which is the preceding one here.
            prev_body = body.pop()
            if local_cuda_mutex and not prev_local_cuda_mutex:
                body.append(get_cuda_mutex_call(True))
            if new_selector:
                body.append(f'{update_index_arg_name} = {prev_body}')
            else:
                body.append(prev_body)

            # If nan_check_column is defined then need to check if value of
            # correct row in that column is NaN and if so do nothing. This
            # check needs to occur before the where.selector is called.
            if nan_check_column is None:
                whitespace = ''
            else:
                var = f"{arg_lk[nan_check_column]}[{subscript}]"
                prev_body = body[-1]
                body[-1] = f'if not isnull({var}):'
                body.append(f'    {prev_body}')
                whitespace = '    '

            body.append(f'{whitespace}if {update_index_arg_name} >= 0:')
            body.append(f'    {whitespace}{func_name}(x, y, {", ".join(args)})')
        else:
            if local_cuda_mutex and not prev_local_cuda_mutex:
                body.append(get_cuda_mutex_call(True))
            if nan_check_column:
                var = f"{arg_lk[nan_check_column]}[{subscript}]"
                body.append(f'if not isnull({var}):')
                body.append(f'    {func_name}(x, y, {", ".join(args)})')
            else:
                body.append(f'{func_name}(x, y, {", ".join(args)})')

        if local_cuda_mutex:
            body.append(get_cuda_mutex_call(False))

        prev_local_cuda_mutex = local_cuda_mutex

    body = head + ['{0} = {1}[y, x]'.format(name, arg_lk[agg])
                   for agg, name in local_lk.items()] + body

    if global_cuda_mutex:
        body = [get_cuda_mutex_call(True)] + body + [get_cuda_mutex_call(False)]

    if antialias:
        signature = ["aa_factor", "prev_aa_factor"] + signature

    if ndims is None:
        code = ('def append(x, y, {0}):\n'
                '    {1}').format(', '.join(signature), '\n    '.join(body))
    else:
        code = ('def append({0}, x, y, {1}):\n'
                '    {2}'
                ).format(subscript, ', '.join(signature), '\n    '.join(body))
    logger.debug(code)

    # --- disk-backed append emission with pre-binding of globals ---

    # Bind every callable that the generated code refers to (the per-reduction
    # kernels you stored into `namespace` with unique names).
    bindings = {k: v for k, v in namespace.items() if callable(v)}

    # Bind helpers that may be referenced by the generated code.
    if need_isnull:
        bindings["isnull"] = isnull
    if any_uses_cuda_mutex:
        bindings["cuda_mutex_lock"] = cuda_mutex_lock
        bindings["cuda_mutex_unlock"] = cuda_mutex_unlock

    # Build a spec that changes whenever codegen or any bound closure meaningfully changes.
    # Include glyph dimensionality, mutex usage, antialias flag, and bytecode of the
    # per-reduction callables (from `calls`).
    funcs_in_order = [call[0] for call in calls]
    func_codes = [
        getattr(f, "__code__", None).co_code if hasattr(f, "__code__") else str(f)
        for f in funcs_in_order
    ]
    spec_parts = [
        "append",
        glyph.ndims,
        bool(any_uses_cuda_mutex),
        bool(need_isnull),
        bool(antialias),
        tuple(func_codes),
        ("versions", nb.__version__, ds.__version__),
    ]

    # Emit as a real module so Numba can persist compiled specializations to disk.
    # `_cache_emit_njit` expects a list of source lines where the first line is `def append(...):`.
    lines = code.split("\n")
    append_fn = _cache_emit_njit(
        "append",
        lines,
        spec_parts,
        bindings=bindings,
    )

    return append_fn, any_uses_cuda_mutex


def make_combine(bases, dshapes, temps, combine_temps, antialias, cuda, partitioned):
    # Lookup of base Reduction to argument index.
    arg_lk = dict((k, v) for (v, k) in enumerate(bases))
    # Also need lookup of by.reduction as the contained reduction is not aware of its wrapper.
    arg_lk.update(dict((k.reduction, v) for (v, k) in enumerate(bases) if isinstance(k, by)))

    # where._combine() deals with combine of preceding reduction so exclude
    # it from explicit combine calls.
    base_is_where = [b.is_where() for b in bases]
    next_base_is_where = base_is_where[1:] + [False]
    calls = [(None if n else b._build_combine(d, antialias, cuda, partitioned),
              [arg_lk[i] for i in (b,) + t + ct])
             for (b, d, t, ct, n) in zip(bases, dshapes, temps, combine_temps, next_base_is_where)]

    def combine(base_tuples):
        bases = tuple(np.stack(bs) for bs in zip(*base_tuples))
        ret = []
        for is_where, (func, inds) in zip(base_is_where, calls):
            if func is None:
                continue
            call = func(*get(inds, bases))
            if is_where:
                # Separate aggs of where reduction and its selector,
                # selector's goes first to match order of bases.
                ret.extend(call[::-1])
            else:
                ret.append(call)
        return tuple(ret)

    return combine


def make_finalize(bases, agg, schema, cuda, partitioned):
    arg_lk = dict((k, v) for (v, k) in enumerate(bases))
    if isinstance(agg, summary):
        calls = []
        for key, val in zip(agg.keys, agg.values):
            f = make_finalize(bases, val, schema, cuda, partitioned)
            try:
                # Override bases if possible
                bases = val._build_bases(cuda, partitioned)
            except AttributeError:
                pass
            inds = [arg_lk[b] for b in bases]
            calls.append((key, f, inds))

        def finalize(bases, cuda=False, **kwargs):
            data = {key: finalizer(get(inds, bases), cuda, **kwargs)
                    for (key, finalizer, inds) in calls}

            # Copy x and y range attrs from any DataArray (their ranges are all the same)
            # to set on parent Dataset
            name = agg.keys[0]  # Name of first DataArray.
            attrs = {attr: data[name].attrs[attr] for attr in ('x_range', 'y_range')}

            return xr.Dataset(data, attrs=attrs)
        return finalize
    else:
        return agg._build_finalize(schema)


def make_antialias_stage_2(reds, bases):
    # Only called if antialias is True.

    # Prefer a single-stage antialiased aggregation, but if any requested
    # reduction requires two stages then force use of two for all reductions.
    self_intersect = True
    for red in reds:
        if red._antialias_requires_2_stages():
            self_intersect = False
            break

    def antialias_stage_2(array_module) -> UnzippedAntialiasStage2:
        return tuple(zip(*concat(b._antialias_stage_2(self_intersect, array_module)
                                 for b in bases)))

    return self_intersect, antialias_stage_2
