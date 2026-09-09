from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .constants import GGML_QUANT_SIZES, GGMLQuantizationType


class GGMLLibraryError(Exception): ...


LIB_ENV_VAR = "GGML_LIB"
NTHREADS_ENV_VAR = "GGML_NTHREADS"

_LIB_NAMES = ("libggml-base.so", "libggml-base.dylib", "ggml-base.dll")
_LIB_BUILD_DIR = Path(__file__).resolve().parents[2] / "build" / "bin"

_lib: ctypes.CDLL | None = None
_pool: ThreadPoolExecutor | None = None
_nthreads: int = 0


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,                   # type
        ctypes.POINTER(ctypes.c_float), # src
        ctypes.c_void_p,                # dst
        ctypes.c_int64,                 # start
        ctypes.c_int64,                 # nrows
        ctypes.c_int64,                 # n_per_row
        ctypes.POINTER(ctypes.c_float), # imatrix
    ]
    lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
    lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
    return lib


def library() -> ctypes.CDLL:
    global _lib

    if _lib is None:
        paths = [os.environ[LIB_ENV_VAR]] if LIB_ENV_VAR in os.environ else [
            *(str(_LIB_BUILD_DIR / name) for name in _LIB_NAMES),
            *_LIB_NAMES,
        ]
        for path in paths:
            try:
                _lib = _bind(ctypes.CDLL(path))
                break
            except (OSError, AttributeError):
                continue
        else:
            raise GGMLLibraryError(
                f"cannot load the ggml shared library, build llama.cpp or set {LIB_ENV_VAR} to its path, tried: {', '.join(paths)}"
            )

    return _lib


def _threads() -> tuple[ThreadPoolExecutor, int]:
    global _pool, _nthreads

    if _pool is None:
        _nthreads = int(os.environ.get(NTHREADS_ENV_VAR, "0")) or os.cpu_count() or 1
        _pool = ThreadPoolExecutor(max_workers=_nthreads)

    return _pool, _nthreads


def quantize(qtype: GGMLQuantizationType, rows: np.ndarray) -> np.ndarray:
    """Encode f32 rows with the ggml reference quantizer, the last axis of the array is a row."""
    lib = library()

    if lib.ggml_quantize_requires_imatrix(int(qtype)):
        raise GGMLLibraryError(f"{qtype.name} requires an importance matrix")

    block_size, type_size = GGML_QUANT_SIZES[qtype]
    n_per_row = rows.shape[-1]

    if n_per_row % block_size != 0:
        raise ValueError(f"row size ({n_per_row}) is not a multiple of the {qtype.name} block size ({block_size})")

    src = np.ascontiguousarray(rows, dtype=np.float32)
    nrows = src.size // n_per_row
    dst = np.empty((nrows, n_per_row // block_size * type_size), dtype=np.uint8)

    src_p = src.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    dst_p = ctypes.c_void_p(dst.ctypes.data)

    pool, nthreads = _threads()
    n_chunks = min(nthreads, nrows)

    # rows are independent, and ggml_quantize_chunk offsets dst from start on its own
    if n_chunks > 1:
        bounds = [(i * nrows) // n_chunks for i in range(n_chunks + 1)]
        tasks = [
            pool.submit(
                lib.ggml_quantize_chunk, int(qtype), src_p, dst_p,
                bounds[i] * n_per_row, bounds[i + 1] - bounds[i], n_per_row, None,
            )
            for i in range(n_chunks)
        ]
        for task in tasks:
            task.result()
    else:
        lib.ggml_quantize_chunk(int(qtype), src_p, dst_p, 0, nrows, n_per_row, None)

    return dst.reshape((*src.shape[:-1], dst.shape[-1]))
