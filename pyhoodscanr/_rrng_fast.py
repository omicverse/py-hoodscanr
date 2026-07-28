"""numba kernels for R's RNG.

Same arithmetic as :mod:`pyhoodscanr._rrng`, compiled.  The pure-Python class
stays as the readable reference; ``tests/test_rrng.py`` asserts the two produce
bit-identical streams, and the parity gate is re-run against R after the swap.

Admissibility: **exact algebraic identity** — identical integer/f64 operations
in a different execution engine.  No reassociation, no approximation.
See ``ACCELERATION_PLAYBOOK`` proof type 1 and ``MATH.md`` section 2.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(f):
            return f

        if args and callable(args[0]):
            return args[0]
        return wrap


_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF
_TB = 0x9D2C5680
_TC = 0xEFC60000
_MASK32 = 0xFFFFFFFF
_I2_32M1 = 2.328306437080797e-10


@njit(cache=True, inline="always")
def _mt_genrand(state):
    """``state`` is an int64[625] array: ``state[0] = mti``, ``state[1:] = mt``."""
    mti = state[0]
    if mti >= _N:
        for kk in range(_N - _M):
            y = (state[1 + kk] & _UPPER_MASK) | (state[2 + kk] & _LOWER_MASK)
            mag = _MATRIX_A if (y & 1) else 0
            state[1 + kk] = (state[1 + kk + _M] ^ (y >> 1) ^ mag) & _MASK32
        for kk in range(_N - _M, _N - 1):
            y = (state[1 + kk] & _UPPER_MASK) | (state[2 + kk] & _LOWER_MASK)
            mag = _MATRIX_A if (y & 1) else 0
            state[1 + kk] = (state[1 + kk + _M - _N] ^ (y >> 1) ^ mag) & _MASK32
        y = (state[_N] & _UPPER_MASK) | (state[1] & _LOWER_MASK)
        mag = _MATRIX_A if (y & 1) else 0
        state[_N] = (state[_M] ^ (y >> 1) ^ mag) & _MASK32
        mti = 0
    y = state[1 + mti]
    mti += 1
    y ^= y >> 11
    y = (y ^ ((y << 7) & _TB)) & _MASK32
    y = (y ^ ((y << 15) & _TC)) & _MASK32
    y ^= y >> 18
    state[0] = mti
    return y


@njit(cache=True, inline="always")
def _unif_rand(state):
    x = _mt_genrand(state) * 2.3283064365386963e-10
    if x <= 0.0:
        return 0.5 * _I2_32M1
    if (1.0 - x) <= 0.0:
        return 1.0 - 0.5 * _I2_32M1
    return x


@njit(cache=True, inline="always")
def _unif_index(state, dn, bits):
    while True:
        v = 0
        n = 0
        while n <= bits:
            v1 = int(np.floor(_unif_rand(state) * 65536.0))
            v = 65536 * v + v1
            n += 16
        dv = float(v & ((1 << bits) - 1))
        if dn > dv:
            return dv


@njit(cache=True)
def sample_int_kernel(state, n, size):
    """``random.c::SampleNoReplace``, 0-based."""
    x = np.empty(n, dtype=np.int64)
    for i in range(n):
        x[i] = i
    y = np.empty(size, dtype=np.int64)
    nn = n
    for i in range(size):
        bits = int(np.ceil(np.log2(float(nn)))) if nn > 1 else 0
        j = int(_unif_index(state, float(nn), bits))
        y[i] = x[j]
        nn -= 1
        x[j] = x[nn]
    return y


@njit(cache=True)
def perm_le_counts(state, observed, n_perm):
    """Counts for ``perplexityPermute`` without materialising the permutations.

    ``counts[i] = #{p : observed[perm_p[i]] <= observed[i]}``.

    Two exact rewrites are folded in here (see ``MATH.md``):

    1. ``perplexity(pm[perm, :]) == perplexity(pm)[perm]`` — a row-wise map
       commutes with a row permutation, so the inner entropy recomputation
       R performs ``n_perm`` times is redundant.
    2. Only the comparison count is needed, so the ``n x n_perm`` matrix R
       allocates is never formed (memory O(n) instead of O(n * n_perm)).
    """
    n = observed.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    x = np.empty(n, dtype=np.int64)
    for _ in range(n_perm):
        for i in range(n):
            x[i] = i
        nn = n
        for i in range(n):
            bits = int(np.ceil(np.log2(float(nn)))) if nn > 1 else 0
            j = int(_unif_index(state, float(nn), bits))
            perm_i = x[j]
            nn -= 1
            x[j] = x[nn]
            if observed[perm_i] <= observed[i]:
                counts[i] += 1
    return counts


def state_from(rng) -> np.ndarray:
    """Export an :class:`~pyhoodscanr._rrng.RRandom` state as int64[625]."""
    return rng._i_seed.astype(np.int64).copy()


def state_into(rng, state: np.ndarray) -> None:
    """Write a kernel state back into an :class:`~pyhoodscanr._rrng.RRandom`."""
    rng._i_seed = state.astype(np.uint32)
