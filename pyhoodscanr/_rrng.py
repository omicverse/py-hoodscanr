"""Bit-faithful port of R's default random-number generator.

Why this file exists
--------------------
Two hoodscanR functions consume R's RNG:

* ``perplexityPermute`` -> ``sample(1:nrow(pm))``
* ``clustByHood``       -> ``stats::kmeans(..., nstart = 5)`` -> ``sample.int``

NumPy's ``MT19937`` is the *same* Mersenne-Twister core but R seeds it
differently (a 50-round LCG scramble followed by a second LCG fill of the
624-word state) and draws integers differently (R >= 3.6 uses *rejection*
sampling on whole bits, not ``floor(n * unif)``).  Using NumPy directly would
therefore give a different stream and downgrade the parity gate for
``perplexity_p`` and ``clusters`` from exact to distributional.

Porting R's RNG upgrades both to **bit-exact**.

Reference: R 4.4.3 ``src/main/RNG.c`` (``RNG_Init``, ``MT_genrand``,
``fixup``, ``R_unif_index``, ``rbits``) and ``src/main/random.c``
(``SampleNoReplace``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["RRandom"]

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF
_TEMPERING_MASK_B = 0x9D2C5680
_TEMPERING_MASK_C = 0xEFC60000
_MASK32 = 0xFFFFFFFF

# RNG.c: i2_32m1 = 2.328306437080797e-10 = 1/(2^32 - 1)
_I2_32M1 = 2.328306437080797e-10


class RRandom:
    """R's Mersenne-Twister, seeded exactly as ``set.seed(seed)`` does.

    Parameters
    ----------
    seed
        Integer passed to R's ``set.seed``.

    Examples
    --------
    >>> rng = RRandom(42)
    >>> round(rng.unif_rand(), 7)          # R: set.seed(42); runif(1)
    0.9148060
    """

    def __init__(self, seed: int):
        self.set_seed(seed)

    # ------------------------------------------------------------------
    # seeding
    # ------------------------------------------------------------------
    def set_seed(self, seed: int) -> None:
        """Replicate ``do_setseed`` + ``RNG_Init(MERSENNE_TWISTER, seed)``."""
        s = int(seed) & _MASK32
        # RNG.c do_setseed: "Initial scrambling"
        for _ in range(50):
            s = (69069 * s + 1) & _MASK32
        # RNG_Init: fill n_seed = 1 + 624 words
        i_seed = np.empty(_N + 1, dtype=np.uint32)
        for j in range(_N + 1):
            s = (69069 * s + 1) & _MASK32
            i_seed[j] = s
        # FixupSeeds(MERSENNE_TWISTER, initial=1): I624 = 624
        i_seed[0] = _N
        self._i_seed = i_seed

    # ------------------------------------------------------------------
    # core generator
    # ------------------------------------------------------------------
    def _mt_genrand(self) -> int:
        """RNG.c ``MT_genrand`` -> 32-bit tempered word."""
        i_seed = self._i_seed
        mti = int(i_seed[0])
        mt = i_seed[1:]  # view: dummy + 1

        if mti >= _N:
            # generate N words at one time
            for kk in range(_N - _M):
                y = (int(mt[kk]) & _UPPER_MASK) | (int(mt[kk + 1]) & _LOWER_MASK)
                mt[kk] = (int(mt[kk + _M]) ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)) & _MASK32
            for kk in range(_N - _M, _N - 1):
                y = (int(mt[kk]) & _UPPER_MASK) | (int(mt[kk + 1]) & _LOWER_MASK)
                mt[kk] = (
                    int(mt[kk + (_M - _N)]) ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)
                ) & _MASK32
            y = (int(mt[_N - 1]) & _UPPER_MASK) | (int(mt[0]) & _LOWER_MASK)
            mt[_N - 1] = (int(mt[_M - 1]) ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)) & _MASK32
            mti = 0

        y = int(mt[mti])
        mti += 1
        y ^= y >> 11
        y ^= (y << 7) & _TEMPERING_MASK_B
        y &= _MASK32
        y ^= (y << 15) & _TEMPERING_MASK_C
        y &= _MASK32
        y ^= y >> 18

        i_seed[0] = mti
        return y

    def unif_rand(self) -> float:
        """RNG.c ``unif_rand`` for MERSENNE_TWISTER, including ``fixup``."""
        x = self._mt_genrand() * 2.3283064365386963e-10
        # fixup(): never return exactly 0 or 1
        if x <= 0.0:
            return 0.5 * _I2_32M1
        if (1.0 - x) <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return x

    # ------------------------------------------------------------------
    # integer draws (R >= 3.6 "Rejection" sample.kind)
    # ------------------------------------------------------------------
    def _rbits(self, bits: int) -> float:
        """RNG.c ``rbits``: build ``bits`` random bits, 16 at a time."""
        v = 0
        n = 0
        while n <= bits:
            v1 = int(np.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
            n += 16
        return float(v & ((1 << bits) - 1))

    def unif_index(self, dn: float) -> float:
        """RNG.c ``R_unif_index`` — rejection sampling on ``[0, dn)``."""
        if dn <= 0:
            return 0.0
        bits = int(np.ceil(np.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample_int(self, n: int, size: int | None = None) -> np.ndarray:
        """``sample.int(n, size)`` without replacement.

        Mirrors ``random.c::SampleNoReplace``.  Returns **0-based** indices
        (R returns 1-based; the caller subtracts 1 anyway).
        """
        if size is None:
            size = n
        x = np.arange(n, dtype=np.int64)
        y = np.empty(size, dtype=np.int64)
        nn = n
        for i in range(size):
            j = int(self.unif_index(float(nn)))
            y[i] = x[j]
            nn -= 1
            x[j] = x[nn]
        return y
