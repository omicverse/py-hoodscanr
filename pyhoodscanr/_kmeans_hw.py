"""Hartigan-Wong k-means (Applied Statistics algorithm AS 136).

Why this file exists
--------------------
``clustByHood`` calls ``stats::kmeans(..., algorithm = "Hartigan-Wong")``,
which is R's default.  scikit-learn implements only Lloyd and Elkan.  These are
different local searches: Lloyd only moves a point when its *nearest centroid*
changes, whereas Hartigan-Wong moves a point whenever doing so reduces the
total within-cluster sum of squares *after accounting for the centroid shift*
(the ``n/(n-1)`` and ``n/(n+1)`` correction factors below).  Hartigan-Wong
therefore escapes local optima Lloyd gets stuck in, and substituting one for
the other would silently change the clustering.

This is a line-by-line port of R 4.4.3 ``src/library/stats/src/kmns.c``
(itself a translation of Hartigan & Wong 1979, JRSS-C 28(1):100-108), which
uses ``double`` throughout.  The optimal-transfer and quick-transfer inner
loops are JIT-compiled with numba when available.

Combined with :mod:`pyhoodscanr._rrng` (which reproduces R's ``sample.int``
for the ``nstart`` initialisations), the result is bit-identical to R's
``kmeans`` on the same input.
"""

from __future__ import annotations

import numpy as np

__all__ = ["kmeans_hartigan_wong", "r_kmeans"]

_BIG = 1.0e30

try:  # pragma: no cover - availability depends on environment
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(f):
            return f

        if args and callable(args[0]):
            return args[0]
        return wrap


@njit(cache=True)
def _optra(a, m, n, c, k, ic1, ic2, nc, an1, an2, ncp, d, itran, live, indx):
    """AS 136 ``OPTRA`` — optimal-transfer stage."""
    for l in range(k):
        if itran[l] == 1:
            live[l] = m + 1

    for i in range(m):
        indx += 1
        l1 = ic1[i]
        l2 = ic2[i]
        ll = l2

        if nc[l1] != 1:
            # If L1 has not yet been updated in this stage, no need to
            # re-compute D(I).
            if ncp[l1] != 0:
                de = 0.0
                for j in range(n):
                    df = a[i, j] - c[l1, j]
                    de += df * df
                d[i] = de * an1[l1]

            # Find the cluster with minimum R2.
            da = 0.0
            for j in range(n):
                db = a[i, j] - c[l2, j]
                da += db * db
            r2 = da * an2[l2]

            for l in range(k):
                if ((i + 1 >= live[l1] and i + 1 >= live[l]) or l == l1 or l == ll):
                    continue
                rr = r2 / an2[l]
                dc = 0.0
                skip = False
                for j in range(n):
                    dd = a[i, j] - c[l, j]
                    dc += dd * dd
                    if dc >= rr:
                        skip = True
                        break
                if skip:
                    continue
                r2 = dc * an2[l]
                l2 = l

            if r2 >= d[i]:
                # No transfer necessary; IC2(I) is the new IC2(I).
                ic2[i] = l2
            else:
                # Update centres, LIVE, NCP, AN1 & AN2 for L1 and L2.
                indx = 0
                live[l1] = m + i + 1
                live[l2] = m + i + 1
                ncp[l1] = i + 1
                ncp[l2] = i + 1
                al1 = float(nc[l1])
                alw = al1 - 1.0
                al2 = float(nc[l2])
                alt = al2 + 1.0
                for j in range(n):
                    c[l1, j] = (c[l1, j] * al1 - a[i, j]) / alw
                    c[l2, j] = (c[l2, j] * al2 + a[i, j]) / alt
                nc[l1] -= 1
                nc[l2] += 1
                an2[l1] = alw / al1
                an1[l1] = _BIG
                if alw > 1.0:
                    an1[l1] = alw / (alw - 1.0)
                an1[l2] = alt / al2
                an2[l2] = alt / (alt + 1.0)
                ic1[i] = l2
                ic2[i] = l1

        if indx == m:
            return indx

    for l in range(k):
        itran[l] = 0
        live[l] -= m
    return indx


@njit(cache=True)
def _qtran(a, m, n, c, k, ic1, ic2, nc, an1, an2, ncp, d, itran, indx, imaxqtr):
    """AS 136 ``QTRAN`` — quick-transfer stage.  Returns ``(indx, ifault)``."""
    icoun = 0
    istep = 0
    while True:
        for i in range(m):
            icoun += 1
            istep += 1
            if istep >= imaxqtr:
                return indx, 4
            l1 = ic1[i]
            l2 = ic2[i]

            if nc[l1] != 1:
                if istep <= ncp[l1]:
                    da = 0.0
                    for j in range(n):
                        db = a[i, j] - c[l1, j]
                        da += db * db
                    d[i] = da * an1[l1]

                if istep < ncp[l1] or istep < ncp[l2]:
                    r2 = d[i] / an2[l2]
                    dd = 0.0
                    skip = False
                    for j in range(n):
                        de = a[i, j] - c[l2, j]
                        dd += de * de
                        if dd >= r2:
                            skip = True
                            break
                    if not skip:
                        icoun = 0
                        indx = 0
                        itran[l1] = 1
                        itran[l2] = 1
                        ncp[l1] = istep + m
                        ncp[l2] = istep + m
                        al1 = float(nc[l1])
                        alw = al1 - 1.0
                        al2 = float(nc[l2])
                        alt = al2 + 1.0
                        for j in range(n):
                            c[l1, j] = (c[l1, j] * al1 - a[i, j]) / alw
                            c[l2, j] = (c[l2, j] * al2 + a[i, j]) / alt
                        nc[l1] -= 1
                        nc[l2] += 1
                        an2[l1] = alw / al1
                        an1[l1] = _BIG
                        if alw > 1.0:
                            an1[l1] = alw / (alw - 1.0)
                        an1[l2] = alt / al2
                        an2[l2] = alt / (alt + 1.0)
                        ic1[i] = l2
                        ic2[i] = l1

            if icoun == m:
                return indx, 0


def kmeans_hartigan_wong(a, centers, iter_max: int = 10):
    """AS 136 ``KMNS`` driver.

    Parameters
    ----------
    a
        ``m x n`` data matrix.
    centers
        ``k x n`` initial centres.
    iter_max
        Maximum number of optimal/quick transfer sweeps (R's ``iter.max``).

    Returns
    -------
    dict
        ``cluster`` (0-based labels), ``centers``, ``wss``, ``size``,
        ``iter``, ``ifault``.
    """
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    c = np.ascontiguousarray(np.asarray(centers, dtype=np.float64).copy())
    m, n = a.shape
    k = c.shape[0]
    if k <= 1 or k >= m:
        raise ValueError("need 1 < k < nrow(x)")

    ic1 = np.zeros(m, dtype=np.int64)
    ic2 = np.zeros(m, dtype=np.int64)
    nc = np.zeros(k, dtype=np.int64)
    an1 = np.zeros(k, dtype=np.float64)
    an2 = np.zeros(k, dtype=np.float64)
    ncp = np.zeros(k, dtype=np.int64)
    d = np.zeros(m, dtype=np.float64)
    itran = np.zeros(k, dtype=np.int64)
    live = np.zeros(k, dtype=np.int64)
    wss = np.zeros(k, dtype=np.float64)

    # --- step 1: two closest centres per point -------------------------
    d0 = ((a[:, None, :] - c[None, :2, :]) ** 2).sum(axis=2)  # m x 2
    swap = d0[:, 0] > d0[:, 1]
    ic1[:] = np.where(swap, 1, 0)
    ic2[:] = np.where(swap, 0, 1)
    dt1 = np.where(swap, d0[:, 1], d0[:, 0])
    dt2 = np.where(swap, d0[:, 0], d0[:, 1])
    for l in range(2, k):
        db = ((a - c[l]) ** 2).sum(axis=1)
        better1 = db < dt1
        better2 = (~better1) & (db < dt2)
        dt2 = np.where(better1, dt1, np.where(better2, db, dt2))
        ic2 = np.where(better1, ic1, np.where(better2, l, ic2))
        dt1 = np.where(better1, db, dt1)
        ic1 = np.where(better1, l, ic1)
    ic1 = np.ascontiguousarray(ic1.astype(np.int64))
    ic2 = np.ascontiguousarray(ic2.astype(np.int64))

    # --- step 2: centres = cluster means -------------------------------
    c[:] = 0.0
    np.add.at(c, ic1, a)
    nc[:] = np.bincount(ic1, minlength=k)
    if (nc == 0).any():
        return {"cluster": ic1, "centers": c, "wss": wss, "size": nc,
                "iter": 0, "ifault": 1}
    c /= nc[:, None].astype(np.float64)

    aa = nc.astype(np.float64)
    an2[:] = aa / (aa + 1.0)
    an1[:] = np.where(aa > 1.0, aa / np.maximum(aa - 1.0, 1e-300), _BIG)
    an1[aa <= 1.0] = _BIG
    itran[:] = 1
    ncp[:] = -1

    indx = 0
    ifault = 0
    imaxqtr = 50 * m  # R's cap on quick-transfer steps
    it = 0
    for ij in range(iter_max):
        it = ij + 1
        indx = _optra(a, m, n, c, k, ic1, ic2, nc, an1, an2, ncp, d,
                      itran, live, indx)
        if indx == m:
            break
        indx, qf = _qtran(a, m, n, c, k, ic1, ic2, nc, an1, an2, ncp, d,
                          itran, indx, imaxqtr)
        if qf == 4:
            ifault = 4
            break
        if k == 2:
            break
        ncp[:] = 0
    else:
        ifault = 2

    # --- within-cluster sum of squares ---------------------------------
    c[:] = 0.0
    np.add.at(c, ic1, a)
    nc[:] = np.bincount(ic1, minlength=k)
    c /= nc[:, None].astype(np.float64)
    diff = a - c[ic1]
    wss[:] = np.bincount(ic1, weights=(diff**2).sum(axis=1), minlength=k)

    return {"cluster": ic1, "centers": c, "wss": wss, "size": nc,
            "iter": it, "ifault": ifault}


def r_kmeans(x, k: int, iter_max: int = 10, nstart: int = 1, seed: int | None = None,
             rng=None):
    """``stats::kmeans(x, centers = k, algorithm = "Hartigan-Wong")``.

    Reproduces R's *initialisation* as well as its optimisation:

    * ``nstart == 1`` -> centres are ``x[sample.int(m, k), ]``
    * ``nstart >= 2`` -> centres are ``cn[sample.int(mm, k), ]`` where
      ``cn = unique(x)`` (first-occurrence order), drawn ``nstart`` times

    with ``sample.int`` supplied by :class:`pyhoodscanr._rrng.RRandom`, so
    ``seed`` behaves exactly like ``set.seed(seed)`` in R.
    """
    from ._rrng import RRandom
    from ._rrng_fast import HAVE_NUMBA, sample_int_kernel, state_from, state_into

    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    m = x.shape[0]
    if rng is None:
        rng = RRandom(42 if seed is None else seed)

    def _sample(n, size):
        # identical stream to rng.sample_int, compiled (MATH.md section 2)
        if not HAVE_NUMBA:
            return rng.sample_int(n, size)
        state = state_from(rng)
        out = sample_int_kernel(state, n, size)
        state_into(rng, state)
        return out

    cn = None
    if nstart == 1:
        centers = x[_sample(m, k)]
    if nstart >= 2:
        # R: cn <- unique(x)  (unique rows, first-occurrence order)
        _, first_idx = np.unique(x, axis=0, return_index=True)
        cn = x[np.sort(first_idx)]
        mm = cn.shape[0]
        if mm < k:
            raise ValueError("more cluster centers than distinct data points.")
        centers = cn[_sample(mm, k)]

    best = kmeans_hartigan_wong(x, centers, iter_max=iter_max)
    best_wss = best["wss"].sum()
    if nstart >= 2 and cn is not None:
        for _ in range(nstart - 1):
            centers = cn[_sample(cn.shape[0], k)]
            trial = kmeans_hartigan_wong(x, centers, iter_max=iter_max)
            z = trial["wss"].sum()
            if z < best_wss:
                best, best_wss = trial, z
    return best
