# MATH.md — derivations behind py-hoodscanR

Every non-trivial rewrite in this port carries an admissibility proof here, per
[ACCELERATION_PLAYBOOK](https://github.com/omicverse/omicverse-rebuildr/blob/main/ACCELERATION_PLAYBOOK.md).
Proof types: **(1)** exact algebraic identity, **(2)** bounded ε-approximation
with a derived constant, **(3)** class-containment theorem.

---

## 1. The algorithm

For cell *i* with sorted k-nearest-neighbour distances `d_i1 … d_ik`:

```
P_ij = exp(-d_ij² / τ) / Σ_{j'} exp(-d_ij'² / τ)          (softmax on squared distance)
```

`proximityFocused`: `τ = median(d²)/5`.
`smoothFadeout`: `τ = argmin_t  −Σ_ij log(P_ij(t) + 10⁻⁸)`, by BFGS.

Collapsed onto cell types via a label indicator `M ∈ {0,1}^{n×k×G}`:

```
H_ig = Σ_j M_ijg · P_ij ,     Σ_g H_ig = 1
```

and then

```
entropy_i    = −Σ_g H_ig log₂ H_ig            (over H_ig > 0)
perplexity_i = 2^{entropy_i}
```

---

## 2. Permutation test: row-permutation equivariance — **type (1), exact**

R (`calc_metrics.R`) runs, for each of `n_perm` permutations `π`:

```r
permuted_matrix <- pm[sample(1:nrow(pm)), ]
permuted_perplexities[, i] <- .get_perplexity(permuted_matrix)
```

**Claim.** Let `f : R^G → R` be any function applied row-wise, and let `Π_π` be
the row-permutation operator. Then `f_row(Π_π X) = Π_π f_row(X)`.

*Proof.* `(Π_π X)_{i·} = X_{π(i)·}` by definition, so
`[f_row(Π_π X)]_i = f(X_{π(i)·}) = [f_row(X)]_{π(i)} = [Π_π f_row(X)]_i`. ∎

Perplexity is applied row-wise, so the recomputation inside R's loop is
redundant: `perplexity(pm[π, :]) ≡ perplexity(pm)[π]`. The port computes the
observed perplexity once and indexes it. Bit-exact — no floating-point
operation is reassociated, the same `double` values are compared.

A second, independent saving: only

```
counts_i = #{π : perplexity_{π(i)} ≤ perplexity_i}
```

is needed for the p-value, so the `n × n_perm` matrix R allocates
(`matrix(NA, nrow(pm), n_perm)`, 2661 × 1000 doubles = 21 MB) is never formed.
Memory drops from `O(n · n_perm)` to `O(n)`.

Measured: 15.6 s → 0.115 s, parity vector unchanged (Pearson r = 1.0 vs R,
and element-wise identical).

## 2b. Compiling R's RNG — **type (1), exact**

`pyhoodscanr/_rrng_fast.py` is the same integer arithmetic as
`pyhoodscanr/_rrng.py`, JIT-compiled. Mersenne-Twister state transitions are
exact `uint32` operations and `unif_rand` is one `uint32 × 2⁻³²` product; no
reassociation is possible. `tests/test_rrng.py` asserts stream equality for
several seeds and both draw sizes.

---

## 3. `merge_by_group`: masking and long-double order — **type (1), exact**

R computes `rowSums(1 * (group_df == g) * pm)`, i.e. it sums *all* `k` entries
per row, with non-matching entries replaced by `0.0`, accumulating in a
long-double register.

**Claim.** Adding `+0.0` to an IEEE-754 accumulator is exact for any finite
accumulator value, so dropping the zero terms cannot change the result **as
long as the order of the non-zero terms is preserved**.

*Proof.* For `a` finite and not `−0.0`, `fl(a + 0.0) = a` exactly (IEEE-754
§6.3: the sum of a finite value and zero is that value, with round-to-nearest
never invoked). Induction over the accumulation gives the claim. ∎

The port therefore keeps the same column order, keeps the long-double
accumulator, and only changes *how the label codes are computed*
(`pandas.factorize(sort=True)` hash pass instead of `np.unique`'s full sort of
all `n·k` strings). Measured: identical to the last bit (`max |Δ| = 0.0`),
0.088 s → 0.016 s.

---

## 4. Where the port is **not** bit-identical to R

Two places, both documented and both bounded.

### 4.1 k-NN ties at the k-th boundary — inherent ambiguity, not an error

`RANN::nn2` (ANN priority search) and `sklearn.NearestNeighbors` both return
*exactly k* neighbours. When several candidates are at **exactly** the same
f64 distance, which of them occupies the k-th slot is decided by the internal
heap/priority-queue order, which is an implementation artefact — it is not part
of the algorithm's specification, and the two libraries disagree.

Two consequences, with very different severity:

* **Reordering ties inside the neighbourhood is harmless.** On the canonical
  fixture 94 of 2661 cells have some pair of equidistant neighbours swapped.
  This changes nothing downstream: `P_ij` depends on `d_ij` only, so equal
  distances carry equal weights, and `H_ig = Σ_j M_ijg P_ij` is invariant to
  permuting equal-weight terms with equal labels — and for terms with
  *different* labels the two weights are equal, so the per-label sums still
  agree to the last bit. Measured contribution: **0**.

* **A tie straddling the k-th slot changes the neighbourhood's membership.**
  2 of 2661 cells (0.075%) are affected. The competing cells have different
  cell types, so `H_i·` shifts by the softmax weight of the k-th neighbour.

  Bound: the k-th neighbour is the farthest, so its weight is the smallest,
  `w_k = exp(−d_ik²/τ) / Σ_j exp(−d_ij²/τ)`, and the total-variation shift is
  at most `w_k` (mass `w_k` moves from one label to another, TV = ½·2·w_k).
  Measured `max TV = 4.27 × 10⁻⁵` over the whole fixture; every unaffected cell
  agrees with R to `≤ 1.17 × 10⁻¹⁵`.

  Propagated to the colocalisation matrix (a Pearson correlation over all 2661
  cells) this contributes `8.48 × 10⁻⁸`. Recomputing the same correlation with
  the two tied cells removed gives `2.2 × 10⁻¹⁶` — i.e. the entire residual is
  attributable to those two cells.

`find_near_cells(tie_break="stable")` (the default) sorts by
`(distance, cell index)`, which makes py-hoodscanR's own output deterministic
and independent of the search backend and the platform — a guarantee neither
`RANN` nor raw scikit-learn provides. `tie_break="backend"` keeps
scikit-learn's raw order (1 affected cell on this fixture instead of 2). The
default was chosen for reproducibility, **not** because it minimises the
parity residual; it does not.

### 4.2 Locale-dependent label ordering

R's `sort()` on a character vector uses the collation of the active locale.
`pandas.factorize(sort=True)` and Python's `sorted()` order by Unicode code
point (equivalent to R under `LC_COLLATE=C`). For ASCII alphanumeric cell-type
labels — every case we are aware of — the two agree. Non-ASCII labels under a
non-C locale could order the *columns* of the neighbourhood matrix differently;
the values are unaffected.

---

## 5. Things deliberately **not** optimised

* **The softmax is computed without a row-max shift.** The numerically stabler
  `exp(x − max x)` form changes the result in the last bits, and
  `d²/τ = 5d²/median(d²)` is O(1) by construction, so there is nothing to
  protect against. Fidelity beats a non-problem.
* **`stats::optim`'s finite-difference gradient is kept**, even though the
  analytic gradient of the NLL is available in closed form. R probes with an
  *absolute* step of `1e-3` on a parameter of order `1e5`; substituting the
  analytic gradient lands on a different `τ`. The port reproduces R's `τ` to
  `1.1 × 10⁻¹⁵` relative precisely because it does not "improve" this.
* **`np.longdouble` accumulators are kept** in `scan_hoods`, `merge_by_group`
  and `r_cor`. They cost ~2x on those steps and are the reason the fitted `τ`
  and the correlation matrix reproduce; see `pyhoodscanr/_rmath.py`.
