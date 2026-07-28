"""matplotlib/seaborn replacements for hoodscanR's ggplot2 + ComplexHeatmap plots.

Mirror of ``R/plot_tissue.R``, ``R/plot_pm.R``, ``R/plot_pd.R``, ``R/plot_heatmap.R``.
Only the numeric part of ``plotColocal`` is under the parity gate (via
``return_matrix=True``); the rendering itself is idiomatic Python.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["plot_tissue", "plot_hood_mat", "plot_prob_dist", "plot_colocal"]


def _obs_pm(adata, pm_cols):
    if pm_cols is None:
        pm_cols = adata.uns.get("hoodscanr", {}).get("pm_cols")
    if pm_cols is None:
        raise ValueError("The pm_cols are not included in the AnnData.")
    return list(pm_cols)


def plot_tissue(adata, color: str = "cell_annotation", basis: str = "spatial",
                size: float = 1.5, alpha: float = 0.8, cmap: str = "viridis",
                palette=None, ax=None, legend: bool = True, title=None):
    """``hoodscanR::plotTissue`` — scatter cells in tissue space.

    ``color`` may be any ``obs`` column: the annotation, ``entropy``,
    ``perplexity``, ``perplexity_p`` or ``clusters``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    xy = np.asarray(adata.obsm[basis], dtype=float)
    vals = adata.obs[color]

    if pd.api.types.is_numeric_dtype(vals):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals.to_numpy(), s=size, alpha=alpha,
                        cmap=cmap, linewidths=0)
        if legend:
            ax.figure.colorbar(sc, ax=ax, label=color, fraction=0.03, pad=0.02)
    else:
        cats = pd.Categorical(vals)
        levels = list(cats.categories)
        if palette is None:
            base = plt.get_cmap("tab10" if len(levels) <= 10 else "tab20")
            palette = [base(i % base.N) for i in range(len(levels))]
        for i, lev in enumerate(levels):
            sel = cats.codes == i
            ax.scatter(xy[sel, 0], xy[sel, 1], s=size, alpha=alpha,
                       color=palette[i], label=str(lev), linewidths=0)
        if legend:
            ax.legend(markerscale=6, fontsize=8, frameon=False,
                      loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title if title is not None else color)
    return ax


def plot_hood_mat(hoods, n: int = 10, target_cells=None, seed: int = 42,
                  cmap: str = "magma", ax=None, title="Neighbourhood probability"):
    """``hoodscanR::plotHoodMat`` — heatmap of per-cell neighbourhood probabilities."""
    import matplotlib.pyplot as plt

    hoods = pd.DataFrame(hoods)
    if target_cells is not None:
        sub = hoods.loc[list(target_cells)]
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(hoods.shape[0], size=min(n, hoods.shape[0]), replace=False)
        sub = hoods.iloc[np.sort(idx)]

    if ax is None:
        _, ax = plt.subplots(figsize=(1.0 * sub.shape[1] + 2, 0.35 * sub.shape[0] + 1.5))
    im = ax.imshow(sub.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(sub.shape[1]))
    ax.set_xticklabels(sub.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(sub.shape[0]))
    ax.set_yticklabels(sub.index, fontsize=7)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if v < 0.6 else "black")
    ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(title)
    return ax


def plot_prob_dist(adata, pm_cols=None, by_cluster: bool = False,
                   val_name: str = "clusters", show_clusters=None,
                   target_cells=None, plot_all: bool = False,
                   sample_size: int = 2, seed: int = 42, ncol: int = 4):
    """``hoodscanR::plotProbDist`` — per-cell (or per-cluster) probability profile."""
    import matplotlib.pyplot as plt

    pm_cols = _obs_pm(adata, pm_cols)
    dat = adata.obs.loc[:, pm_cols]

    if by_cluster:
        if val_name not in adata.obs.columns:
            raise ValueError(f"Cannot find '{val_name}' in adata.obs.")
        clusters = adata.obs[val_name].astype(str)
        levels = (
            [str(c) for c in show_clusters]
            if show_clusters is not None
            else sorted(clusters.unique(), key=lambda s: (len(s), s))
        )
        fig, axes = plt.subplots(
            int(np.ceil(len(levels) / ncol)), ncol,
            figsize=(3.2 * ncol, 2.4 * int(np.ceil(len(levels) / ncol))),
            squeeze=False,
        )
        for ax, lev in zip(axes.ravel(), levels):
            sub = dat.loc[clusters == lev]
            if plot_all:
                ax.boxplot([sub[c].to_numpy() for c in pm_cols], labels=pm_cols,
                           showfliers=False)
            else:
                rng = np.random.default_rng(seed)
                take = rng.choice(sub.shape[0], size=min(sample_size, sub.shape[0]),
                                  replace=False)
                for t in take:
                    ax.plot(range(len(pm_cols)), sub.iloc[t].to_numpy(), marker="o")
                ax.set_xticks(range(len(pm_cols)))
                ax.set_xticklabels(pm_cols)
            ax.set_title(f"{val_name} {lev}", fontsize=9)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=60, labelsize=7)
        for ax in axes.ravel()[len(levels):]:
            ax.axis("off")
        fig.tight_layout()
        return axes

    if target_cells is None:
        target_cells = list(adata.obs_names[:6])
    fig, axes = plt.subplots(
        int(np.ceil(len(target_cells) / ncol)), ncol,
        figsize=(3.2 * ncol, 2.4 * int(np.ceil(len(target_cells) / ncol))),
        squeeze=False,
    )
    for ax, cell in zip(axes.ravel(), target_cells):
        ax.bar(range(len(pm_cols)), dat.loc[cell].to_numpy())
        ax.set_xticks(range(len(pm_cols)))
        ax.set_xticklabels(pm_cols, rotation=60, fontsize=7, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(str(cell), fontsize=8)
    for ax in axes.ravel()[len(target_cells):]:
        ax.axis("off")
    fig.tight_layout()
    return axes


def plot_colocal(adata, pm_cols=None, self_cor: bool = True, by_group=None,
                 return_matrix: bool = False, ax=None, cmap: str = "RdBu_r",
                 title=None, annot: bool = True):
    """``hoodscanR::plotColocal`` — neighbourhood colocalisation heatmap.

    ``self_cor=True`` (default) returns/plots the Pearson correlation **between
    neighbourhood columns** — two neighbourhood types that co-occur around the
    same cells are positively correlated.  ``self_cor=False`` returns the mean
    probability of each neighbourhood within each ``by_group`` level.
    """
    import matplotlib.pyplot as plt

    pm_cols = _obs_pm(adata, pm_cols)
    dat = adata.obs.loc[:, pm_cols].astype(float)

    if self_cor:
        from ._rmath import r_cor

        mat = pd.DataFrame(r_cor(dat.to_numpy()), index=dat.columns, columns=dat.columns)
        default_title = "Neighbourhood colocalisation (Pearson r)"
        vmin, vmax = -1.0, 1.0
    else:
        if by_group is None:
            raise ValueError("by_group is required when self_cor=False.")
        from ._rmath import r_mean

        g = adata.obs[by_group].astype(str).to_numpy()
        levels = sorted(set(g.tolist()))  # R: aggregate() sorts the group levels
        arr = dat.to_numpy()
        mat = pd.DataFrame(
            np.vstack([r_mean(arr[g == lev], axis=0) for lev in levels]),
            index=levels, columns=dat.columns,
        )
        default_title = "Mean probability within groups"
        vmin, vmax = 0.0, float(np.nanmax(mat.to_numpy()))

    if return_matrix:
        return mat

    if ax is None:
        _, ax = plt.subplots(figsize=(0.8 * mat.shape[1] + 3, 0.6 * mat.shape[0] + 2))
    im = ax.imshow(mat.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    if annot:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat.iat[i, j]:.2f}", ha="center", va="center",
                        fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    ax.set_title(title if title is not None else default_title)
    return ax
