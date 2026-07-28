"""Fast "does it run at all" tests — no R, no reference artefacts needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyhoodscanr as ph


@pytest.fixture(scope="module")
def adata():
    return ph.load_spe_test()


def test_version():
    assert ph.__version__ == "0.1.0"


def test_fixture_shape(adata):
    assert adata.n_obs == 2661
    assert adata.obsm["spatial"].shape == (2661, 2)
    assert adata.obs["cell_annotation"].nunique() == 6


def test_scan_hoods_rows_sum_to_one():
    rng = np.random.default_rng(0)
    m = np.abs(rng.normal(size=(200, 50)))
    pm = ph.scan_hoods(m, verbose=False)
    assert pm.shape == (200, 50)
    np.testing.assert_allclose(pm.sum(axis=1), 1.0, atol=1e-12)


def test_scan_hoods_rejects_bad_mode():
    with pytest.raises(ValueError):
        ph.scan_hoods(np.ones((5, 5)), mode="nope")


def test_scan_hoods_smoothfadeout_returns_tau():
    rng = np.random.default_rng(1)
    m = np.abs(rng.normal(size=(100, 20))) * 100
    pm, tau = ph.scan_hoods(m, mode="smoothFadeout", verbose=False, return_tau=True)
    assert np.isfinite(tau) and tau > 0
    np.testing.assert_allclose(pm.sum(axis=1), 1.0, atol=1e-12)


def test_pipeline_end_to_end(adata):
    fnc = ph.find_near_cells(adata, k=20)
    assert set(fnc) == {"cells", "distance"}
    assert fnc["distance"].shape == (2661, 20)
    # distances are sorted ascending within each row
    d = fnc["distance"].to_numpy()
    assert (np.diff(d, axis=1) >= -1e-12).all()

    pm = ph.scan_hoods(d, verbose=False)
    hoods = ph.merge_by_group(pm, fnc["cells"])
    assert hoods.shape == (2661, 6)
    np.testing.assert_allclose(hoods.sum(axis=1).to_numpy(), 1.0, atol=1e-12)

    ph.merge_hood_adata(adata, hoods.set_axis(adata.obs_names))
    ph.calc_metrics(adata, pm=hoods.to_numpy())
    assert (adata.obs["entropy"] >= -1e-12).all()
    # perplexity is bounded by the number of neighbourhood types
    assert adata.obs["perplexity"].max() <= 6 + 1e-9
    assert adata.obs["perplexity"].min() >= 1 - 1e-9
    np.testing.assert_allclose(
        adata.obs["perplexity"].to_numpy(),
        2 ** adata.obs["entropy"].to_numpy(),
        rtol=1e-12,
    )


def test_entropy_of_uniform_and_onehot():
    uniform = np.full((1, 4), 0.25)
    res = ph.calculate_metrics(uniform)
    assert res["entropy"].iloc[0] == pytest.approx(2.0)
    assert res["perplexity"].iloc[0] == pytest.approx(4.0)

    onehot = np.array([[1.0, 0.0, 0.0, 0.0]])
    res = ph.calculate_metrics(onehot)
    assert res["entropy"].iloc[0] == pytest.approx(0.0)
    assert res["perplexity"].iloc[0] == pytest.approx(1.0)


def test_merge_by_group_fuzzy():
    rng = np.random.default_rng(2)
    pm = rng.random((50, 10))
    pm /= pm.sum(axis=1, keepdims=True)
    fuzzy = rng.random((10, 3))
    fuzzy /= fuzzy.sum(axis=1, keepdims=True)
    out = ph.merge_by_group(pm, fuzzy, continuous_annotation=True)
    assert out.shape == (50, 3)
    np.testing.assert_allclose(out.sum(axis=1).to_numpy(), 1.0, atol=1e-12)


def test_merge_by_group_shape_check():
    with pytest.raises(ValueError):
        ph.merge_by_group(np.ones((5, 4)), np.array([["a"] * 3] * 5))


def test_class_api(adata):
    hs = ph.HoodScanR(adata, anno_col="cell_annotation")
    hs.find_near_cells(k=15).scan_hoods().merge_by_group().calc_metrics()
    assert hs.hoods.shape == (2661, 6)
    assert "perplexity" in hs.adata.obs
    assert hs.tau > 0
    assert "HoodScanR(" in repr(hs)


def test_clust_by_hood_matrix_path():
    rng = np.random.default_rng(3)
    x = np.vstack([rng.normal(0, 0.1, (100, 3)), rng.normal(3, 0.1, (100, 3))])
    res = ph.clust_by_hood(x, k=2)
    assert len(set(res["cluster"])) == 2
    # the two well-separated blobs must be recovered
    assert res["cluster"][:100].std() == 0 and res["cluster"][100:].std() == 0


def test_clust_by_hood_rejects_other_algorithms():
    with pytest.raises(NotImplementedError):
        ph.clust_by_hood(np.random.random((50, 3)), k=2, algo="Lloyd")


def test_plot_colocal_matrix(adata):
    fnc = ph.find_near_cells(adata, k=20)
    pm = ph.scan_hoods(fnc["distance"].to_numpy(), verbose=False)
    hoods = ph.merge_by_group(pm, fnc["cells"]).set_axis(adata.obs_names)
    ph.merge_hood_adata(adata, hoods)
    m = ph.plot_colocal(adata, return_matrix=True)
    assert m.shape == (6, 6)
    np.testing.assert_allclose(np.diag(m.to_numpy()), 1.0, atol=1e-12)
    assert isinstance(m, pd.DataFrame)


def test_read_hood_data_from_frames():
    pos = pd.DataFrame({"cell_id": ["a", "b", "c"], "x": [0.0, 1.0, 2.0],
                        "y": [0.0, 0.0, 1.0]})
    ann = pd.DataFrame({"cell_id": ["a", "b", "c"], "ct": ["T", "B", "T"]})
    out = ph.read_hood_data(cell_pos_dat=pos, cell_anno_dat=ann)
    assert out.n_obs == 3
    assert list(out.obs["cell_annotation"]) == ["T", "B", "T"]
