import numpy as np
import pandas as pd


def _cluster_order_index(n_sites):
    return np.arange(1, n_sites + 1, dtype=float)


def _standardized_cluster_index(n_sites):
    order_index = _cluster_order_index(n_sites)
    centered = order_index - order_index.mean()
    scale = centered.std(ddof=0)
    if scale == 0:
        return np.zeros(n_sites, dtype=float)
    return centered / scale


def build_cluster_cov_X(
    methylation_matrix,
    site_coords,
    cluster_col,
    pos_col="pos",
    fillna=None,
):
    """
    Build one per-cluster feature for each sample: covariance(cluster_index_z, beta).

    For each non-noise cluster, the function computes covariance between the
    standardized within-cluster CpG order index and methylation level.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (n_samples, n_clusters).
    """
    clustered = site_coords.loc[site_coords[cluster_col] != -1, [cluster_col, pos_col]].copy()
    clustered = clustered.loc[clustered.index.intersection(methylation_matrix.index)]

    if clustered.empty:
        return pd.DataFrame(index=methylation_matrix.columns)

    methylation_aligned = methylation_matrix.loc[clustered.index]
    sample_index = methylation_aligned.columns
    cluster_features = {}

    def _nan_cov(values, positions):
        mask = ~np.isnan(values)
        n = mask.sum(axis=0).astype(float)

        x = positions[:, None]
        x_masked = x * mask
        y_masked = np.where(mask, values, 0.0)

        sum_x = x_masked.sum(axis=0)
        sum_y = y_masked.sum(axis=0)
        sum_xy = (y_masked * x).sum(axis=0)

        cov = np.full(values.shape[1], np.nan, dtype=float)
        valid = n >= 2
        cov[valid] = (sum_xy[valid] - (sum_x[valid] * sum_y[valid]) / n[valid]) / n[valid]
        return cov

    for cluster_id, cluster_sites in clustered.groupby(cluster_col, sort=False):
        cluster_sites = cluster_sites.sort_values(pos_col)
        positions = _standardized_cluster_index(len(cluster_sites))
        values = methylation_aligned.loc[cluster_sites.index].to_numpy(dtype=float)
        cluster_features[cluster_id] = _nan_cov(values, positions)

    X_cov = pd.DataFrame(cluster_features, index=sample_index)

    if fillna is None:
        return X_cov
    if fillna == "feature_mean":
        return X_cov.apply(lambda col: col.fillna(col.mean()), axis=0)
    if fillna == "zero":
        return X_cov.fillna(0.0)
    if isinstance(fillna, (int, float)):
        return X_cov.fillna(float(fillna))

    raise ValueError(
        "fillna must be one of: None, 'feature_mean', 'zero', or a numeric value"
    )


def build_cluster_corr_X(
    methylation_matrix,
    site_coords,
    cluster_col,
    pos_col="pos",
    method="pearson",
    scale_by=None,
    fillna=None,
):
    """
    For each sample and cluster, compute correlation between CpG order inside the
    cluster and methylation level within that cluster.

    Returns a DataFrame of shape (n_samples, n_clusters).

    Notes
    -----
    - CpGs are ordered by genomic position, but the correlation is computed against
      the within-cluster CpG order index rather than absolute genomic coordinates.
    - Fast vectorized path is used for "pearson" and "spearman".
    - "kendall" falls back to a slower per-sample implementation.
    - NaNs are handled pairwise within each sample/cluster.
    - Optionally, the correlation can be multiplied by the per-sample cluster
      mean or median methylation level using `scale_by`.
    """
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be one of: 'pearson', 'spearman', 'kendall'")
    if scale_by not in {None, "mean", "median"}:
        raise ValueError("scale_by must be one of: None, 'mean', 'median'")

    clustered = site_coords.loc[site_coords[cluster_col] != -1, [cluster_col, pos_col]].copy()
    clustered = clustered.loc[clustered.index.intersection(methylation_matrix.index)]

    if clustered.empty:
        return pd.DataFrame(index=methylation_matrix.columns)

    methylation_aligned = methylation_matrix.loc[clustered.index]
    sample_index = methylation_aligned.columns
    cluster_features = {}

    def _corr_against_positions(values, positions):
        mask = ~np.isnan(values)
        n = mask.sum(axis=0).astype(float)

        x = positions[:, None]
        x_masked = x * mask
        y_masked = np.where(mask, values, 0.0)

        sum_x = x_masked.sum(axis=0)
        sum_y = y_masked.sum(axis=0)
        sum_x2 = (x_masked * x).sum(axis=0)
        sum_y2 = (y_masked * y_masked).sum(axis=0)
        sum_xy = (y_masked * x).sum(axis=0)

        num = n * sum_xy - sum_x * sum_y
        den = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))

        corr = np.full(values.shape[1], np.nan, dtype=float)
        valid = (n >= 2) & np.isfinite(den) & (den > 0)
        corr[valid] = num[valid] / den[valid]
        return corr

    for cluster_id, cluster_sites in clustered.groupby(cluster_col, sort=False):
        cluster_sites = cluster_sites.sort_values(pos_col)
        positions = _cluster_order_index(len(cluster_sites))
        values = methylation_aligned.loc[cluster_sites.index].to_numpy(dtype=float)

        if values.shape[0] < 2:
            cluster_features[cluster_id] = np.full(values.shape[1], np.nan, dtype=float)
            continue

        if method == "pearson":
            corr = _corr_against_positions(values, positions)
        elif method == "spearman":
            ranked_values = (
                pd.DataFrame(values, index=cluster_sites.index, columns=sample_index)
                .rank(axis=0, method="average", na_option="keep")
                .to_numpy(dtype=float)
            )
            ranked_positions = pd.Series(positions).rank(method="average").to_numpy(dtype=float)
            corr = _corr_against_positions(ranked_values, ranked_positions)
        else:
            corr = np.full(values.shape[1], np.nan, dtype=float)
            pos_series = pd.Series(positions, index=cluster_sites.index, dtype=float)
            for sample_idx, _sample_name in enumerate(sample_index):
                val_series = pd.Series(values[:, sample_idx], index=cluster_sites.index, dtype=float)
                valid = pos_series.notna() & val_series.notna()
                if valid.sum() < 2:
                    continue
                if pos_series[valid].nunique() < 2 or val_series[valid].nunique() < 2:
                    continue
                corr[sample_idx] = pos_series[valid].corr(val_series[valid], method="kendall")

        if scale_by == "mean":
            cluster_stat = np.nanmean(values, axis=0)
            corr = corr * cluster_stat
        elif scale_by == "median":
            cluster_stat = np.nanmedian(values, axis=0)
            corr = corr * cluster_stat

        cluster_features[cluster_id] = corr

    X_corr = pd.DataFrame(cluster_features, index=sample_index)

    if fillna is None:
        return X_corr
    if fillna == "cluster_mean":
        return X_corr.apply(lambda col: col.fillna(col.mean()), axis=0)
    if fillna == "zero":
        return X_corr.fillna(0.0)
    if isinstance(fillna, (int, float)):
        return X_corr.fillna(float(fillna))

    raise ValueError(
        "fillna must be one of: None, 'cluster_mean', 'zero', or a numeric value"
    )


def build_cluster_binned_X(
    methylation_matrix,
    site_coords,
    cluster_col,
    pos_col="pos",
    bin_mode="n_bins",
    n_bins=5,
    sites_per_bin=None,
    bin_agg="mean",
    fillna=None,
):
    """
    Aggregate methylation within clusters using simple consecutive bins.

    CpGs are first sorted by genomic position inside each cluster. Then each cluster is split
    into consecutive bins using one of two strategies:
    - `bin_mode="n_bins"`: split each cluster into at most `n_bins` consecutive bins.
    - `bin_mode="sites_per_bin"`: put `sites_per_bin` CpGs into each consecutive bin.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (n_samples, n_cluster_bins).
    """
    if bin_mode not in {"n_bins", "sites_per_bin"}:
        raise ValueError("bin_mode must be one of: 'n_bins', 'sites_per_bin'")
    if bin_agg not in {"mean", "median"}:
        raise ValueError("bin_agg must be 'mean' or 'median'")
    if n_bins is not None and n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if sites_per_bin is not None and sites_per_bin < 1:
        raise ValueError("sites_per_bin must be >= 1")
    if bin_mode == "n_bins" and n_bins is None:
        raise ValueError("n_bins must be provided when bin_mode='n_bins'")
    if bin_mode == "sites_per_bin" and sites_per_bin is None:
        raise ValueError("sites_per_bin must be provided when bin_mode='sites_per_bin'")

    clustered = site_coords.loc[site_coords[cluster_col] != -1, [cluster_col, pos_col]].copy()
    clustered = clustered.loc[clustered.index.intersection(methylation_matrix.index)]

    if clustered.empty:
        return pd.DataFrame(index=methylation_matrix.columns)

    clustered = clustered.sort_values([cluster_col, pos_col], kind="stable")
    methylation_aligned = methylation_matrix.reindex(clustered.index)
    sample_index = methylation_aligned.columns
    values_all = methylation_aligned.to_numpy(dtype=float)
    cluster_ids = clustered[cluster_col].to_numpy()
    unique_clusters, cluster_starts, cluster_sizes = np.unique(
        cluster_ids, return_index=True, return_counts=True
    )

    cluster_features = {}
    bin_reducer = np.nanmean if bin_agg == "mean" else np.nanmedian

    def _aggregate_rows(values):
        with np.errstate(all="ignore"):
            return bin_reducer(values, axis=0)

    def _build_bin_ranges(n_sites):
        if bin_mode == "n_bins":
            n_bins_eff = min(n_bins, n_sites)
            bounds = np.linspace(0, n_sites, n_bins_eff + 1, dtype=int)
            return [
                (bounds[idx], bounds[idx + 1])
                for idx in range(n_bins_eff)
                if bounds[idx] < bounds[idx + 1]
            ]

        return [
            (start, min(start + sites_per_bin, n_sites))
            for start in range(0, n_sites, sites_per_bin)
        ]

    for cluster_id, start, size in zip(unique_clusters, cluster_starts, cluster_sizes):
        cluster_values = values_all[start : start + size]

        for bin_id, (bin_start, bin_end) in enumerate(_build_bin_ranges(size), start=1):
            cluster_features[f"{cluster_id}_bin{bin_id}"] = _aggregate_rows(
                cluster_values[bin_start:bin_end]
            )

    X_binned = pd.DataFrame(cluster_features, index=sample_index)

    if fillna is None:
        return X_binned
    if fillna == "feature_mean":
        return X_binned.apply(lambda col: col.fillna(col.mean()), axis=0)
    if fillna == "zero":
        return X_binned.fillna(0.0)
    if isinstance(fillna, (int, float)):
        return X_binned.fillna(float(fillna))

    raise ValueError(
        "fillna must be one of: None, 'feature_mean', 'zero', or a numeric value"
    )
