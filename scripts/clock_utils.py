import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LassoCV


def build_cluster_X(petkovich, site_coords, cluster_col):
    """
    Compute mean methylation per cluster per sample; returns a (samples × clusters) DataFrame.

    NaN handling
    ------------
    1) Intra-cluster NaNs (sites with coverage ≤ 5 reads are NaN after preprocessing):
       During the preprocesing step, all the methylation sites that had coverage <= 5 were
       handled as NaN. Therefore, some of the enteries in `petkovich` dataframe are NaN.
       Since the whole philosophy of clustering and calculating using the cluster mean is based
       on the assumption that neighbouring sites have the same methylation status, we can ignore
       these NaN values when calculating the mean methylation level for each cluster.
       This approach repicates one that was used in the Simpson et al. (2023) paper.
       However, this approach has a room for improvement. If a cluster has too many NaN values,
       it may not be reliable to calculate the mean methylation level for that cluster. In such
       cases, we can set a threshold for the maximum allowed percentage of NaN values in a
       cluster. If the percentage of NaN values exceeds this threshold, we can choose to exclude
       that cluster from further analysis. I haven't implemented this approach yet, but it is
       something that can be considered in the future.

    2) Residual per-sample NaNs after cluster averaging:
       After the mean methylation per cluster per sample is calculated, some clusters in some
       samples may still have NaN values.
       I couldn't get how Simpson et al. (2023) dealed with those NaN values. Unfortunately,
       they haven't uploaded code with their analysis anywhere. Therefore, I can only speculate:
       According to the 4.6. in the Methods section they say: "Regions with NaN (i.e. no reads
       recorded) were removed." It's not clear to me how to interpret this statement.
       THerefore, I impute these NaN values with the global mean of the cluster across all
       samples. This way, we can retain more clusters in our analysis.
       If a cluster is NaN across all samples, it is dropped.
    """
    clustered = site_coords[site_coords[cluster_col] != -1]
    cluster_met = (
        petkovich.loc[clustered.index].groupby(clustered[cluster_col]).mean(skipna=True)
    )
    row_means = cluster_met.mean(axis=1)
    cluster_met = cluster_met.where(cluster_met.notna(), row_means, axis=0)
    return cluster_met.dropna(axis=0).T


def build_thompson_cluster_features(thompson, site_coords, cluster_col, X_train):
    """
    Build Thompson cluster feature matrix (n_samples × n_clusters) aligned to X_train.

    Only Thompson CpG sites that belong to clusters present in X_train are used.
    Clusters entirely absent from Thompson (no covered sites) are filled with the
    Petkovich training-set mean for that cluster.

    Parameters
    ----------
    thompson : DataFrame, shape (n_thompson_samples, n_cpg_sites)
        Thompson methylation matrix with CpG site IDs as columns.
    site_coords : DataFrame
        Site-level metadata; must contain a column named `cluster_col`.
    cluster_col : str
        Column in `site_coords` with integer cluster IDs (-1 = noise).
    X_train : DataFrame, shape (n_petkovich_samples, n_clusters)
        Petkovich cluster feature matrix whose columns define the feature space.

    Returns
    -------
    numpy.ndarray, shape (n_thompson_samples, n_clusters)

    Imputation (two-step)
    -----------------------------------
    1. Clusters with NaN values in some samples → filled with Thompson column mean for that cluster.
    2. Fully absent clusters (no sites in Thompson dataset) → filled with Petkovich train mean.
    """
    train_means = X_train.mean().values.astype("float64")  # (n_clusters,)

    clustered = site_coords[site_coords[cluster_col] != -1]
    active = clustered[
        clustered[cluster_col].isin(X_train.columns)
        & clustered.index.isin(thompson.columns)
    ]
    X = (
        thompson[active.index]
        .T.assign(cluster=active[cluster_col])
        .groupby("cluster")
        .mean()
        .T.reindex(columns=X_train.columns)
    ).values.astype("float64")

    col_means = np.nanmean(X, axis=0)                     # NaN where entire column is NaN
    nan_mask = np.isnan(X)
    partial = nan_mask & ~np.all(nan_mask, axis=0)         # NaN but column has some data
    rows_p, cols_p = np.where(partial)
    X[rows_p, cols_p] = col_means[cols_p]                 # fill with Thompson col mean
    rows_s, cols_s = np.where(np.isnan(X))
    X[rows_s, cols_s] = train_means[cols_s]               # fill absent clusters with train mean
    return X


def build_thompson_cpg_features(thompson, X_train):
    """
    Build Thompson CpG feature matrix (n_samples × n_cpgs) aligned to X_train.

    CpGs absent from Thompson entirely are filled with the Petkovich training-set mean.
    CpGs present but partially NaN are filled with the Thompson mean for these CpGs across samples.

    Parameters
    ----------
    thompson : DataFrame, shape (n_thompson_samples, n_cpg_sites)
        Thompson methylation matrix with CpG site IDs as columns.
    X_train : DataFrame, shape (n_petkovich_samples, n_cpgs)
        Petkovich CpG feature matrix whose columns define the feature space.

    Returns
    -------
    numpy.ndarray, shape (n_thompson_samples, n_cpgs)

    Imputation (two-step)
    -----------------------------------
    1. CpGs with NaN values in some samples → filled with Thompson column mean for that CpG.
    2. Fully absent CpGs (not measured in Thompson dataset) → filled with Petkovich train mean.
    """
    train_means = X_train.mean().values.astype("float64")  # (n_cpgs,)
    common = X_train.columns.intersection(thompson.columns)

    out = np.empty((len(thompson), len(X_train.columns)), dtype="float64")
    out[:] = train_means  # default: Petkovich mean for every CpG

    if len(common):
        col_idx = np.where(X_train.columns.isin(common))[0]
        vals = thompson[common].values.astype("float64")   # (n_samples, n_common)
        col_means = np.nanmean(vals, axis=0)
        nan_mask = np.isnan(vals)
        rows_n, cols_n = np.where(nan_mask)
        vals[rows_n, cols_n] = col_means[cols_n]           # Thompson col mean fallback
        out[:, col_idx] = vals

    return out


def nested_cv(X, y, k=5):
    """
    Nested cross-validation with LassoCV.
    Outer loop (k folds): evaluates model performance (R², RMSE).
    Inner loop (k folds): LassoCV selects optimal alpha via MSE.
    """
    inner_cv = KFold(n_splits=k, shuffle=True, random_state=42)
    outer_cv = KFold(n_splits=k, shuffle=True, random_state=42)

    model = LassoCV(cv=inner_cv, max_iter=10000, random_state=42)
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=outer_cv,
        n_jobs=-1,
        return_estimator=True,
        return_indices=True,
    )

    # Out-of-fold predictions for plotting
    oof_pred = np.empty(len(y))
    for estimator, test_idx in zip(
        cv_results["estimator"], cv_results["indices"]["test"]
    ):
        oof_pred[test_idx] = estimator.predict(X.iloc[test_idx])

    y_true = np.asarray(y)
    residuals = y_true - oof_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    return {
        "R2 (OOF)": 1 - ss_res / ss_tot,
        "RMSE (OOF)": np.sqrt(ss_res / len(y_true)),
        "Avg Features": np.mean(
            [np.count_nonzero(e.coef_) for e in cv_results["estimator"]]
        ),
        "Best Alphas": [e.alpha_ for e in cv_results["estimator"]],
        "oof_pred": oof_pred,
        "y_true": y_true,
    }


def plot_age_clock(predictions, y, title, style="scatter", ax=None):
    """
    Plot predicted vs. chronological age for one or more clocks.

    Parameters
    ----------
    predictions : list of (pred_array, label, color)
        Each tuple contains:
        - pred_array : array-like of predicted ages (same length as y).
        - label      : legend string (supports newlines for multi-line labels).
        - color      : matplotlib color.
    y : Series
        Chronological ages; index is used for grouping in "lines" mode.
    title : str
        Axes title.
    style : {"scatter", "lines"}
        "scatter" — individual points + linear fit line.
        "lines"   — mean predicted age per unique chronological age ± 1 std band.
    ax : matplotlib Axes, optional
        Draw into an existing axes; a new figure is created if None.
    """
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(8, 7))

    lims = [y.min(), y.max()]
    ax.plot(lims, lims, "--k", alpha=0.5, label="Baseline (x=y)")

    for pred, lbl, color in predictions:
        if style == "scatter":
            m, b = np.polyfit(y, pred, 1)
            ax.scatter(y, pred, color=color, alpha=0.4, s=20, zorder=3)
            ax.plot(lims, np.polyval([m, b], lims), color=color, label=lbl, lw=2.5)
        else:
            stats = (
                pd.Series(pred, index=y.index).groupby(y).agg(["mean", "std"]).fillna(0)
            )
            ax.plot(stats.index, stats["mean"], color=color, label=lbl, lw=2.5)
            ax.fill_between(
                stats.index,
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                alpha=0.2,
                color=color,
            )

    ax.set(
        xlabel="Chronological Age (months)",
        ylabel="Predicted Age (months)",
        title=title,
    )
    ax.legend(loc="upper left", frameon=True, shadow=True, fontsize=11)
    ax.grid(True, ls=":", alpha=0.6)
    if own_fig:
        plt.tight_layout()
        plt.show()
