import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pathlib import Path

def _fit_chrom(coords, clustering_algorithm, algor_params):
    """
    1) Guards against small scaffolds that have fewer sites than min_samples.
    2) Then fits clustering algorithm to CpG positions for each separate chromosome.
    coords: (n_sites, 1) array of genomic positions on a chromosome (like 55155).
    Returns integer labels, -1 = noise.
    """
    if len(coords) < algor_params.get("min_samples", 1):
        return np.full(len(coords), -1, dtype=int)
    return clustering_algorithm(**algor_params).fit_predict(coords)


def cluster_chroms(
    chr_positions_df, cpg_site_index, clustering_algorithm, algor_params
):
    """
    Parallel per-chromosome clustering with globally unique cluster IDs.

    Each chromosome is clustered independently — algorithm assigns labels 0, 1, 2, ...
    starting from zero. To avoid collisions across chromosomes we shift each
    chromosome's labels up by cluster_offset, then advance the offset past the
    highest label used. Noise sites (-1) are never shifted.

    Returns a Series aligned to cpg_site_index with integer cluster IDs (noise = -1).
    """

    # Perform clustering in parallel across chromosomes; each returns local labels
    with Pool() as pool:
        chrom_labels = pool.starmap(
            _fit_chrom,
            [
                (chrom["pos"].values.reshape(-1, 1), clustering_algorithm, algor_params)
                for _, chrom in chr_positions_df
            ],
        )

    # Combine chromosome-specific labels into a single Series with globally unique labels for clusters
    site_labels = pd.Series(
        -1, index=cpg_site_index, dtype=int
    )  # initialize all CpGs as noise
    cluster_offset = 0  # to ensure unique cluster labels across chromosomes
    for (_, chrom), labels in zip(chr_positions_df, chrom_labels):
        is_cluster = labels != -1
        shifted = labels.copy()
        # DBSCAN returns -1 for noise (sites that are not in clusters)
        # Shift cluster labels to be globally unique
        shifted[is_cluster] += cluster_offset  # make IDs globally unique
        site_labels[chrom.index] = shifted
        if is_cluster.any():
            cluster_offset += (
                labels[is_cluster].max() + 1
            )  # advance past this chrom's IDs
    return site_labels


def cluster_summary(labels, plot_dir=".", name="Clustering", plot=True):
    """
    Print and optionally plot summary statistics for cluster labels.
    labels: pd.Series with integer cluster IDs (-1 = noise).
    """
    clustered = labels[labels != -1]
    n_clusters = clustered.nunique()
    n_noise = (labels == -1).sum()
    noise_pct = 100 * n_noise / len(labels)

    sizes = clustered.value_counts()

    print(f"\n{'=' * 50}")
    print(f"{name}")
    print(f"{'=' * 50}")
    print(f"Clusters:      {n_clusters:>10,}")
    print(f"Noise CpGs:    {n_noise:>10,}  ({noise_pct:.1f}%)")
    print(f"Clustered CpGs:{len(clustered):>10,}  ({100 - noise_pct:.1f}%)")
    print("\nCpGs per cluster:")
    print(f"  mean:   {sizes.mean():>8.1f}")
    print(f"  median: {sizes.median():>8.1f}")
    print(f"  std:    {sizes.std():>8.1f}")
    print(f"  min:    {sizes.min():>8}")
    print(f"  p25:    {int(sizes.quantile(0.25)):>8}")
    print(f"  p75:    {int(sizes.quantile(0.75)):>8}")
    print(f"  p90:    {int(sizes.quantile(0.90)):>8}")
    print(f"  p99:    {int(sizes.quantile(0.99)):>8}")
    print(f"  max:    {sizes.max():>8}")

    if plot:
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.hist(sizes, bins=100, color="steelblue", edgecolor="none", log=True)
        ax.set_xlabel("CpGs per cluster")
        ax.set_ylabel("Number of clusters (log scale)")
        ax.set_title(f"{name} — cluster size distribution")

        for q, label, color in [
            (0.50, "median", "orange"),
            (0.75, "p75", "red"),
            (0.90, "p90", "darkred"),
        ]:
            val = sizes.quantile(q)
            ax.axvline(
                val, color=color, lw=1.5, linestyle="--", label=f"{label}={val:.0f}"
            )

        ax.legend()
        plt.tight_layout()
        folder = Path(plot_dir)
        folder.mkdir(parents=True, exist_ok=True)
        save_path = folder / f"cluster_sizes_{name.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=150)
        plt.show()

    return None