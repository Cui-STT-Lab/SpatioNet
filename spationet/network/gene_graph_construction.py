import os
import numpy as np
import pandas as pd
# from scipy.stats import rankdata, t
# from statsmodels.stats.multitest import multipletests


def construct_refined_edges(
    corpus_path,
    string_edge_path,
    output_dir,
    method="spearman",
    min_obs=10,
    alpha=0.05,
    corpus_index_col=0,
    node1_col="#node1",
    node2_col="node2",
    score_col="combined_score",
    output_filename="refined_edges.csv"
):
    """
    Construct and refine gene-gene edges using STRING network and gene expression correlation.

    Parameters
    ----------
    corpus_path : str
        Path to the spatial transcriptomics corpus CSV.
        Rows are spots/pixels and columns are genes.

    string_edge_path : str
        Path to STRING edge file, for example .tsv file with columns:
        #node1, node2, combined_score.

    output_dir : str
        Directory where refined_edges.csv will be saved.

    method : {"spearman", "pearson"}, default="spearman"
        Correlation method used to refine edges.

    min_obs : int, default=10
        Minimum number of pairwise non-missing observations required.

    alpha : float, default=0.05
        Adjusted p-value threshold after Benjamini-Hochberg correction.

    corpus_index_col : int or None, default=0
        Column to use as row names when reading corpus.

    node1_col : str, default="#node1"
        Name of first gene column in STRING file.

    node2_col : str, default="node2"
        Name of second gene column in STRING file.

    score_col : str, default="combined_score"
        Name of STRING score column.

    output_filename : str, default="refined_edges.csv"
        Name of output CSV file.

    Returns
    -------
    refined_edges : pandas.DataFrame
        Dataframe containing only edges that pass the correlation refinement.
    """

    if method not in ["spearman", "pearson"]:
        raise ValueError("method must be either 'spearman' or 'pearson'.")

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Load corpus and gene list
    # ------------------------------------------------------------
    corpus = pd.read_csv(corpus_path, index_col=corpus_index_col)
    genes_ds = corpus.columns.astype(str).tolist()

    gene_list_path = os.path.join(output_dir, "gene_list.csv")
    pd.DataFrame({"gene": genes_ds}).to_csv(gene_list_path, index=False)

    # ------------------------------------------------------------
    # 2. Load STRING edges
    # ------------------------------------------------------------
    string_edges = pd.read_csv(string_edge_path, sep=None, engine="python")

    required_cols = [node1_col, node2_col, score_col]
    missing_cols = [col for col in required_cols if col not in string_edges.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in STRING file: {missing_cols}")

    graph_nodes = sorted(
        set(string_edges[node1_col].astype(str)).union(
            set(string_edges[node2_col].astype(str))
        )
    )

    hits = sorted(set(genes_ds).intersection(graph_nodes))
    miss = sorted(set(genes_ds).difference(graph_nodes))

    print(f"Matches: {len(hits)}")
    print(f"Missing from STRING network: {len(miss)}")

    # ------------------------------------------------------------
    # 3. Keep only STRING edges where both genes are in corpus
    # ------------------------------------------------------------
    string_edges[node1_col] = string_edges[node1_col].astype(str)
    string_edges[node2_col] = string_edges[node2_col].astype(str)

    edges = string_edges[
        string_edges[node1_col].isin(genes_ds)
        & string_edges[node2_col].isin(genes_ds)
    ][[node1_col, node2_col, score_col]].copy()

    edges.columns = ["gene_from", "gene_to", "ppi"]

    filtered_path = os.path.join(output_dir, "filtered_edges.csv")
    edges.to_csv(filtered_path, index=False)

    # ------------------------------------------------------------
    # 4. Refine edges by correlation
    # ------------------------------------------------------------
    refined = refine_edges_by_corr(
        corpus=corpus,
        edges=edges,
        method=method,
        min_obs=min_obs,
        alpha=alpha
    )

    refined_edges = refined[refined["keep"]].copy()

    output_path = os.path.join(output_dir, output_filename)
    refined_edges.to_csv(output_path, index=False)

    print(f"Saved refined edges to: {output_path}")
    print(f"Number of filtered STRING edges: {edges.shape[0]}")
    print(f"Number of refined edges kept: {refined_edges.shape[0]}")

    return refined_edges


def refine_edges_by_corr(
    corpus,
    edges,
    method="spearman",
    min_obs=10,
    alpha=0.05
):
    """
    Refine gene-gene edges by testing pairwise gene expression correlation.

    Parameters
    ----------
    corpus : pandas.DataFrame
        Expression matrix with rows as spots/cells and columns as genes.

    edges : pandas.DataFrame
        Edge dataframe with columns: gene_from, gene_to, and optionally ppi.

    method : {"spearman", "pearson"}, default="spearman"
        Correlation method.

    min_obs : int, default=10
        Minimum number of non-missing paired observations.

    alpha : float, default=0.05
        BH-adjusted p-value cutoff.

    Returns
    -------
    out : pandas.DataFrame
        Original edges plus correlation statistics and keep indicator.
    """

    required_cols = {"gene_from", "gene_to"}
    if not required_cols.issubset(edges.columns):
        raise ValueError("edges must contain columns: gene_from and gene_to")

    if method not in ["spearman", "pearson"]:
        raise ValueError("method must be either 'spearman' or 'pearson'.")

    corpus = corpus.copy()
    edges = edges.copy()

    corpus.columns = corpus.columns.astype(str)
    edges["gene_from"] = edges["gene_from"].astype(str)
    edges["gene_to"] = edges["gene_to"].astype(str)

    needed_genes = pd.unique(
        pd.concat([edges["gene_from"], edges["gene_to"]], ignore_index=True)
    )

    present_genes = [gene for gene in needed_genes if gene in corpus.columns]

    if len(present_genes) == 0:
        raise ValueError("None of the edge genes are present in corpus.")

    X = corpus[present_genes].apply(pd.to_numeric, errors="coerce")

    # Spearman correlation is Pearson correlation on ranks
    if method == "spearman":
        X_ranked = X.copy()
        for col in X_ranked.columns:
            values = X_ranked[col].to_numpy(dtype=float)
            mask = ~np.isnan(values)

            ranked_values = np.full(values.shape, np.nan)
            ranked_values[mask] = rankdata(values[mask], method="average")

            X_ranked[col] = ranked_values

        X = X_ranked

    # Pairwise correlation matrix
    R = X.corr(method="pearson", min_periods=1)

    # Pairwise sample size matrix
    not_missing = (~X.isna()).astype(int)
    N = not_missing.T @ not_missing
    N = pd.DataFrame(N, index=X.columns, columns=X.columns)

    r_values = []
    n_values = []

    for _, row in edges.iterrows():
        g1 = row["gene_from"]
        g2 = row["gene_to"]

        if g1 in R.index and g2 in R.columns:
            r_values.append(R.loc[g1, g2])
            n_values.append(N.loc[g1, g2])
        else:
            r_values.append(np.nan)
            n_values.append(np.nan)

    r = np.array(r_values, dtype=float)
    n = np.array(n_values, dtype=float)

    abs_corr = np.abs(r)

    # ------------------------------------------------------------
    # Compute p-values for H0: rho = 0
    # t = r * sqrt((n - 2) / (1 - r^2))
    # ------------------------------------------------------------
    p_values = np.full(len(r), np.nan)
    df = n - 2

    ok = (
        ~np.isnan(r)
        & ~np.isnan(n)
        & (n >= 3)
        & np.isfinite(r)
        & (np.abs(r) < 1)
    )

    t_stat = np.full(len(r), np.nan)
    t_stat[ok] = r[ok] * np.sqrt(df[ok] / (1 - r[ok] ** 2))

    p_values[ok] = 2 * t.sf(np.abs(t_stat[ok]), df=df[ok])

    # ------------------------------------------------------------
    # Benjamini-Hochberg correction
    # ------------------------------------------------------------
    p_adj = np.full(len(p_values), np.nan)
    valid_p = ~np.isnan(p_values)

    if valid_p.sum() > 0:
        p_adj[valid_p] = multipletests(
            p_values[valid_p],
            method="fdr_bh"
        )[1]

    out = edges.copy()
    out["corr"] = r
    out["abs_corr"] = abs_corr
    out["n_pairwise"] = n
    out["p_value"] = p_values
    out["p_adj"] = p_adj
    out["method"] = method

    out["keep"] = (
        ~out["abs_corr"].isna()
        & ~out["n_pairwise"].isna()
        & (out["n_pairwise"] >= min_obs)
        & ~out["p_adj"].isna()
        & (out["p_adj"] < alpha)
    )

    return out