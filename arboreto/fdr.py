from html.parser import piclose

from arboreto.core import EARLY_STOP_WINDOW_LENGTH, SGBM_KWARGS, DEMON_SEED, to_tf_matrix, target_gene_indices, clean, fit_model, to_links_df
from arboreto.fdr_utils import compute_wasserstein_distance_matrix, cluster_genes_to_dict, merge_gene_clusterings, compute_medoids, partition_input_grn, invert_tf_to_cluster_dict, count_helper, subset_tf_matrix, _prepare_client, _prepare_input
import numpy as np
import pandas as pd
from dask import delayed, compute
from dask.dataframe import from_delayed
from dask.dataframe.utils import make_meta
import pickle
import os

FDR_GRN_SCHEMA = make_meta({'TF': str, 'target': str, 'importance': float, 'count' : float})

def perform_fdr(
        expression_data : pd.DataFrame,
        input_grn : dict,
        num_non_tf_clusters : int,
        num_tf_clusters : int,
        cluster_representative_mode : str,
        tf_names,
        client_or_address,
        early_stop_window_length,
        seed,
        verbose,
        num_permutations,
        output_dir
):
    # Extract TF name and non-TF name lists from expression matrix object.
    expression_matrix, gene_names, tf_names = _prepare_input(expression_data, None, tf_names)
    non_tf_names = [gene for gene in gene_names if not gene in tf_names]

    # Check if TF clustering is desired.
    if num_tf_clusters == -1:
        num_tf_clusters = len(tf_names)
        are_tfs_clustered = False
    else:
        are_tfs_clustered = True

    # No clustering necessary, just create 'dummy' clustering with singleton clusters.
    tf_representatives = []
    non_tf_representatives = []
    if cluster_representative_mode == 'all_genes':
        all_gene_clustering=None
        tf_representatives = tf_names
        non_tf_representatives = non_tf_names
        are_tfs_clustered = False
    else: # Cluster genes based on Wasserstein distance.
        # Compute full distance matrix between all pairs of input genes.
        dist_matrix_all = compute_wasserstein_distance_matrix(expression_data, num_threads=-1)

        if not output_dir is None:
            dist_matrix_all.to_csv(os.path.join(output_dir, 'distance_matrix.csv'))

        # Separate TF and non-TF distances and cluster both types individually.
        tf_bool = [True if gene in tf_names else False for gene in dist_matrix_all.columns]
        non_tf_bool = [False if gene in tf_names else True for gene in dist_matrix_all.columns]
        dist_mat_non_tfs = dist_matrix_all.loc[non_tf_bool, non_tf_bool]
        dist_mat_tfs = dist_matrix_all.loc[tf_bool, tf_bool]

        non_tf_to_clust = cluster_genes_to_dict(dist_mat_non_tfs, num_clusters=num_non_tf_clusters)
        tf_to_clust = cluster_genes_to_dict(dist_mat_tfs, num_clusters=num_tf_clusters)
        all_gene_clustering = merge_gene_clusterings(tf_to_clust, non_tf_to_clust)

        if not output_dir is None:
            with open(os.path.join(output_dir, 'gene_clustering.pkl'), 'wb') as f:
                pickle.dump(all_gene_clustering, f)

        if cluster_representative_mode == 'medoid':
            tf_representatives = compute_medoids(tf_to_clust, dist_matrix_all)
            non_tf_representatives = compute_medoids(non_tf_to_clust, dist_matrix_all)
        else: # cluster_representative_mode='random'
            tf_representatives = tf_names
            non_tf_representatives = non_tf_names

    if not output_dir is None and cluster_representative_mode=='medoid':
        with open(os.path.join(output_dir, 'tf_medoids.pkl'), 'wb') as f:
            pickle.dump(tf_representatives, f)
        with open(os.path.join(output_dir, 'non_tf_medoids.pkl'), 'wb') as f:
            pickle.dump(non_tf_representatives, f)

    return diy_fdr(expression_data=expression_data,
                   regressor_type='GBM',
                   regressor_kwargs=SGBM_KWARGS,
                   gene_names=gene_names,
                   are_tfs_clustered=are_tfs_clustered,
                   tf_representatives=tf_representatives,
                   non_tf_representatives=non_tf_representatives,
                   gene_to_cluster=all_gene_clustering,
                   input_grn=input_grn,
                   client_or_address=client_or_address,
                   early_stop_window_length=early_stop_window_length,
                   seed=seed,
                   verbose=verbose,
                   n_permutations=num_permutations,
                   output_dir=output_dir
                   )


def diy_fdr(expression_data,
            regressor_type,
            regressor_kwargs,
            are_tfs_clustered,
            tf_representatives,
            non_tf_representatives,
            gene_to_cluster,
            input_grn,
            gene_names=None,
            client_or_address='local',
            early_stop_window_length=None,
            seed=None,
            verbose=False,
            n_permutations=1000,
            output_dir=None
            ):
    """
    :param are_tfs_clustered: True if TFs have also been clustered for FDR control.
    :param tf_representatives: Either list of pre-chosen TF representatives or simply all TFs.
    :param non_tf_representatives: Either list of pre-chosen non-TF representatives or all non-TFs.
    :param gene_to_cluster: Keys are gene names and values are cluster IDs as integers.
    :param input_grn: Dict storing input GRN for FDR control with keys as edge tuples, and as values dicts with
        {'importance' : <float>} structure.
    :param expression_data: one of:
           * a pandas DataFrame (rows=observations, columns=genes)
           * a dense 2D numpy.ndarray
           * a sparse scipy.sparse.csc_matrix
    :param regressor_type: string. One of: 'RF', 'GBM', 'ET'. Case insensitive.
    :param regressor_kwargs: a dictionary of key-value pairs that configures the regressor.
    :param gene_names: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
                       'expression_data' instead of a DataFrame.
    :param early_stop_window_length: early stopping window length.
    :param client_or_address: one of:
           * None or 'local': a new Client(LocalCluster()) will be used to perform the computation.
           * string address: a new Client(address) will be used to perform the computation.
           * a Client instance: the specified Client instance will be used to perform the computation.
    :param seed: optional random seed for the regressors. Default 666. Use None for random seed.
    :param verbose: print info.
    :return: a pandas DataFrame['TF', 'target', 'importance'] representing the inferred gene regulatory links.
    """
    if verbose:
        print('preparing dask client')

    client, shutdown_callback = _prepare_client(client_or_address)

    try:
        if verbose:
            print('parsing input')

        # TF names do not matter in FDR mode, hence can be set to dummy list.
        tf_names = None
        expression_matrix, gene_names, _ = _prepare_input(expression_data, gene_names, tf_names)

        if verbose:
            print('creating dask graph')

        if input_grn is None:
            raise ValueError(f'Input GRN is None, but needs to be passed in FDR mode.')
        if tf_representatives is None or non_tf_representatives is None:
            raise ValueError(f'TF or non-TF representatives are None, but need to passed in FDR mode.')
        if gene_to_cluster is None:
            if verbose:
                print("Genes have not been clustered, running full FDR mode.")

        graph = create_graph_fdr(expression_matrix,
                                 gene_names=gene_names,
                                 are_tfs_clustered=are_tfs_clustered,
                                 tf_representatives=tf_representatives,
                                 non_tf_representatives=non_tf_representatives,
                                 gene_to_cluster=gene_to_cluster,
                                 input_grn=input_grn,
                                 regressor_type=regressor_type,
                                 regressor_kwargs=regressor_kwargs,
                                 client=client,
                                 early_stop_window_length=early_stop_window_length,
                                 seed=seed,
                                 n_permutations=n_permutations,
                                 output_dir=output_dir)

        if verbose:
            print('{} partitions'.format(graph.npartitions))
            print('computing dask graph')

        return client \
            .compute(graph, sync=True) \
            .sort_values(by='importance', ascending=False)

    finally:
        shutdown_callback(verbose)

        if verbose:
            print('finished')


def create_graph_fdr(expression_matrix: np.ndarray,
                     gene_names: list[str],
                     are_tfs_clustered,
                     tf_representatives: list[str],
                     non_tf_representatives: list[str],
                     gene_to_cluster: dict[str, int],
                     input_grn: dict,
                     regressor_type,
                     regressor_kwargs,
                     client,
                     early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
                     repartition_multiplier=1,
                     seed=DEMON_SEED,
                     n_permutations=1000,
                     output_dir=None
                     ):
    """
    Main API function for FDR control. Create a Dask computation graph.

    Note: fixing the GC problems was fixed by 2 changes: [1] and [2] !!!

    :param expression_matrix: numpy matrix. Rows are observations and columns are genes.
    :param gene_names : list[str]. List of gene names, in the order as they appear in the expression matrix columns.
    :param fdr_mode: str. One of 'medoid', 'random' to indicate representative drawing mode.
    :param are_tfs_clustered: bool. True if TFs have also been clustered, False otherwise.
    :param tf_representatives: list[str]. List of TFs, either only medoids, or all TFs in 'random' mode.
    :param non_tf_representatives: list[str]. List of non-TFs, either only medoids, or all non-TFs in 'random' mode.
    :param gene_to_cluster: dict[str, int]. Keys are gene names, values are cluster IDs they belong to. If set to
        None, run full 'groundtruth' FDR control on all genes.
    :param input_grn: dict. Input GRN to be FDR-controlled in dict format with (tf,target) as keys.
    :param regressor_type: regressor type. Case insensitive.
    :param regressor_kwargs: dict of key-value pairs that configures the regressor.
    :param client: a dask.distributed client instance.
                   * Used to scatter-broadcast the tf matrix to the workers instead of simply wrapping in a delayed().
    :param early_stop_window_length: window length of the early stopping monitor.
    :param repartition_multiplier: multiplier
    :param n_permutations: int. Number of random permutations to run for FDR control.
    :param seed: (optional) random seed for the regressors. Default 666.
    :return: if include_meta is False, returns a Dask graph that computes the links DataFrame.
             If include_meta is True, returns a tuple: the links DataFrame and the meta DataFrame.
    """
    '''
    4 cases
    A) TF matrix (normal) + medoid gene expression VECTOR (TFs are passed as is)
    Pass variables
        - full TF matrix
        - subset target gene expression matrix (medoid)
        - per target cluster: input GRN subsetted to all target gene represented by the medoid 
    Loop logic the same.
    Counting logic must be modified in infer_partial_network:
        Count logic A.
        - for every edge in input_grn:
            iterate over all edges in input grn and compare importance values for every edge with permuted importance value.
            It does not matter whether it is TF or medoid, because the input GRN is already subset to only include the 
            relevant edges.

    B) TF matrix (normal) + sampling gene expression MATRIX
    Pass variables
        - full TF matrix
        - gex_mat: subset target gene expression matrix (submatrix containing all genes in the cluster)
        - per target cluster: input GRN subsetted to all target genes in the cluster
    Loop logic needs to be modified: We need to loop over/'sample from' gex_mat and permute the current vector of gene expression
    Count logic: Count logic A

    C) TF matrix clustered medoid + medioid gene expression VECTOR
    Conceptual output: Edge between TF cluster and target gene cluster
    Pass variables:
        - medoid TF matrix (only medioid TFs can be targets)
        - subset target gene expression matrix (medoid)
        - per target cluster and TF cluster: input GRN subsetted to all trancription factors 
            and target genes represented by the medoid tfs and target medoids.
    Loop logic: same as A
    Count logic C: For every edge in input_GRN increase counter if permuted value is larger that input value.

    D) TF matrix clustered sampling + sampleing gene expression MATRIX
    Pass variables:
        - Full TF matrix + cluster information (shared globally with all nodes in dansk)
        - gex_mat: subset target gene expression matrix (submatrix containing all genes in the cluster)
        - per target cluster: input GRN subsetted to all target genes in the cluster
    Loop logic needs to be modified: 
    in each iteration
     -  over/'sample from' gex_mat and permute the current vector of gene expression
     -  sample one TF per TF cluster as predictors.
    Count logic: count logic C

    Functions we want to implement:

    Groundtruth = everything is singleton cluster
    1) Subset input_grn: get the relevant tf-target edges. takes lists as input, TFs always all
        targets always the ones represented by the current medioid (A+c)/present in the current cluster (B+C)

    2) Counting logic: takes input GRN, shuffled GNR and mappings between clusters and genes for genes and/or TFs.

    3) Two versions of infer_partial network one for A+C and one for B+D

    4) Reset GRNboost to standard for input_GRN computation and create alternative versions for FDR functionalities with different names.


    '''
    assert client, "Client not given, but is required in create_graph_fdr!"
    # Extract FDR mode information from TF and non-TF representative lists.
    fdr_mode = None
    if len(tf_representatives) + len(non_tf_representatives) == len(gene_names) and not gene_to_cluster is None:
        fdr_mode = 'random'
    elif not gene_to_cluster is None:
        fdr_mode = 'medoid'
    else:  # Full FDR mode coincides with medoid mode, with all genes assigned to dummy singleton clusters.
        fdr_mode = 'medoid'
        print("Running full FDR mode...")
        gene_to_cluster = {gene: cluster_id for cluster_id, gene in
                           enumerate(tf_representatives + non_tf_representatives)}

    # Check if gene_to_cluster is complete, i.e. if for every gene in expression matrix, a corresponding cluster has
    # been precomputed.
    all_genes = {gene for gene, _ in gene_to_cluster.items()}
    assert expression_matrix.shape[1] == len(all_genes), "Size of expression matrix does not match gene names."
    assert len(gene_names) == len(all_genes), "Number of clustered genes and genes in expression matrix do not match."

    # Subset expression matrix to TF representatives ('medoid' mode). Leave as is, if TFs have not been clustered or
    # FDR mode is 'random'.
    tf_matrix, tf_matrix_gene_names = to_tf_matrix(expression_matrix, gene_names, tf_representatives)

    # Partition input GRN into dict storing target-cluster IDs as keys and edge dicts (as in input GRN) as values.
    # Second data structure stores target genes per cluster.
    grn_subsets_per_target, genes_per_target_cluster = partition_input_grn(input_grn, gene_to_cluster)

    future_tf_matrix = client.scatter(tf_matrix, broadcast=True)
    # [1] wrap in a list of 1 -> unsure why but Matt. Rocklin does this often...
    [future_tf_matrix_gene_names] = client.scatter([tf_matrix_gene_names], broadcast=True)

    # Broadcast gene-to-cluster dictionary among all workers.
    # future_gene_to_cluster = client.scatter(gene_to_cluster, broadcast=True) --> gives dict key error...
    [future_gene_to_cluster] = client.scatter([gene_to_cluster], broadcast=True)

    delayed_link_dfs = []  # collection of delayed link DataFrames

    # Use pre-computed medoid representatives for TFs and/or non-TFs.
    if fdr_mode == 'medoid':
        # Loop over all representative targets, i.e. non-TF medoids.
        for target_gene_index in target_gene_indices(gene_names, non_tf_representatives + tf_representatives):
            target_gene_name = gene_names[target_gene_index]
            target_gene_expression = delayed(expression_matrix[:, target_gene_index])
            target_subset_grn = delayed(grn_subsets_per_target[gene_to_cluster[target_gene_name]])

            # Pass subset of GRN which is represented by the medoids.
            delayed_link_df = delayed(count_computation_medoid_representative, pure=True)(
                regressor_type,
                regressor_kwargs,
                future_tf_matrix,
                are_tfs_clustered,
                future_tf_matrix_gene_names,
                target_gene_name,
                target_gene_expression,
                target_subset_grn,
                future_gene_to_cluster,
                n_permutations,
                early_stop_window_length,
                seed,
            )

            if not output_dir is None:
                compute(save_df(delayed_link_df, os.path.join(output_dir, f'target_{target_gene_name}.feather')))

            if delayed_link_df is not None:
                delayed_link_dfs.append(delayed_link_df)

    # Loop over all genes of cluster, i.e. simulate random drawing of genes from clusters.
    elif fdr_mode == 'random':
        # Loop over all target clusters (that includes TF clusters).
        for cluster_id, cluster_targets in genes_per_target_cluster.items():
            target_cluster_idxs = target_gene_indices(gene_names, cluster_targets)
            # Like this, order of cluster gene names should be consistent with order in cluster_idxs and hence with
            # gene column order in expression matrix.
            target_cluster_gene_names = [gene for index, gene in enumerate(gene_names) if index in target_cluster_idxs]
            cluster_expression = delayed(expression_matrix)

            # Dask does not allow iterating over delayed dictionary, so no delayed() at this point.
            target_subset_grn = grn_subsets_per_target[cluster_id]

            # If TFs have been clustered in 'random' mode, then per permutation, one TF per cluster needs to be
            # drawn. Precompute the necessary cluster-TF relationships here such that keys are cluster IDs
            # and values are list of TFs.
            if are_tfs_clustered:
                cluster_to_tfs = invert_tf_to_cluster_dict(tf_representatives, gene_to_cluster)
                [future_cluster_to_tfs] = client.scatter([cluster_to_tfs], broadcast=True)
            else:
                future_cluster_to_tfs = None

            delayed_link_df = delayed(count_computation_sampled_representative, pure=True)(
                regressor_type,
                regressor_kwargs,
                future_tf_matrix,
                future_tf_matrix_gene_names,
                future_cluster_to_tfs,
                target_cluster_gene_names,
                target_cluster_idxs,
                cluster_expression,
                target_subset_grn,
                gene_to_cluster,
                n_permutations,
                early_stop_window_length,
                seed,
            )

            if not output_dir is None:
                compute(save_df(delayed_link_df, os.path.join(output_dir, f'cluster_{cluster_id}.feather')))

            if delayed_link_dfs is not None:
                delayed_link_dfs.append(delayed_link_df)
    else:
        raise ValueError(f'Unknown FDR mode: {fdr_mode}.')

    # Gather the DataFrames into one distributed DataFrame.
    all_links_df = from_delayed(delayed_link_dfs, meta=FDR_GRN_SCHEMA)

    # [2] repartition to nr of workers -> important to avoid GC problems!
    # see: http://dask.pydata.org/en/latest/dataframe-performance.html#repartition-to-reduce-overhead
    n_parts = len(client.ncores()) * repartition_multiplier

    return all_links_df.repartition(npartitions=n_parts)

def count_computation_medoid_representative(
        regressor_type,
        regressor_kwargs,
        tf_matrix,
        are_tfs_clustered,
        tf_matrix_gene_names: list[str],
        target_gene_name: str,
        target_gene_expression: np.ndarray,
        partial_input_grn: dict,  # {(TF, target): {'importance': float}}
        gene_to_clust: dict[str, int],
        n_permutations: int,
        early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
        seed=DEMON_SEED,
):
    # Remove target from TF-list and TF-expression matrix if target itself is a TF
    if not are_tfs_clustered:
        (clean_tf_matrix, clean_tf_matrix_gene_names) = clean(tf_matrix, tf_matrix_gene_names, target_gene_name)
    else:
        clean_tf_matrix, clean_tf_matrix_gene_names = tf_matrix, tf_matrix_gene_names

    # Special case in which only a single TF is passed and the target gene
    # here is the same as the TF (clean_tf_matrix is empty after cleaning):
    if clean_tf_matrix.size == 0:
        raise ValueError("Cleaned TF matrix is empty, skipping inference of target {}.".format(target_gene_name))

    # Initialize counts
    for _, val in partial_input_grn.items():
        val.update({'count': 0.0})

    # Iterate for num permutations
    for _ in range(n_permutations):

        # Shuffle target gene expression vector
        permuted_target_gene_expression = np.random.permutation(target_gene_expression)

        # Train the random forest regressor
        try:
            trained_regressor = fit_model(
                regressor_type,
                regressor_kwargs,
                clean_tf_matrix,
                permuted_target_gene_expression,
                early_stop_window_length,
                seed
            )
        except ValueError as e:
            raise ValueError(
                "Count_computation_medoid: regression for target gene {0} failed. Cause {1}.".format(target_gene_name, repr(e))
            )

        # Construct the shuffled GRN dataframe from the trained regressor
        shuffled_grn_df = to_links_df(
            regressor_type,
            regressor_kwargs,
            trained_regressor,
            clean_tf_matrix_gene_names,
            target_gene_name
        )

        # Update the count values of the partial input GRN
        count_helper(shuffled_grn_df, partial_input_grn, gene_to_clust)

    # Change partial input GRN format from dict to df
    partial_input_grn_fdr_df = pd.DataFrame(
        [(TF, target, v['importance'], v['count']) for (TF, target), v in partial_input_grn.items()],
        columns=['TF', 'target', 'importance', 'count']
    )

    return partial_input_grn_fdr_df


def count_computation_sampled_representative(
        regressor_type,
        regressor_kwargs,
        tf_matrix,
        tf_matrix_gene_names,
        cluster_to_tfs,
        target_gene_names,
        target_gene_idxs,
        target_gene_expressions,
        partial_input_grn: dict,
        gene_to_cluster: dict[str, int],
        n_permutations : int,
        early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
        seed=DEMON_SEED,

):
    are_tfs_clustered = (not cluster_to_tfs is None)
    # Initialize counts on input GRN edges.
    for _, val in partial_input_grn.items():
        val.update({'count': 0.0})

    for perm in range(n_permutations):
        # Retrieve "random" target gene from cluster.
        perm_index = perm % len(target_gene_names)
        target_gene_name = target_gene_names[perm_index]
        target_gene_index = target_gene_idxs[perm_index]
        target_expression = target_gene_expressions[:, target_gene_index]

        # Remove target from TF-list and TF-expression matrix if target itself is a TF
        if not are_tfs_clustered:
            (clean_tf_matrix, clean_tf_matrix_gene_names) = clean(tf_matrix, tf_matrix_gene_names, target_gene_name)
        else:
            clean_tf_matrix, clean_tf_matrix_gene_names = tf_matrix, tf_matrix_gene_names

        # Sample one TF per TF-cluster and subset TF expression matrix in case of TFs having been clustered.
        if not cluster_to_tfs is None:
            tf_representatives = []
            for cluster, tf_list in cluster_to_tfs.items():
                cluster_size = len(tf_list)
                representative = tf_list[perm % cluster_size]
                tf_representatives.append(representative)
            # Subset TF expression matrix.
            clean_tf_matrix, clean_tf_matrix_gene_names = subset_tf_matrix(clean_tf_matrix,
                                                                    clean_tf_matrix_gene_names,
                                                                    tf_representatives)

        # Special case in which only a single TF is passed and the target gene
        # here is the same as the TF (clean_tf_matrix is empty after cleaning):
        if clean_tf_matrix.size == 0:
            raise ValueError("Cleaned TF matrix is empty, skipping inference of target {}.".format(target_gene_name))

        # Shuffle target gene expression vector
        permuted_target_gene_expression = np.random.permutation(target_expression)

        # Train the random forest regressor.
        try:
            trained_regressor = fit_model(
                regressor_type,
                regressor_kwargs,
                clean_tf_matrix,
                permuted_target_gene_expression,
                early_stop_window_length,
                seed
            )
        except ValueError as e:
            raise ValueError(
                "Count_computation_sampled: regression for target gene {0} failed. Cause {1}.".format(target_gene_name, repr(e))
            )

        # Construct the shuffled GRN dataframe from the trained regressor
        shuffled_grn_df = to_links_df(
            regressor_type,
            regressor_kwargs,
            trained_regressor,
            clean_tf_matrix_gene_names,
            target_gene_name
        )

        # Update the count values of the partial input GRN.
        count_helper(shuffled_grn_df, partial_input_grn, gene_to_cluster)

    # Change partial input GRN format from dict to df
    partial_input_grn_fdr_df = pd.DataFrame(
        [(TF, target, v['importance'], v['count']) for (TF, target), v in partial_input_grn.items()],
        columns=['TF', 'target', 'importance', 'count']
    )

    return partial_input_grn_fdr_df

@delayed
def save_df(df, filename):
    df.to_feather(filename)