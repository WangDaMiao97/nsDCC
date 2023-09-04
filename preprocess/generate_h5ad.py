import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import os

from collections import Counter

parser = argparse.ArgumentParser(description='PyTorch scRNA-seq format transformation and preprocessing')

# input & ouput
parser.add_argument('--input_h5ad_path', type=str, default=None,
                    help='path to input h5ad file')
parser.add_argument('--input_10X_path', type=str, default=None,
                    help='path to input 10X file')
parser.add_argument('--count_csv_path', type=str, default=None,
                    help='path to counts csv file')
parser.add_argument('--label_csv_path', type=str, default=None,
                    help='path to labels csv file')
parser.add_argument('--save_h5ad_dir', type=str, default=None,
                    help='dir to savings')

# preprocessing
parser.add_argument('--filter', type=bool, default=True,
                    help='Whether do filtering')
parser.add_argument('--norm', type=bool, default=True,
                    help='Whether do normalization')
parser.add_argument("--log", type=bool, default=True,
                    help='Whether do log operation')
parser.add_argument("--scale", type=bool, default=True,
                    help='Whether do scale operation')
parser.add_argument("--select_hvg", type=bool, default=True,
                    help="Whether select highly variable gene")


def preprocess_csv_to_h5ad(
        input_h5ad_path=None, input_10X_path=None, count_csv_path=None, label_csv_path=None, save_h5ad_dir="./",
        do_filter=False, do_log=False, do_select_hvg=False, do_norm=False, do_scale=False
):
    # read data from h5ad.
    adata = sc.read_h5ad(input_h5ad_path)
    type_counter = Counter(adata.obs.Group)
    type_mask = {k: v for k, v in type_counter.items() if v < 10}
    low_type_MASK = ~adata.obs.Group.isin(['NA'])
    adata = adata[low_type_MASK]
    low_type_MASK = ~adata.obs.Group.isin(list(type_mask.keys()))
    adata = adata[low_type_MASK]
    print("Read data from h5ad file: {}".format(input_h5ad_path))

    _, h5ad_file_name = os.path.split(input_h5ad_path)
    save_file_name = h5ad_file_name

    # preprocess anndata
    preprocessed_flag = do_filter | do_log | do_select_hvg | do_norm | do_scale
    # filter operation
    if do_filter == True:
        # basic filtering, filter the genes and cells
        print(adata)
        mito_genes = adata.var_names.str.startswith('MT-')
        adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
        adata.obs['n_counts'] = adata.X.sum(axis=1)
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)

        # calculate the qc metrics, then do the normalization
        # filter the mitochondrial genes
        adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        low_MT_MASK = (adata.obs.pct_counts_mt < 5)
        adata = adata[low_MT_MASK]

        # filter the ERCC spike-in RNAs
        adata.var['ERCC'] = adata.var_names.str.startswith('ERCC-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['ERCC'], percent_top=None, log1p=False, inplace=True)
        low_ERCC_mask = (adata.obs.pct_counts_ERCC < 10)
        adata = adata[low_ERCC_mask]

    if do_norm == True:
        sc.pp.normalize_total(adata, target_sum=1e4)

    if do_log and np.max(adata.X > 100):
        sc.pp.log1p(adata)
        if do_select_hvg:
            sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            adata = adata[:, adata.var.highly_variable]
            adata.raw = adata
    else:
        if do_select_hvg and not do_log:
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
            adata = adata[:, adata.var.highly_variable]
            adata.raw = adata
            print("HVG length:", len(adata.var.highly_variable))

    if do_scale == True:
        sc.pp.scale(adata, max_value=10, zero_center=False)

    # save preprocessed h5ad
    if save_h5ad_dir is not None:
        if os.path.exists(save_h5ad_dir) != True:
            os.makedirs(save_h5ad_dir)

        if preprocessed_flag == True:
            save_file_name = save_file_name.replace(".h5ad", "_preprocessed.h5ad")
        save_path = os.path.join(save_h5ad_dir, save_file_name)
        adata.write(save_path)
        print("Successfully generate preprocessed file: {}".format(save_file_name))

    return adata

if __name__=="__main__":
    args = parser.parse_args()
    args.input_h5ad_path = "../dataset/Klein.h5ad"
    args.save_h5ad_dir = "../dataset/"

    processed_adata = preprocess_csv_to_h5ad(
        args.input_h5ad_path, args.input_10X_path, args.count_csv_path, args.label_csv_path, args.save_h5ad_dir,
        do_filter=args.filter, do_log=args.log, do_norm=args.norm, do_select_hvg=args.select_hvg, do_scale=args.scale
    )
