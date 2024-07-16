# -*- coding: utf-8 -*-
'''
@Time    : 2024/6/24 8:24
@Author  : Linjie Wang
@FileName: clustering performance eval.py
@Software: PyCharm
'''
import pandas as pd
import numpy as np
import scanpy as sc
from termcolor import cprint
import argparse

import random
import torch
from models import GCL_clu

from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def get_args(dataset_name, dataset_class, log_path , train_path, seed) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parser for Simple Unsupervised Graph Representation Learning')
    # Basics
    parser.add_argument("--seed", default=seed)
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 32)')
    parser.add_argument('--log_path', type=str, default=log_path)

    # Dataset
    parser.add_argument("--dataset_class", default=dataset_class,
                        help='labels for data holding classified information')
    parser.add_argument("--dataset_name", default=dataset_name)
    # Train
    parser.add_argument('--train_epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--train_lr', default=0.0025, type=float,
                        help='initial learning rate of train')
    parser.add_argument('--train_path', type = str, default=train_path,
                        help='Save the model after training has finished. If no file '
                             'is available run pre-training again.')
    parser.add_argument('--batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--T', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parser.add_argument('--eval_freq', default=50, type=int, metavar='N',
                        help='Save frequency (default: 10)')

    parser.add_argument("--hidden_dim", default=128, type=int, nargs="+")
    parser.add_argument("--clu_cfg", default=[128, 64], type=int, nargs="+")
    parser.add_argument("--fea_dropout", default=0.2, type=float)
    parser.add_argument("--clu_dropout", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.0001, type=float)

    args = parser.parse_args()
    return args
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values

    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    # CJY 2021.6.20 for batch effect dataset
    #y_true = y_true.to_list()
    y_pred = y_pred.astype(np.int64)

    label_to_number = {label: number for number, label in enumerate(set(y_true))}
    label_numerical = np.array([label_to_number[i] for i in y_true])

    y_true = label_numerical.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    # ind = linear_assignment(w.max() - w)
    # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    # https://stackoverflow.com/questions/62390517/no-module-named-sklearn-utils-linear-assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    row_ind, col_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

def Jaccard_index(y_true, y_pred):
    from sklearn.metrics.cluster import pair_confusion_matrix
    contingency = pair_confusion_matrix(y_true, y_pred)
    JI = contingency[1,1]/(contingency[1,1]+contingency[0,1]+contingency[1,0])
    return JI

# def proportion_non_overlapping_clusters(X, labels):
#     n_clusters = len(np.unique(labels))
#     proportions = []
#     for i in range(n_clusters):
#         intra_distances = []
#         inter_distances = []
#         for j in range(n_clusters):
#             if i != j:
#                 intra_distances.append(np.mean([np.linalg.norm(x1 - x2) for x1 in X[labels == i] for x2 in X[labels == i]]))
#                 inter_distances.append(np.mean([np.linalg.norm(x1 - x2) for x1 in X[labels == i] for x2 in X[labels == j]]))
#         proportions.append(np.mean([1 if intra < min(inter_distances) else 0 for intra in intra_distances]))
#     return np.mean(proportions)


def proportion_non_overlapping_clusters(X, labels, k=5):
    n_clusters = len(np.unique(labels))
    total_points = X.shape[0]
    overlap_count = 0
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        other_points = X[labels != i]

        inter_distances = cdist(cluster_points, other_points, 'euclidean')
        # Calculation of the average distance from the point in one cluster to its k-nearest neighbors in the other clusters.
        mean_inter_distances = np.mean(np.sort(inter_distances, axis=1)[:, :k], axis=1)
        # Calculation of the average distance from the point in one cluster to its k-nearest neighbors in the same cluster.
        intra_distances = cdist(cluster_points, cluster_points, 'euclidean')
        max_intra_distances = np.mean(np.sort(intra_distances, axis=1)[:, 1:k+1], axis=1)
        # Determine whether the point is marked as overlapping based on the intra-cluster nearest neighbor distance and inter-cluster nearest neighbor distance.
        overlap_count += np.sum(mean_inter_distances < max_intra_distances)

    # Calculation of the proportion of overlapping points
    overlap_proportion = overlap_count / total_points
    return 1 - overlap_proportion


def compute_metrics(embs, y_true, y_pred):
    metrics = {}
    metrics["ARI"] = round(ARI(y_true, y_pred),3)
    metrics["NMI"] = round(NMI(y_true, y_pred),3)
    metrics["CA"] = round(cluster_acc(y_true, y_pred),3)
    metrics["JI"] = round(Jaccard_index(y_true, y_pred),3)
    metrics["NOP"] = round(proportion_non_overlapping_clusters(embs, y_pred),3)
    return metrics

if __name__=="__main__":

    dataset_name = "Klein"
    seed = 8

    dataset_path = "dataset/{}_preprocessed.h5ad".format(dataset_name)
    dataset_class = "Group"
    log_path = "logs/{}_preprocessed_train.txt".format(dataset_name)
    # 随机种子

    cprint("## Loading Dataset ##", "yellow")
    adata = sc.read_h5ad(dataset_path)
    print("*** {} ***".format(dataset_name))

    train_path = "logs/{}_preprocessed_train_model.pth".format(dataset_name)
    args = get_args(
        # information of dataset
        dataset_name=dataset_name, dataset_class=dataset_class,
        log_path=log_path, train_path=train_path,
        # random seed
        seed=seed)

    # ===================================================#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # ===================================================#
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    running_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ===================================================#
    num_cluster = len(list(set(adata.obs[dataset_class])))
    args.clu_cfg.append(num_cluster)

    # =======================Create Model============================#
    num_feature = adata.shape[1]
    model = GCL_clu(num_feature, out_channel=args.hidden_dim, fea_dropout=0.2,
                    clu_cfg=args.clu_cfg, clu_dropout=0.2, clu_batch_norm=True)
    model.to(running_device)
    model.load_state_dict(torch.load(args.train_path))

    model.eval()
    embs_a, pred_a_lables = model(torch.tensor(adata.X.toarray()).to(running_device))

    gt_labels = adata.obs[dataset_class]

    if isinstance(adata.X, np.ndarray):
        features = adata.X
    else:
        features = adata.X.toarray()
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(features)
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(pca_result)

    if isinstance(pred_a_lables, np.ndarray):
        cluHead_eval_supervised_metrics = compute_metrics(tsne_result, gt_labels, pred_a_lables.argmax(axis=1))
    else:
        cluHead_eval_supervised_metrics = compute_metrics(tsne_result, gt_labels,
                                                          pred_a_lables.argmax(dim=1).cpu().detach().numpy())
    print("nsDCC {}".format(cluHead_eval_supervised_metrics))