from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

import numpy as np
import pandas as pd


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

def get_train(y_kmeans, y_clu):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    if isinstance(y_kmeans, pd.DataFrame):
        y_kmeans = y_kmeans.values

    if isinstance(y_clu, pd.DataFrame):
        y_clu = y_clu.values

    # CJY 2021.6.20 for batch effect dataset
    #y_true = y_true.to_list()
    y_kmeans = y_kmeans.astype(np.int64)
    y_clu = y_clu.astype(np.int64)

    # label_to_number = {label: number for number, label in enumerate(set(y_kmeans))}
    # label_numerical = np.array([label_to_number[i] for i in y_kmeans])
    #
    # y_clu = label_numerical.astype(np.int64)
    assert y_clu.size == y_kmeans.size
    D = max(y_clu.max(), y_kmeans.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_clu.size):
        w[y_clu[i], y_kmeans[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    # ind = linear_assignment(w.max() - w)
    # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    # https://stackoverflow.com/questions/62390517/no-module-named-sklearn-utils-linear-assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    row_ind, col_ind = linear_assignment(w.max() - w)
    train_index = np.array([])
    for i, j in zip(row_ind, col_ind):
        # print(np.where(y_clu==i)[-1])
        # print(np.where(y_kmeans==j)[-1])
        temp_index = np.intersect1d(np.where(y_clu==i)[-1], np.where(y_kmeans==j)[-1])
        train_index = np.concatenate((train_index, temp_index), axis=0)
    return train_index


# 2. Jaccard score (JS)
# Please refer to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
def Jaccard_index(y_true, y_pred):
    from sklearn.metrics.cluster import pair_confusion_matrix
    contingency = pair_confusion_matrix(y_true, y_pred)
    JI = contingency[1,1]/(contingency[1,1]+contingency[0,1]+contingency[1,0])
    return JI

def compute_metrics(embs, y_true, y_pred):
    metrics = {}
    metrics["ARI"] = round(ARI(y_true, y_pred),3)
    metrics["NMI"] = round(NMI(y_true, y_pred),3)
    metrics["CA"] = round(cluster_acc(y_true, y_pred),3)
    metrics["JI"] = round(Jaccard_index(y_true, y_pred),3)
    metrics["CS"] = round(completeness_score(y_true, y_pred),3)
    metrics["HS"] = round(homogeneity_score(y_true, y_pred),3)
    metrics["VMS"] = round(v_measure_score(y_true, y_pred),3)
    metrics["Sil"] = round(silhouette_score(embs, y_pred),3)
    metrics["CH"] = round(calinski_harabasz_score(embs, y_pred),3)

    return metrics