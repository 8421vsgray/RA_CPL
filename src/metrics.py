import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size


def evaluate_all(features, labels, centroids=None):
    """
    一键输出 ACC, NMI, ARI。
    如果 centroids 为 None，跑 KMeans (上限)；
    如果 centroids 有值，跑距离指派 (原生实力)。
    """
    if centroids is not None:
        # 指派模式 (Pure Inference)
        dist = cdist(features, centroids.detach().cpu().numpy(), metric='euclidean')
        y_pred = np.argmin(dist, axis=1)
    else:
        # KMeans 模式 (Feature Bound)
        km = KMeans(n_clusters=10, n_init=20, random_state=42).fit(features)
        y_pred = km.labels_
        return cluster_acc(labels, y_pred), nmi_score(labels, y_pred), ari_score(labels, y_pred), km.cluster_centers_

    return cluster_acc(labels, y_pred), nmi_score(labels, y_pred), ari_score(labels, y_pred)