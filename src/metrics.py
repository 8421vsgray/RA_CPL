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
    统一评估标准：
    1. 如果提供 centroids: 执行原生投影指派（Pure Inference）
    2. 如果 centroids 为 None: 执行 KMeans 发现（Potential Analysis）
    返回：dict 格式，彻底解决解包变量个数不匹配的问题
    """
    results = {}

    if centroids is not None:
        # 原生质心映射
        from scipy.spatial.distance import cdist
        dist = cdist(features, centroids, metric='euclidean')
        y_pred = np.argmin(dist, axis=1)
        results['centers'] = centroids
    else:
        # KMeans 发现
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=10, n_init=20, random_state=42).fit(features)
        y_pred = km.labels_
        results['centers'] = km.cluster_centers_

    # 计算三大核心指标
    results['acc'] = cluster_acc(labels, y_pred)
    results['nmi'] = nmi_score(labels, y_pred)
    results['ari'] = ari_score(labels, y_pred)

    return results