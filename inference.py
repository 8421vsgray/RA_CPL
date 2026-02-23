import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.spatial.distance import cdist
from src.models import RACPLModel_IDEC
from src.data_loader import get_loader
from src.metrics import cluster_acc


def inference_and_validate():
    """
    RA-CPL 端到端预测校验脚本 - 严谨版
    验证模型在『从未见过』且『完全干净』的测试集上的表现
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 架构还原 ---
    model = RACPLModel_IDEC(view_dims=[30, 9, 30], latent_dim=128).to(device)

    # --- 2. 权重加载 ---
    # 建议加载刚才跑出的 best_model_strict.pth
    model_path = 'best_model_strict.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ [校验引擎] 已成功加载严谨版权重：{model_path}")
    except Exception as e:
        print(f"❌ 权重加载失败，请检查路径。报错: {e}")
        return

    # --- 3. 准备测试数据 (Test Mode) ---
    # 这里必须用 mode='test'，确保获取的是那 20% 独立且干净的数据
    loader = get_loader('data/MNIST_10k.mat', batch_size=512, noise_rate=0.0, mode='test')

    all_features = []
    all_labels = []

    print(f"🚀 正在提取测试集特征 (N={len(loader.dataset)})...")

    with torch.no_grad():
        for views, labels in loader:
            # 仅使用 Anchor View (View 0) 进行推理
            v1 = views[0].to(device).float()

            # 使用 model.encode 进行特征提取（内部包含 L2 归一化）
            z1 = model.encode(v1, view_idx=0)

            all_features.append(z1.cpu().numpy())
            all_labels.append(labels.numpy())

    # 汇总数据
    z_final = np.concatenate(all_features)
    y_true = np.concatenate(all_labels)

    # --- 4. 原生质心指派 (Pure Inference Logic) ---
    # 直接提取模型训练好的 centroids 矩阵
    centroids = model.centroids.detach().cpu().numpy()

    # 计算欧氏距离矩阵 [N, 10]
    dist = cdist(z_final, centroids, metric='euclidean')
    # 指派最近的质心编号作为预测类别
    y_pred = np.argmin(dist, axis=1)

    # --- 5. 计算三大核心指标 ---
    final_acc = cluster_acc(y_true, y_pred)
    final_nmi = nmi_score(y_true, y_pred)
    final_ari = ari_score(y_true, y_pred)

    # --- 6. 最终战神校验报告 ---
    print("\n" + "★" * 60)
    print(f"📊  RA-CPL 最终校验报告 (泛化性能评估)")
    print("-" * 60)
    print(f"🔥 原生预测精度 (Pure ACC): {final_acc:.4f}")
    print(f"🧩 归一化互信息 (Pure NMI): {final_nmi:.4f}")
    print(f"💎 调整兰德系数 (Pure ARI): {final_ari:.4f}")
    print("-" * 60)
    print(f"🕒 校验环境: 独立测试集 | 0% 噪声 | 纯距离映射")

    # 战力评级
    if final_acc > 0.8:
        grade = "【传说·泛化超神】"
    elif final_acc > 0.7:
        grade = "【史诗·硬抗乱序成功】"
    else:
        grade = "【精英·特征空间仍需压缩】"

    print(f"📢 战力判定: {grade}")
    print("★" * 60)


if __name__ == "__main__":
    inference_and_validate()