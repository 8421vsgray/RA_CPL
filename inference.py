import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.spatial.distance import cdist  # 用于矩阵化计算欧氏距离
from src.models import RACPLModel_IDEC
from src.data_loader import get_loader
from src.metrics import cluster_acc


def inference_and_validate():
    """
    RA-CPL 端到端预测校验脚本 (全指标三维验证版)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 架构还原 ---
    model = RACPLModel_IDEC(view_dims=[30, 9, 30], latent_dim=128).to(device)

    # --- 2. 权重加载 ---
    model_path = 'best_audit_model.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ [指标对齐成功] 已加载权重：{model_path}")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    # --- 3. 测试环境准备 ---
    loader = get_loader('data/MNIST_10k.mat', batch_size=512, noise_rate=0.0)

    all_features = []
    all_labels = []

    print("\n🚀 正在提取潜空间特征并执行原生质心投影...")

    with torch.no_grad():
        for views, labels in loader:
            v1 = views[0].to(device).float()
            # 提取特征并执行 L2 归一化（战神位移法的核心：单位超球面分布）
            z1_raw = model.encoders[0](v1)
            z1 = F.normalize(z1_raw, p=2, dim=1)

            all_features.append(z1.cpu().numpy())
            all_labels.append(labels.numpy())

    # 汇总数据
    z_final = np.concatenate(all_features)
    y_true = np.concatenate(all_labels)

    # --- 4. 原生质心指派 (Pure Inference Logic) ---
    # 获取模型内置的 10 个质心
    centroids = model.centroids.detach().cpu().numpy()

    # 使用 cdist 计算样本与质心的距离，比 torch.cdist 在 CPU 汇总时更稳
    dist = cdist(z_final, centroids, metric='euclidean')
    y_pred = np.argmin(dist, axis=1)

    # --- 5. 计算三大核心指标 ---
    final_acc = cluster_acc(y_true, y_pred)
    final_nmi = nmi_score(y_true, y_pred)
    final_ari = ari_score(y_true, y_pred)

    # --- 6. 最终战神校验报告 ---
    print("\n" + "★" * 60)
    print(f"📊  RA-CPL 最终校验报告 (真·端到端全指标版)")
    print("-" * 60)
    print(f"🔥 原生预测精度 (Pure ACC): {final_acc:.4f}")
    print(f"🧩 归一化互信息 (Pure NMI): {final_nmi:.4f}")
    print(f"💎 调整兰德系数 (Pure ARI): {final_ari:.4f}")
    print("-" * 60)
    print(f"🕒 运行状态: 0% 噪声测试 | 100% 神经网络判别 (脱离 KMeans)")

    # 评级逻辑
    if final_acc > 0.8:
        grade = "【传说·封神成功】"
    elif final_acc > 0.7:
        grade = "【史诗·单挑 ROLL 成功】"
    else:
        grade = "【精英·仍在进化中】"

    print(f"📢 战力判定: {grade}")
    print("★" * 60)


if __name__ == "__main__":
    inference_and_validate()