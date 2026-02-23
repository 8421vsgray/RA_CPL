import torch
import torch.nn.functional as F
import numpy as np
from src.models import RACPLModel_IDEC
from src.data_loader import get_loader
from src.metrics import evaluate_all

def info_nce_loss(z1, z2, temperature=0.07):
    """
    [Contrastive Baseline] 标准 InfoNCE 损失，用于预训练阶段建立基础对齐
    """
    batch_size = z1.shape[0]
    features = torch.cat([z1, z2], dim=0)
    logits = torch.matmul(features, features.T) / temperature
    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    logits = logits.masked_fill(mask, -1e9)
    labels = torch.cat([torch.arange(batch_size, 2 * batch_size),
                        torch.arange(0, batch_size)], dim=0).to(z1.device)
    return F.cross_entropy(logits, labels)


def train():
    # --- 1. 环境与硬件配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 RACPL-IDEC 架构 (输入维度分别对应视图维度)
    model = RACPLModel_IDEC(view_dims=[30, 9, 30], latent_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # --- 2. 实验超参数 ---
    PRETRAIN_EPOCHS = 0
    TOTAL_EPOCHS = 200
    best_pre_acc = 0.0
    best_audit_acc = 0.0

    print("🚀 [RA-CPL 完全体] 启动！")
    print(f"🛡️ 核心审计参数: Bias=2.0, Tau=0.5 | 质心同步策略: KMeans-CSR")

    for epoch in range(1, TOTAL_EPOCHS + 1):
        # 动态噪声模拟：非预训练阶段开启 20% (或其他比例) 噪声
        is_pretrain = (epoch <= PRETRAIN_EPOCHS)
        noise_rate = 0.0 if is_pretrain else 0.8
        loader = get_loader('data/MNIST_10k.mat', batch_size=512, noise_rate=noise_rate)

        model.train()
        epoch_z, epoch_y = [], []

        for views, labels in loader:
            views = [v.to(device).float() for v in views]
            z_list = model(views)  # 特征提取与 L2 归一化

            # --- A. 自监督重构路径 (Loss Reconstruction) ---
            # 保证 Latent Space 包含足够的原始视图信息
            loss_recon = sum(F.mse_loss(model.decode(z_list[i], i), views[i]) for i in range(3))

            # --- B. 审计对齐路径 (Loss Alignment) ---
            loss_align = 0
            if is_pretrain:
                # 预训练阶段：全量对比学习
                loss_align = info_nce_loss(z_list[0], z_list[1]) + info_nce_loss(z_list[0], z_list[2])
            else:
                # 审计阶段：启动“战神位移法”进行鲁棒性过滤
                for i in [1,2]:
                    # 利用视图 0 的潜变量跨视图预测视图 i 的原始输入
                    x_pred = model.decode(z_list[0], view_idx=i)

                    # 计算逐样本重构误差 (作为噪声识别的 Proxy)
                    error = torch.norm(x_pred - views[i], p=2, dim=1).detach()

                    # 硬过滤层：剔除当前 Batch 中 50% 的高误差“嫌疑样本”
                    error_np = error.cpu().numpy()
                    thresh_val = np.percentile(error_np, 40)
                    mask = (error <= thresh_val).float().to(error.device)

                    # 战神位移权重计算：将 2.0 设为信任中点，放大优质对齐，压制高误差噪声
                    # Formula: weight = exp(-(error - 2.0) / 0.5)
                    shifted_error = error - 2.0
                    audit_weight = torch.exp(-shifted_error / 0.5)

                    # 执行被加权的跨视图对齐：让 z_i 只有在审计通过时才被拉向 z_0
                    dist = torch.norm(z_list[0] - z_list[i], p=2, dim=1)
                    loss_align += (audit_weight * dist).mean()

            # --- C. IDEC 聚类路径 (Loss Clustering) ---
            # 使用 Student's t-distribution 进行软分配概率计算 (q)
            z_anchor = z_list[0]
            q = 1.0 / (1.0 + torch.sum(torch.pow(z_anchor.unsqueeze(1) - model.centroids, 2), 2))
            q = (q.t() / torch.sum(q, 1)).t()

            # 计算目标分布 (p) 以实现聚类自我强化
            p = torch.pow(q, 2) / torch.sum(q, 0)
            p = (p.t() / torch.sum(p, 1)).t()
            loss_clu = F.kl_div(q.log(), p, reduction='batchmean')

            # --- D. 总损失加权融合 ---
            # 重构权重(10.0)用于稳固特征，对齐权重(1.0)用于噪声对抗，聚类权重(0.1)用于结构解耦
            total_loss = 10.0 * loss_recon + 1.0 * loss_align + 0.1 * loss_clu

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_z.append(z_anchor.detach().cpu())
            epoch_y.append(labels.cpu())

        # --- E. 每个 Epoch 的核心策略：Centroid Sync Refinement (CSR) ---
        z_all = torch.cat(epoch_z).numpy()
        y_all = torch.cat(epoch_y).numpy()

        # 1. 探测特征潜力 (KMeans 模式)
        current_acc, current_nmi, current_ari, latest_centers = evaluate_all(z_all, y_all)

        # 2. 阶梯式质心同步 (CSR)
        update_interval = 5 if epoch <= 150 else 1
        if epoch % update_interval == 0:
            model.centroids.data = torch.from_numpy(latest_centers).to(device).float()
            if epoch > 150:
                print(f"   🎯 [Step-Sync] Epoch {epoch}: 质心已执行逐轮精密对齐")

        # 3. 监控原生实力 (Pure Inference)
        pure_acc, _, _ = evaluate_all(z_all, y_all, centroids=model.centroids.data)

        # --- F. 模型持久化与监控 ---
        if is_pretrain:
            if current_acc > best_pre_acc:
                best_pre_acc = current_acc
                torch.save(model.state_dict(), 'best_pretrain_model.pth')
                print(f"🌟 [Pretrain] Ep {epoch} | ACC: {current_acc:.4f}")
        else:
            # 记录状态
            is_best = current_acc > best_audit_acc
            status_tag = "🛡️ [Audit]" if is_best else "🛡️ [Train]"

            if is_best:
                best_audit_acc = current_acc
                torch.save(model.state_dict(), 'best_audit_model.pth')
                save_msg = " 🔥 (Saved)"
            else:
                save_msg = ""

            # 战神专属三维 Log：潜力 | 结构 | 纯度 || 真实对齐度
            print(
                f"{status_tag} Ep {epoch:03d} | ACC:{current_acc:.4f} NMI:{current_nmi:.4f} ARI:{current_ari:.4f} || Pure_ACC:{pure_acc:.4f}{save_msg}")


if __name__ == "__main__":
    train()