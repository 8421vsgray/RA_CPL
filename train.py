import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from src.models import RACPLModel_IDEC
from src.data_loader import get_loader
from src.metrics import evaluate_all


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 物理隔离种子，确保 DataEngine 划分一致
    DATA_SEED = 42

    # --- 1. 严格三路加载器调用 ---
    # 训练集：80%，带 ACN 噪声
    train_loader = get_loader('data/MNIST_10k.mat', batch_size=512, noise_rate=0.9,
                              mode='train', seed=DATA_SEED, split_ratio=0.8, val_ratio=0.1)

    # 验证集：10%，干净 (用于决定是否保存 best_model.pth)
    val_loader = get_loader('data/MNIST_10k.mat', batch_size=512, noise_rate=0.0,
                            mode='val', seed=DATA_SEED, split_ratio=0.8, val_ratio=0.1)

    # 测试集：10%，干净 (终极审判，不参与保存决策)
    test_loader = get_loader('data/MNIST_10k.mat', batch_size=512, noise_rate=0.0,
                             mode='test', seed=DATA_SEED, split_ratio=0.8, val_ratio=0.1)

    # 模型初始化
    model = RACPLModel_IDEC(view_dims=[30, 9, 30], latent_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    TOTAL_EPOCHS = 100
    best_val_acc = 0.0  # 核心：保存基准改为验证集 ACC
    save_path = 'best_model_strict.pth'

    # 启动前清理旧权重，防止加载到“幽灵 0.72”
    if os.path.exists(save_path): os.remove(save_path)

    print(f"🚀 [RA-CPL 严谨版] 启动 | 噪声率: 90% | Val-Save 逻辑锁定")

    for epoch in range(1, TOTAL_EPOCHS + 1):
        # --- A. 训练阶段 (含审计对齐) ---
        model.train()
        epoch_z_train = []

        for views, _ in train_loader:
            views = [v.to(device).float() for v in views]
            z_list = model(views)
            z_anchor = z_list[0]

            # 1. 重构损失
            loss_recon = sum(F.mse_loss(model.decode(z_list[i], i), views[i]) for i in range(3))

            # 2. 战神审计对齐逻辑
            loss_align = 0
            for i in [1, 2]:
                x_pred = model.decode(z_anchor, view_idx=i)
                error = torch.norm(x_pred - views[i], p=2, dim=1).detach()
                # 审计位移：Bias=2.0, Tau=0.5
                audit_weight = torch.exp(-(error - 2.0) / 0.5)
                dist = torch.norm(z_anchor - z_list[i], p=2, dim=1)
                loss_align += (audit_weight * dist).mean()

            # 3. IDEC 聚类损失
            q = 1.0 / (1.0 + torch.sum(torch.pow(z_anchor.unsqueeze(1) - model.centroids, 2), 2))
            q = (q.t() / torch.sum(q, 1)).t()
            p = torch.pow(q, 2) / torch.sum(q, 0)
            p = (p.t() / torch.sum(p, 1)).t()
            loss_clu = F.kl_div(q.log(), p, reduction='batchmean')

            total_loss = 10.0 * loss_recon + 1.0 * loss_align + 1.0 * loss_clu
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_z_train.append(z_anchor.detach().cpu())

        # --- B. 质心同步 (CSR) ---
        if epoch % 5 == 0:
            z_all_train = torch.cat(epoch_z_train).numpy()
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=10, n_init=10, random_state=42).fit(z_all_train)
            model.centroids.data = torch.from_numpy(km.cluster_centers_).to(device).float()

        # --- C. 评估函数定义 (复用逻辑) ---
        def run_eval(loader):
            model.eval()
            z_list, y_list = [], []
            with torch.no_grad():
                for v, l in loader:
                    z = model.encode(v[0].to(device).float(), 0)
                    z_list.append(z.cpu().numpy())
                    y_list.append(l.numpy())
            vz, vy = np.concatenate(z_list), np.concatenate(y_list)
            # 基于当前质心的原生实力
            return evaluate_all(vz, vy, centroids=model.centroids.detach().cpu().numpy())

        # --- D. 验证与保存 (Val Set) ---
        val_res = run_eval(val_loader)
        val_acc = val_res['acc']

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            flag = "🌟 SAVED"
        else:
            flag = ""

        # --- E. 盲测报告 (Test Set) ---
        test_res = run_eval(test_loader)
        test_acc = test_res['acc']

        print(f"Ep {epoch:03d} | Val_ACC: {val_acc:.4f} | Test_ACC: {test_acc:.4f} | {flag}")

    print("\n" + "★" * 60)
    print(f"✅ 训练结束。最终最优验证集精度: {best_val_acc:.4f}")
    print(f"📊 请运行独立推理脚本加载 {save_path} 进行最终测试。")
    print("★" * 60)


if __name__ == "__main__":
    train()