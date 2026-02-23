import faiss
import numpy as np
import torch

class GradientAuditor:
    def __init__(self, latent_dim=128, num_clusters=10):
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        self.index = faiss.IndexFlatL2(latent_dim)
        self.centroids = None

    def audit_and_align(self, z_list):
        device = z_list[0].device
        # 1. 锚点定江山：只用视图 1 训练质心
        anchor_features = z_list[0].detach().cpu().numpy()
        kmeans = faiss.Kmeans(self.latent_dim, self.num_clusters, niter=25, verbose=False)
        kmeans.train(anchor_features)
        self.centroids = kmeans.centroids  # 存入实例属性，供评估调用

        self.index.reset()
        self.index.add(self.centroids)

        # 2. 计算各视图到质心的距离
        view_dists = []
        for z in z_list:
            z_np = z.detach().cpu().numpy()
            dists, _ = self.index.search(z_np, 1)
            view_dists.append(torch.from_numpy(dists).to(device))

        # 3. 寻找全视图最优锚点 (竞标机制)
        combined_dists = torch.cat(view_dists, dim=1)
        best_dists, _ = torch.min(combined_dists, dim=1, keepdim=True)

        # 4. 生成硬核审计掩码 (严苛模式：只有最接近真理的样本才有资格产生梯度)
        masks = [(d <= best_dists * 1.1).float() for d in view_dists]

        return masks, self.centroids