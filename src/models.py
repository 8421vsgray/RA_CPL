import torch
import torch.nn as nn
import torch.nn.functional as F

class RACPLModel_IDEC(nn.Module):
    def __init__(self, view_dims, latent_dim=128, n_clusters=10):
        super(RACPLModel_IDEC, self).__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, latent_dim))
            for dim in view_dims
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, dim))
            for dim in view_dims
        ])
        # 质心初始化
        self.centroids = nn.Parameter(torch.randn(n_clusters, latent_dim))
        nn.init.xavier_uniform_(self.centroids.data)

    # --- 新增战神级接口：统一编码逻辑 ---
    def encode(self, x, view_idx):
        """
        专门用于推理和评估的编码接口
        确保特征被映射到 L2 单位超球面上
        """
        z = self.encoders[view_idx](x)
        return F.normalize(z, p=2, dim=1)

    def forward(self, views):
        z_list = []
        for i, v in enumerate(views):
            # 调用内部定义的 encode 保证逻辑复用
            z_norm = self.encode(v, i)
            z_list.append(z_norm)
        return z_list

    def decode(self, z, view_idx):
        return self.decoders[view_idx](z)