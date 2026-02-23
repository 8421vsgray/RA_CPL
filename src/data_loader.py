import torch
from torch.utils.data import Dataset, DataLoader
from scipy import io


class MultiViewDataset(Dataset):
    def __init__(self, mat_path, noise_rate=0.0, seed=42, mode='train', split_ratio=0.8, val_ratio=0.1):
        # 1. 独立生成器：确保随机过程局部化，不干扰全局 torch/numpy 种子
        self.rng = torch.Generator().manual_seed(seed)

        # 2. 数据加载
        mat_data = io.loadmat(mat_path)
        X_raw = mat_data['X']
        y_all = torch.from_numpy(mat_data['y']).long().squeeze()
        v1_all = torch.from_numpy(X_raw.flatten()[0]).float()
        v2_all = torch.from_numpy(X_raw.flatten()[1]).float()
        v3_all = torch.from_numpy(X_raw.flatten()[2]).float()

        total_samples = v1_all.shape[0]

        # 3. 严格三集划分策略：由 seed 锁定 full_indices，确保互不重叠
        full_indices = torch.randperm(total_samples, generator=self.rng)
        train_size = int(total_samples * split_ratio)
        val_size = int(total_samples * val_ratio)

        if mode == 'train':
            idx = full_indices[:train_size]
        elif mode == 'val':
            idx = full_indices[train_size: train_size + val_size]
        else:  # test
            idx = full_indices[train_size + val_size:]

        self.y = y_all[idx]
        self.v1_data = v1_all[idx]
        self.v2_data = v2_all[idx]
        self.v3_data = v3_all[idx]
        self.num_samples = len(idx)

        # 4. 均匀 ACN 噪声：仅在 train 且指定 noise_rate 时执行
        self.v1_idx = torch.arange(self.num_samples)
        self.v2_idx = torch.arange(self.num_samples)
        self.v3_idx = torch.arange(self.num_samples)

        if mode == 'train' and noise_rate > 0:
            num_noise = int(self.num_samples * noise_rate)
            # 随机均匀采样污染点
            noise_positions = torch.randperm(self.num_samples, generator=self.rng)[:num_noise]

            # v2/v3 分别打乱，模拟异步错位
            p2 = torch.randperm(num_noise, generator=self.rng)
            p3 = torch.randperm(num_noise, generator=self.rng)

            self.v2_idx[noise_positions] = noise_positions[p2]
            self.v3_idx[noise_positions] = noise_positions[p3]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 异步映射逻辑
        # v1_idx[idx] 永远是原始顺序 0, 1, 2...
        # v2_idx[idx] 可能变成了 500, 12, 99... (如果被污染)

        data = [
            self.v1_data[self.v1_idx[idx]],
            self.v2_data[self.v2_idx[idx]],
            self.v3_data[self.v3_idx[idx]]
        ]

        # 核心：标签必须跟随锚点视图 v1
        # 这样即使 v2, v3 是乱序的，y 依然代表当前 sample 的真实类别
        label = self.y[self.v1_idx[idx]]

        return data, label


def get_loader(mat_path, batch_size=512, noise_rate=0.0, mode='train',
               split_ratio=0.8, val_ratio=0.1, seed=42, verbose=True):
    # 严格比例约束审查
    assert split_ratio + val_ratio < 1.0, "划分比例总和必须小于 1.0"

    dataset = MultiViewDataset(mat_path, noise_rate=noise_rate, seed=seed,
                               mode=mode, split_ratio=split_ratio, val_ratio=val_ratio)

    if verbose and mode == 'train':
        print(f"✅ [DataEngine] Mode: {mode.upper()} | Samples: {len(dataset)} | Noise: {noise_rate * 100}% (Uniform)")
    elif verbose:
        print(f"📊 [DataEngine] Mode: {mode.upper()} | Samples: {len(dataset)} | Clean")

    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=0)