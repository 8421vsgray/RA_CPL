import torch
from torch.utils.data import Dataset, DataLoader
from scipy import io
import numpy as np


class MultiViewDataset(Dataset):
    def __init__(self, mat_path, noise_rate=0.0, seed=42):
        """
        mat_path: .mat 文件路径
        noise_rate: 乱序率 (0.0 - 1.0)，1.0 代表 100% 全乱序
        seed: 随机种子，确保实验可复现
        """
        # 设置随机种子，保证索引打乱后实验仍可复现
        torch.manual_seed(seed)
        np.random.seed(seed)

        mat_data = io.loadmat(mat_path)
        X_raw = mat_data['X']

        # 基础数据转换
        self.y = torch.from_numpy(mat_data['y']).long().squeeze()
        self.v1_data = torch.from_numpy(X_raw.flatten()[0]).float()
        self.v2_data = torch.from_numpy(X_raw.flatten()[1]).float()
        self.v3_data = torch.from_numpy(X_raw.flatten()[2]).float()

        self.num_samples = self.v1_data.shape[0]
        self.noise_rate = noise_rate

        # --- 战神逻辑：全员大乱斗索引生成 ---
        # 视图 1 保持原始采样顺序 (作为标签参照物，但模型不知道)
        self.v1_idx = torch.arange(self.num_samples)

        # 视图 2 和 3 的索引
        self.v2_idx = torch.arange(self.num_samples)
        self.v3_idx = torch.arange(self.num_samples)

        if noise_rate > 0:
            # 计算需要打乱的数量
            num_shuffle = int(self.num_samples * noise_rate)

            # 生成打乱的子序列
            # 注意：即便噪声率相同，v2 和 v3 也会被打乱成不同的顺序，确保全异步
            shuffle_part2 = torch.randperm(num_shuffle)
            shuffle_part3 = torch.randperm(num_shuffle)

            # 如果 noise_rate < 1.0，我们只打乱后半部分
            start_idx = self.num_samples - num_shuffle
            self.v2_idx[start_idx:] = shuffle_part2 + start_idx
            self.v3_idx[start_idx:] = shuffle_part3 + start_idx

            # 【终极自证：视图 1 也打乱】
            # 如果你想要彻底去中心化，可以取消下面这行的注释
            # self.v1_idx = torch.randperm(self.num_samples)


        else:
            print(f"✅ [数据引擎] 理想状态：三视图完美对齐")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 重点：根据三套独立的索引，从原始数据中抓取特征
        # 哪怕 idx 相同，拿到的 v1, v2, v3 也是原本属于不同行的数据
        data = [
            self.v1_data[self.v1_idx[idx]],
            self.v2_data[self.v2_idx[idx]],
            self.v3_data[self.v3_idx[idx]]
        ]

        # 这里的标签 y 始终跟随 v1 的物理索引
        # 这样模型如果能通过共识对齐 v2/v3，最终预测结果就会向 y 靠拢
        label = self.y[self.v1_idx[idx]]

        return data, label


def get_loader(mat_path, batch_size=512, noise_rate=1.0):
    dataset = MultiViewDataset(mat_path, noise_rate=noise_rate)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时的 Shuffle 增加 Batch 随机性
        num_workers=0  # 如果在 Windows 建议设为 0
    )