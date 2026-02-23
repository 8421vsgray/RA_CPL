# RA-CPL: Robust Audit - Cross-Predictive Learning for Multi-view Clustering


**RA-CPL** 是一种专为极高噪声环境下设计的深度多视图聚类架构。在面对 **80% 标签噪声** 的极端挑战时，该架构通过动态审计与跨视图对齐机制，实现了从混沌中恢复语义秩序的能力。

---

## 🌟 核心特性 (Key Features)

* **🛡️ 动态位移审计 (Dynamic Shifted Audit, DSA)**: 独创指数权重衰减函数，自动识别并过滤 80% 的干扰噪声。
* **🔗 跨视图一致性对齐 (Cross-view Alignment)**: 利用 View-0 预测 View-1/2 的重构误差作为信任代理，实现高质量特征蒸馏。
* **🎯 质心同步细化 (Centroid Synchronous Refinement, CSR)**: 消除对离线 KMeans 的依赖，实现端到端（End-to-End）的原生模型预测。
* **🔥 极致鲁棒性**: 在 MNIST 80% 噪声设置下，原生预测精度 (Pure ACC) 突破 **0.7072**。

---

## 🏗️ 架构概览 (Architecture)

RA-CPL 架构由三个核心维度组成，确保模型在噪声海洋中不迷失方向：

1.  **特征提取层**: 多视图专用 Encoder，将原始输入映射至单位超球面潜空间。
2.  **跨视图预测审计**: 
    - 使用 $z_0$ 尝试还原 $x_1, x_2$。
    - 依据公式 $W = \exp(-(error - Bias) / \tau)$ 计算样本可信度。
3.  **聚类决策层**: 内置 IDEC 聚类头，配合 CSR 策略动态校准全局质心。



---

## 📊 实验表现 (Experimental Results)

在 **MNIST 10k** 数据集上，设定 **80% 噪声率** 的挑战性实验结果如下：

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Pure ACC** (原生精度) | **0.7072** | ✅ 史诗级封神 |
| **Pure NMI** (归一化互信息) | **0.6655** | ✅ 结构逻辑清晰 |
| **Pure ARI** (调整兰德系数) | **0.5965** | ✅ 纯度表现卓越 |

> **注**: 以上指标均为 **Zero-KMeans Dependency** 下的模型原生推理结果。



---

## 🚀 快速启动 (Getting Started)

### 1. 环境配置
```bash
git clone [https://github.com/8421vsgray/RA-CPL.git](https://github.com/8421vsgray/RA-CPL.git)
cd RA-CPL
conda create -n ra_cpl python=3.10
pip install -r requirements.txt