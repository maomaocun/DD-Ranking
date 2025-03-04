import numpy as np

# 假设我们有以下准确率（这些可以从训练的模型中获得）
Acc_real_hard = 0.8789  # 真实数据集带硬标签的准确率
Acc_syn_hard = 0.7026  # 合成数据集带硬标签的准确率
Acc_syn_any = 0.7026  # 合成数据集（硬标签或软标签）准确率
Acc_rdm_any = 0.5226  # 随机选择数据集的准确率（相同压缩比）

# 计算 HLR
HLR = Acc_real_hard - Acc_syn_hard
print(f"HLR (Hard Label Recovery): {HLR:.4f}")

# 计算 IOR
IOR = Acc_syn_any - Acc_rdm_any
print(f"IOR (Improvement Over Random): {IOR:.4f}")

# 默认权重 w = 0.5
w = 0.5

# 计算 DDRS
alpha = w * IOR - (1 - w) * HLR
DDRS = (np.exp(alpha) - np.exp(-1)) / (np.exp(1) - np.exp(-1))

print(f"DDRS (DD-Ranking Score): {DDRS:.4f}")