import os
import torch
import warnings
from ddranking.metrics import HardLabelEvaluator
from ddranking.config import Config
warnings.filterwarnings("ignore")

""" Use config file to specify the arguments (Recommended) """
config = Config.from_file("./configs/DM_Hard_Label.yaml")
hard_label_evaluator = HardLabelEvaluator(config)

syn_data_dir = "/root/DD-Ranking/baselines/DM/CIFAR10/IPC50/"
data = torch.load(os.path.join(syn_data_dir, f"images.pt"), map_location='cpu')

# 提取第一组图像和标签数据
syn_images = data['data'][0][0]  # 第一个子列表的第一个tensor(图像)
syn_labels = data['data'][0][1]  # 第一个子列表的第二个tensor(标签)

print("提取的数据信息:")
print(f"图像形状: {syn_images.shape}")
print(f"标签形状: {syn_labels.shape}")

# 验证标签范围
assert syn_labels.min() >= 0 and syn_labels.max() <= 9, "CIFAR-10标签应为0-9"

# 进行评估
syn_lr = 0.01
metrics = hard_label_evaluator.compute_metrics(
    image_tensor=syn_images,
    syn_lr=syn_lr
)
print("\n评估结果:")
print(metrics)