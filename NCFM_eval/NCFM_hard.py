import warnings
import argparse
from ddranking.metrics import HardLabelEvaluator
from ddranking.config import Config
from decode import load_synthetic_data
from config import get_params

# 关闭警告
warnings.filterwarnings("ignore")

# 配置文件读取
config = Config.from_file("../configs/Demo_Hard_Label.yaml")
hard_label_evaluator = HardLabelEvaluator(config)

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Choose dataset and data path for synthetic data")
    
    # 添加数据集选择和数据路径参数
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10', help='Choose dataset: cifar10 or cifar100')
    parser.add_argument('--data_path', type=str, default='./distilled_data_ddranking/data_init.pt', help='Path to synthetic data')

    return parser.parse_args()

# 读取命令行参数
args = parse_args()

params = get_params(args.dataset)
# 加载数据
data_dec, target_dec = load_synthetic_data(args.data_path, params)

# 计算指标
hard_label_evaluator.compute_metrics(
    image_tensor=data_dec,
    hard_labels=target_dec,
    syn_lr=params["syn_lr"]
)
