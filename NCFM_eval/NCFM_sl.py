import warnings
import argparse
from ddranking.metrics import SoftLabelEvaluator, HardLabelEvaluator
from ddranking.config import Config
from decode import load_synthetic_data

# 关闭警告
warnings.filterwarnings("ignore")

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Choose dataset and data path for synthetic data")
    parser.add_argument('--data_path', type=str, default='./distilled_data_ddranking/data_init.pt', help='Path to synthetic data')
    parser.add_argument('--config', type=str, default='../configs/Demo_Hard_Label.yaml', help='Path to config file')
    parser.add_argument('--softlabel', dest='softlabel', action='store_true', help='Use the softlabel to evaluate the dataset')
    return parser.parse_args()

# 读取命令行参数
args = parse_args()

# 配置文件读取
config = Config.from_file(args.config)

# 加载数据
data_dec, target_dec = load_synthetic_data(args.data_path, config)

# 根据 --softlabel 参数判断使用哪种评估方式
if args.softlabel:
    soft_label_evaluator = SoftLabelEvaluator(config)
    soft_label_evaluator.compute_metrics(image_tensor=data_dec, soft_labels=target_dec, syn_lr=config.get('syn_lr'))
    print("Using soft label evaluation.")
else:
    hard_label_evaluator = HardLabelEvaluator(config)
    hard_label_evaluator.compute_metrics(image_tensor=data_dec, hard_labels=target_dec, syn_lr=config.get('syn_lr'))
    print("Using hard label evaluation.")