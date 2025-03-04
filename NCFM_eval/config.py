# CIFAR-10 参数
cifar10_params = {
    "size": (32, 32),
    "nclass": 10,
    "syn_lr": 0.01,
}

# CIFAR-100 参数
cifar100_params = {
    "size": (32, 32),
    "nclass": 100,
    "syn_lr": 0.01,
}

def get_params(dataset="cifar10"):
    # 根据选择的数据集加载参数
    if dataset == 'cifar10':
        params = cifar10_params
    elif dataset == 'cifar100':
        params = cifar100_params
    else:
        # 如果 dataset 不是 'cifar10' 或 'cifar100'，抛出错误
        raise ValueError(f"Invalid dataset name '{dataset}'. Please choose either 'cifar10' or 'cifar100'.")
    
    return params