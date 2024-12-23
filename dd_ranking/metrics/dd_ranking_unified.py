import os
import time
import torch
import random
import numpy as np
import torch.nn.functional as F
from typing import List
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchvision import transforms, datasets
from dd_ranking.utils import build_model, get_pretrained_model_path
from dd_ranking.utils import TensorDataset, get_random_images, get_dataset
from dd_ranking.utils import set_seed, get_optimizer, get_lr_scheduler
from dd_ranking.utils import train_one_epoch, validate
from dd_ranking.loss import SoftCrossEntropyLoss, KLDivergenceLoss
from dd_ranking.aug import DSA_Augmentation, Mixup_Augmentation, Cutmix_Augmentation, ZCA_Whitening_Augmentation


class Unified_Evaluator:

    def __init__(self, 
        dataset: str, 
        real_data_path: str, 
        ipc: int,
        model_name: str,
        use_soft_label: bool,
        optimizer: str='sgd',
        lr_scheduler: str='step',
        data_aug_func: str=None,
        aug_params: dict=None,
        soft_label_mode: str=None,
        soft_label_criterion: str=None,
        num_eval: int=5,
        im_size: tuple=(32, 32), 
        num_epochs: int=300,
        batch_size: int=256,
        weight_decay: float=0.0005,
        momentum: float=0.9,
        use_zca: bool=False,
        temperature: float=1.0,
        save_path: str=None,
        device: str="cuda"
    ):

        channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv = get_dataset(dataset, real_data_path, im_size, use_zca)
        self.num_classes = num_classes
        self.im_size = im_size
        self.test_loader = DataLoader(dst_test, batch_size=batch_size, num_workers=4, shuffle=False)

        self.ipc = ipc
        self.model_name = model_name
        self.use_soft_label = use_soft_label
        if use_soft_label:
            assert soft_label_mode is not None, "soft_label_mode must be provided if use_soft_label is True"
            assert soft_label_criterion is not None, "soft_label_criterion must be provided if use_soft_label is True"
            self.soft_label_mode = soft_label_mode
            self.soft_label_criterion = soft_label_criterion
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.temperature = temperature

        self.num_eval = num_eval
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

        if not save_path:
            save_path = f"./results/{dataset}/{model_name}/ipc{ipc}/eval_scores.csv"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        self.save_path = save_path

        pretrained_model_path = get_pretrained_model_path(model_name, dataset, ipc)
        self.teacher_model = build_model(
            model_name=model_name, 
            num_classes=num_classes, 
            im_size=self.im_size, 
            pretrained=True, 
            device=self.device, 
            model_path=pretrained_model_path
        )
        self.teacher_model.eval()

        if data_aug_func is None:
            self.aug_func = None
        elif data_aug_func == 'DSA':
            self.aug_func = DSA_Augmentation(aug_params)
        elif data_aug_func == 'ZCA':
            self.aug_func = ZCA_Whitening_Augmentation(aug_params)
        elif data_aug_func == 'Mixup':
            self.aug_func = Mixup_Augmentation(aug_params)  
        elif data_aug_func == 'Cutmix':
            self.aug_func = Cutmix_Augmentation(aug_params)
        else:
            raise ValueError(f"Invalid data augmentation function: {data_aug_func}")

    def generate_soft_labels(self, images):
        batches = torch.split(images, self.batch_size)
        soft_labels = []
        with torch.no_grad():
            for image_batch in batches:
                image_batch = image_batch.to(self.device)
                soft_labels.append(self.teacher_model(image_batch).detach().cpu())
        soft_labels = torch.cat(soft_labels, dim=0)
        return soft_labels

    def hyper_param_search(self, loader):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        best_acc = 0
        best_lr = None
        for lr in lr_list:
            print(f"Searching lr: {lr}")
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False, 
                device=self.device
            )
            acc = self.compute_metrics_helper(
                model=model, 
                loader=loader, 
                lr=lr
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
        return best_acc, best_lr
    
    def get_loss_fn(self):
        if self.use_soft_label:
            if self.soft_label_criterion == 'kl':
                return KLDivergenceLoss(temperature=self.temperature)
            elif self.soft_label_criterion == 'sce':
                return SoftCrossEntropyLoss()
            else:
                raise ValueError(f"Invalid soft label criterion: {self.soft_label_criterion}")
        else:
            return nn.CrossEntropyLoss()
    
    def compute_metrics_helper(self, model, loader, lr):
        loss_fn = self.get_loss_fn()
        
        optimizer = get_optimizer(model, self.optimizer, lr, self.weight_decay, self.momentum)
        scheduler = get_lr_scheduler(optimizer, self.lr_scheduler, self.num_epochs)
        
        best_acc = 0
        for epoch in range(self.num_epochs):
            train_one_epoch(
                model=model, 
                loader=loader, 
                optimizer=optimizer, 
                loss_fn=loss_fn,
                soft_label_mode=self.soft_label_mode,
                aug_func=self.aug_func,
                lr_scheduler=scheduler,
                tea_model=self.teacher_model,
                logging=True,
                device=self.device
            )
            acc = validate(
                model=model, 
                loader=loader,
                aug_func=self.aug_func,
                logging=True,
                device=self.device
            )
            if acc > best_acc:
                best_acc = acc
        return best_acc
        
    def compute_metrics(self, images, labels, syn_lr=None):
        syn_dataset = TensorDataset(images, labels)
        syn_loader = DataLoader(syn_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        accs = []
        lrs = []
        for i in range(self.num_eval):
            set_seed()
            print(f"########################### {i+1}th Evaluation ###########################")
            if syn_lr:
                model = build_model(
                    model_name=self.model_name, 
                    num_classes=self.num_classes,
                    im_size=self.im_size, 
                    pretrained=False, 
                    device=self.device
                )
                syn_data_acc = self.compute_metrics_helper(
                    model=model, 
                    loader=syn_loader, 
                    lr=syn_lr
                )
                del model
            else:
                syn_data_acc, best_lr = self.hyper_param_search(syn_loader)
            accs.append(syn_data_acc)
            if syn_lr:
                lrs.append(syn_lr)
            else:
                lrs.append(best_lr)
        
        results_to_save = {
            "accs": accs,
            "lrs": lrs
        }
        save_results(results_to_save, self.save_path)

        accs_mean = np.mean(accs)
        accs_std = np.std(accs)
        return {
            "acc_mean": accs_mean,
            "acc_std": accs_std
        }
