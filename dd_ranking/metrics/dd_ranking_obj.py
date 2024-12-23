import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from dd_ranking.utils import build_model, get_pretrained_model_path
from dd_ranking.utils import TensorDataset, get_random_images, get_dataset, save_results
from dd_ranking.utils import set_seed, train_one_epoch, train_one_epoch_dc, validate, validate_dc, get_optimizer, get_lr_scheduler
from dd_ranking.loss import SoftCrossEntropyLoss, KLDivergenceLoss
from dd_ranking.aug import DSA_Augmentation, ZCA_Whitening_Augmentation, Mixup_Augmentation, Cutmix_Augmentation
from dd_ranking.config import Config

class Soft_Label_Objective_Metrics:

    def __init__(self, config: Config=None, dataset: str='CIFAR10', real_data_path: str='./dataset/', ipc: int=10, model_name: str='ConvNet-3', 
                 soft_label_criterion: str='kl', data_aug_func: str='cutmix', aug_params: dict={'cutmix_p': 1.0}, soft_label_mode: str='S',
                 optimizer: str='sgd', lr_scheduler: str='step', temperature: float=1.0, weight_decay: float=0.0005, 
                 momentum: float=0.9, num_eval: int=5, im_size: tuple=(32, 32), num_epochs: int=300, use_zca: bool=False,
                 batch_size: int=256, default_lr: float=0.01, save_path: str=None, device: str="cuda"):

        if config is not None:
            self.config = config
            dataset = self.config.get('dataset')
            real_data_path = self.config.get('real_data_path')
            ipc = self.config.get('ipc')
            model_name = self.config.get('model_name')
            soft_label_criterion = self.config.get('soft_label_criterion')
            data_aug_func = self.config.get('data_aug_func')
            aug_params = self.config.get('aug_params')
            soft_label_mode = self.config.get('soft_label_mode')
            optimizer = self.config.get('optimizer')
            lr_scheduler = self.config.get('lr_scheduler')
            temperature = self.config.get('temperature')
            weight_decay = self.config.get('weight_decay')
            momentum = self.config.get('momentum')
            num_eval = self.config.get('num_eval')
            im_size = self.config.get('im_size')
            num_epochs = self.config.get('num_epochs')
            batch_size = self.config.get('batch_size')
            default_lr = self.config.get('default_lr')
            save_path = self.config.get('save_path')
            device = self.config.get('device')

        channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv = get_dataset(dataset, 
                                                                                                   real_data_path, 
                                                                                                   im_size, 
                                                                                                   use_zca)
        self.images_train, self.labels_train, self.class_indices_train = self.load_real_data(dst_train, class_map, num_classes)
        self.test_loader = DataLoader(dst_test, batch_size=batch_size, num_workers=4, shuffle=False)

        self.soft_label_mode = soft_label_mode
        self.soft_label_criterion = soft_label_criterion
        self.temperature = temperature

        # data info
        self.im_size = im_size
        self.num_classes = num_classes
        self.ipc = ipc

        # training info
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay

        self.momentum = momentum
        self.num_eval = num_eval
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.default_lr = default_lr
        self.test_interval = 10
        self.device = device

        if data_aug_func == 'dsa':
            self.aug_func = DSA_Augmentation(aug_params)
            self.num_epochs = 1000
        elif data_aug_func == 'zca':
            self.aug_func = ZCA_Whitening_Augmentation(aug_params)
        elif data_aug_func == 'mixup':
            self.aug_func = Mixup_Augmentation(aug_params)
        elif data_aug_func == 'cutmix':
            self.aug_func = Cutmix_Augmentation(aug_params)
        else:
            self.aug_func = None

        if not save_path:
            save_path = f"./results/{dataset}/{model_name}/ipc{ipc}/obj_scores.csv"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        self.save_path = save_path

        # teacher model
        pretrained_model_path = get_pretrained_model_path(model_name, dataset, ipc)
        self.teacher_model = build_model(model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=True, device=self.device, model_path=pretrained_model_path)
        self.teacher_model.eval()

    def load_real_data(self, dataset, class_map, num_classes):
        images_all = []
        labels_all = []
        class_indices = [[] for c in range(num_classes)]
        for i, (image, label) in enumerate(dataset):
            images_all.append(torch.unsqueeze(image, 0))
            labels_all.append(class_map[label])
        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all)
        for i, label in enumerate(labels_all):
            class_indices[label].append(i)
        
        return images_all, labels_all, class_indices
    
    def hyper_param_search_for_hard_label(self, images, hard_labels):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            print(f"Searching lr:{lr} for hard label...")
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False, 
                device=self.device
            )
            acc = self.compute_hard_label_metrics(
                model=model, 
                images=images, 
                lr=lr, 
                hard_labels=hard_labels
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr

    def hyper_param_search_for_soft_label(self, images, soft_labels):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            print(f"Searching lr:{lr} for soft label...")
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False, 
                device=self.device
            )
            acc = self.compute_soft_label_metrics(
                model=model, 
                images=images, 
                lr=lr, 
                soft_labels=soft_labels
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr
    
    def compute_hard_label_metrics(self, model, images, lr, hard_labels):
        
        hard_label_dataset = TensorDataset(images, hard_labels)
        train_loader = DataLoader(hard_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = get_optimizer(self.optimizer, model, lr, self.weight_decay, self.momentum)
        lr_scheduler = get_lr_scheduler(self.lr_scheduler, optimizer, self.num_epochs)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model, 
                loader=train_loader, 
                loss_fn=loss_fn, 
                optimizer=optimizer,
                aug_func=self.aug_func,
                lr_scheduler=lr_scheduler, 
                tea_model=self.teacher_model, 
                device=self.device
            )
            if epoch % self.test_interval == 0:
                metric = validate(
                    model=model, 
                    loader=self.test_loader,
                    aug_func=self.aug_func,
                    device=self.device
                )
                if metric['top1'] > best_acc1:
                    best_acc1 = metric['top1']

        return best_acc1
        
    def compute_soft_label_metrics(self, model, images, lr, soft_labels):
        if soft_labels is None:
            # replace soft labels with hard labels to create dataset. During training, soft labels are generated by teacher model.
            labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), dtype=torch.long, requires_grad=False).view(-1)
        else:
            labels = soft_labels
        soft_label_dataset = TensorDataset(images, labels)
        train_loader = DataLoader(soft_label_dataset, batch_size=self.batch_size, shuffle=True)

        if self.soft_label_criterion == 'sce':
            loss_fn = SoftCrossEntropyLoss()
        elif self.soft_label_criterion == 'kl':
            loss_fn = KLDivergenceLoss()
        else:
            raise NotImplementedError(f"Soft label criterion {self.soft_label_criterion} not implemented")
        
        optimizer = get_optimizer(self.optimizer, model, lr, self.weight_decay, self.momentum)
        lr_scheduler = get_lr_scheduler(self.lr_scheduler, optimizer, self.num_epochs)
        
        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model,
                loader=train_loader, 
                loss_fn=loss_fn, 
                optimizer=optimizer,
                aug_func=self.aug_func,
                soft_label_mode=self.soft_label_mode,
                lr_scheduler=lr_scheduler, 
                tea_model=self.teacher_model, 
                device=self.device
            )
            if epoch % self.test_interval == 0:
                metric = validate(
                    model=model, 
                    loader=self.test_loader,
                    aug_func=self.aug_func,
                    device=self.device
                )
                if metric['top1'] > best_acc1:
                    best_acc1 = metric['top1']
        
        return best_acc1

    def generate_soft_labels(self, images):
        batches = torch.split(images, self.batch_size)
        soft_labels = []
        with torch.no_grad():
            for image_batch in batches:
                image_batch = image_batch.to(self.device)
                soft_labels.append(self.teacher_model(image_batch).detach().cpu())
        soft_labels = torch.cat(soft_labels, dim=0)
        return soft_labels
    
    def compute_metrics(self, syn_images, soft_labels=None, syn_lr=None):
        if self.soft_label_mode == 'S' and soft_labels is None:
            raise ValueError("Soft label mode 'S' requires soft labels")

        hard_labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), 
                                   dtype=torch.long, requires_grad=False).view(-1)
        obj_metrics = []
        hard_recs = []
        soft_imps = []
        for i in range(self.num_eval):
            set_seed()
            print(f"########################### {i+1}th Evaluation ###########################")

            print("Caculating syn data hard label metrics...")
            syn_data_hard_label_acc, best_lr = self.hyper_param_search_for_hard_label(
                images=syn_images, 
                hard_labels=hard_labels
            )
            print(f"Syn data hard label acc: {syn_data_hard_label_acc:.2f}%")

            print("Caculating full data hard label metrics...")
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False, 
                device=self.device
            )
            full_data_hard_label_acc = self.compute_hard_label_metrics(
                model=model, 
                images=self.images_train, 
                lr=self.default_lr, 
                hard_labels=self.labels_train
            )
            del model
            print(f"Full data hard label acc: {full_data_hard_label_acc:.2f}%")

            print("Caculating syn data soft label metrics...")
            if syn_lr:
                model = build_model(
                    model_name=self.model_name, 
                    num_classes=self.num_classes, 
                    im_size=self.im_size,
                    pretrained=False, 
                    device=self.device
                )
                syn_data_soft_label_acc = self.compute_soft_label_metrics(
                    model=model, 
                    images=syn_images, 
                    lr=syn_lr, 
                    soft_labels=soft_labels
                )
                del model
            else:
                syn_data_soft_label_acc, best_lr = self.hyper_param_search_for_soft_label(
                    images=syn_images, 
                    soft_labels=soft_labels
                )
            print(f"Syn data soft label acc: {syn_data_soft_label_acc:.2f}%")

            print("Caculating random data soft label metrics...")
            random_images, _ = get_random_images(self.images_train, self.labels_train, self.class_indices_train, self.ipc)
            if self.soft_label_mode == 'S':
                random_data_soft_labels = self.generate_soft_labels(random_images)
            else:
                random_data_soft_labels = None
            random_data_soft_label_acc, best_lr = self.hyper_param_search_for_soft_label(
                images=random_images, 
                soft_labels=random_data_soft_labels
            )
            print(f"Random data soft label acc: {random_data_soft_label_acc:.2f}%")

            hard_rec = 1.00 * (full_data_hard_label_acc - syn_data_hard_label_acc)
            soft_imp = 1.00 * (syn_data_soft_label_acc - random_data_soft_label_acc)

            hard_recs.append(hard_rec)
            soft_imps.append(soft_imp)
            obj_metrics.append(soft_imp / hard_rec)
        
        results_to_save = {
            "hard_recs": hard_recs,
            "soft_imps": soft_imps,
            "obj_metrics": obj_metrics
        }
        save_results(results_to_save, self.save_path)

        hard_recs_mean = np.mean(hard_recs)
        hard_recs_std = np.std(hard_recs)
        soft_imps_mean = np.mean(soft_imps)
        soft_imps_std = np.std(soft_imps)
        obj_metrics_mean = np.mean(obj_metrics)
        obj_metrics_std = np.std(obj_metrics)

        print(f"Hard Recovery Mean: {hard_recs_mean:.2f}%  Std: {hard_recs_std:.2f}")
        print(f"Soft Improvement Mean: {soft_imps_mean:.2f}%  Std: {soft_imps_std:.2f}")
        print(f"Objective Metrics Mean: {obj_metrics_mean:.2f}  Std: {obj_metrics_std:.2f}")
        return {
            "hard_recs_mean": hard_recs_mean,
            "hard_recs_std": hard_recs_std,
            "soft_imps_mean": soft_imps_mean,
            "soft_imps_std": soft_imps_std,
            "obj_metrics_mean": obj_metrics_mean,
            "obj_metrics_std": obj_metrics_std
        }


class Hard_Label_Objective_Metrics:

    def __init__(self, config: Config=None, dataset: str='CIFAR10', real_data_path: str='./dataset/', ipc: int=10, 
                 model_name: str='ConvNet-3', data_aug_func: str='cutmix', aug_params: dict={'cutmix_p': 1.0},
                 optimizer: str='SGD', lr_scheduler: str='StepLR', weight_decay: float=0.0005, momentum: float=0.9, 
                 use_zca: bool=False, num_eval: int=5, im_size: tuple=(32, 32), num_epochs: int=300, batch_size: int=256, 
                 default_lr: float=0.01, save_path: str=None, device: str="cuda"):
        
        if config is not None:
            self.config = config
            dataset = self.config.get('dataset')
            real_data_path = self.config.get('real_data_path')
            ipc = self.config.get('ipc')
            model_name = self.config.get('model_name')
            data_aug_func = self.config.get('data_aug_func')
            aug_params = self.config.get('aug_params')
            optimizer = self.config.get('optimizer')
            lr_scheduler = self.config.get('lr_scheduler')
            weight_decay = self.config.get('weight_decay')
            momentum = self.config.get('momentum')
            num_eval = self.config.get('num_eval')
            im_size = self.config.get('im_size')
            num_epochs = self.config.get('num_epochs')
            batch_size = self.config.get('batch_size')
            default_lr = self.config.get('default_lr')
            save_path = self.config.get('save_path')
            device = self.config.get('device')

        channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv = get_dataset(dataset, 
                                                                                                   real_data_path, 
                                                                                                   im_size, 
                                                                                                   use_zca)
        self.images_train, self.labels_train, self.class_indices_train = self.load_real_data(dst_train, class_map, num_classes)
        self.test_loader = DataLoader(dst_test, batch_size=batch_size, num_workers=4, shuffle=False)

        # data info
        self.im_size = im_size
        self.num_classes = num_classes
        self.ipc = ipc

        # training info
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.num_eval = num_eval
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.default_lr = default_lr
        self.test_interval = 10
        self.device = device

        if data_aug_func == 'dsa':
            self.aug_func = DSA_Augmentation(aug_params)
            self.num_epochs = 1000
        elif data_aug_func == 'zca':
            self.aug_func = ZCA_Whitening_Augmentation(aug_params)
        elif data_aug_func == 'mixup':
            self.aug_func = Mixup_Augmentation(aug_params)
        elif data_aug_func == 'cutmix':
            self.aug_func = Cutmix_Augmentation(aug_params)
        else:
            self.aug_func = None

        if not save_path:
            save_path = f"./results/{dataset}/{model_name}/ipc{ipc}/obj_scores.csv"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        self.save_path = save_path

    def load_real_data(self, dataset, class_map, num_classes):
        images_all = []
        labels_all = []
        class_indices = [[] for c in range(num_classes)]
        for i, (image, label) in enumerate(dataset):
            images_all.append(torch.unsqueeze(image, 0))
            labels_all.append(class_map[label])
        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all)
        for i, label in enumerate(labels_all):
            class_indices[label].append(i)
        
        return images_all, labels_all, class_indices
    
    def hyper_param_search_for_hard_label(self, images, hard_labels):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            print(f"Searching lr:{lr} for hard label...")
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False, 
                device=self.device
            )
            acc = self.compute_hard_label_metrics(
                model=model, 
                images=images, 
                lr=lr,
                hard_labels=hard_labels
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr

    def compute_hard_label_metrics(self, model, images, lr, hard_labels):
        
        hard_label_dataset = TensorDataset(images, hard_labels)
        train_loader = DataLoader(hard_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = get_optimizer(self.optimizer, model, lr, self.weight_decay, self.momentum)
        lr_scheduler = get_lr_scheduler(self.lr_scheduler, optimizer, self.num_epochs)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model, 
                loader=train_loader, 
                loss_fn=loss_fn, 
                optimizer=optimizer,
                aug_func=self.aug_func,
                lr_scheduler=lr_scheduler, 
                device=self.device
            )
            if epoch % self.test_interval == 0:
                metric = validate(
                    model=model, 
                    loader=self.test_loader,
                    aug_func=self.aug_func,
                    device=self.device
                )
                if metric['top1'] > best_acc1:
                    best_acc1 = metric['top1']

        return best_acc1
    
    def compute_metrics(self, syn_images, hard_labels=None, syn_lr=None):
        if not hard_labels:
            hard_labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), dtype=torch.long, requires_grad=False).view(-1)

        obj_metrics = []
        hard_recs = []
        hard_imps = []
        for i in range(self.num_eval):
            set_seed()
            print(f"########################### {i+1}th Evaluation ###########################")

            print("Caculating syn data hard label metrics...")
            if syn_lr:
                model = build_model(
                    model_name=self.model_name, 
                    num_classes=self.num_classes, 
                    im_size=self.im_size, 
                    pretrained=False, 
                    device=self.device
                )
                syn_data_hard_label_acc = self.compute_hard_label_metrics(
                    model=model, 
                    images=syn_images, 
                    lr=syn_lr, 
                    hard_labels=hard_labels
                )
                del model
            else:
                syn_data_hard_label_acc, best_lr = self.hyper_param_search_for_hard_label(
                    images=syn_images, 
                    hard_labels=hard_labels
                )
            print(f"Syn data hard label acc: {syn_data_hard_label_acc:.2f}%")

            print("Caculating full data hard label metrics...")
            model = build_model(
                model_name=self.model_name,
                num_classes=self.num_classes, 
                im_size=self.im_size,
                pretrained=False,
                device=self.device
            )
            full_data_hard_label_acc = self.compute_hard_label_metrics(
                model=model, 
                images=self.images_train, 
                lr=self.default_lr, 
                hard_labels=self.labels_train
            )
            del model
            print(f"Full data hard label acc: {full_data_hard_label_acc:.2f}%")

            print("Caculating random data hard label metrics...")
            random_images, random_data_hard_labels = get_random_images(self.images_train, self.labels_train, self.class_indices_train, self.ipc)
            random_data_hard_label_acc, best_lr = self.hyper_param_search_for_hard_label(
                images=random_images, 
                hard_labels=random_data_hard_labels
            )
            print(f"Random data hard label acc: {random_data_hard_label_acc:.2f}%")

            hard_rec = 1.00 * (full_data_hard_label_acc - syn_data_hard_label_acc)
            hard_imp = 1.00 * (syn_data_hard_label_acc - random_data_hard_label_acc)

            hard_recs.append(hard_rec)
            hard_imps.append(hard_imp)
            obj_metrics.append(hard_imp / hard_rec)
        
        results_to_save = {
            "hard_recs": hard_recs,
            "hard_imps": hard_imps,
            "obj_metrics": obj_metrics
        }
        save_results(results_to_save, self.save_path)

        hard_recs_mean = np.mean(hard_recs)
        hard_recs_std = np.std(hard_recs)
        hard_imps_mean = np.mean(hard_imps)
        hard_imps_std = np.std(hard_imps)
        obj_metrics_mean = np.mean(obj_metrics)
        obj_metrics_std = np.std(obj_metrics)

        print(f"SCE Hard Recovery Mean: {hard_recs_mean:.2f}%  Std: {hard_recs_std:.2f}")
        print(f"Hard Improvement Mean: {hard_imps_mean:.2f}%  Std: {hard_imps_std:.2f}")
        print(f"Objective Metrics Mean: {obj_metrics_mean:.2f}  Std: {obj_metrics_std:.2f}")
        return {
            "hard_recs_mean": hard_recs_mean,
            "hard_recs_std": hard_recs_std,
            "hard_imps_mean": hard_imps_mean,
            "hard_imps_std": hard_imps_std,
            "obj_metrics_mean": obj_metrics_mean,
            "obj_metrics_std": obj_metrics_std
        }