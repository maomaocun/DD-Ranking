import torch
import numpy as np
import kornia


class Mixup_Augmentation:
    def __init__(self, params: dict):
        self.mixup_p = params["mixup_p"]

    def mixup(self, images):
        rand_index = torch.randperm(images.size()[0]).to(images.device)
        lam = np.random.beta(self.mixup_p, self.mixup_p)

        mixed_images = lam * images + (1 - lam) * images[rand_index]
        return mixed_images
        
    def __call__(self, images):
        return self.mixup(images)
