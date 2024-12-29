# Augmentations

DD-Ranking supports commonly used data augmentations in existing methods. A list of augmentations is provided below:

- [Torchvision transforms](https://pytorch.org/vision/stable/transforms.html)
- [DSA](datm.md)
- [Mixup](mixup.md)
- [Cutmix](cutmix.md)

In DD-Ranking, data augmentations are specified when initializing an evaluator. 
The following arguments are related to data augmentations:

- **data_aug_func**(<span style="color:#FF6B00;">str</span>): The name of the data augmentation function used during training. Currently, we support `dsa`, `mixup`, `cutmix`.
- **aug_params**(<span style="color:#FF6B00;">dict</span>): The parameters for the data augmentation function.
- **custom_train_trans**(<span style="color:#FF6B00;">torchvision.transforms.Compose</span>): The custom train transform used to load the synthetic data when it's in '.jpg' or '.png' format.
- **custom_val_trans**(<span style="color:#FF6B00;">torchvision.transforms.Compose</span>): The custom val transform used to load the test dataset.
- **use_zca**(<span style="color:#FF6B00;">bool</span>): Whether to use ZCA whitening for the data augmentation. This is only applicable to methods that use ZCA whitening during distillation.

```python
# When initializing an evaluator, the data augmentation function is specified.
>>> evaluator = SoftLabelEvaluator(
    ...
    data_aug_func=..., # Specify the data augmentation function
    aug_params=..., # Specify the parameters for the data augmentation function
    custom_train_trans=..., # Specify the custom train transform
    custom_val_trans=..., # Specify the custom val transform
    use_zca=..., # Specify whether to use ZCA whitening
    ...
)
```