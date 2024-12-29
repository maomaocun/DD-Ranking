## Torchvision Transfoms

We notice that some methods use jpg or png format images instead of image tensors during evaluation, and apply an additional torchvision-based tranformation to preprocess these images. Also, they may apply different augmentations to the test dataset. Thus, we support the torchvision-based transformations in DD-Ranking for both synthetic and real data.

We require the torchvision-based transformations to be a `torchvision.transforms.Compose` object. If you have customized transformations, please make sure they have a `__call__` method. For the list of torchvision transformations, please refer to [torchvision-transforms](https://pytorch.org/vision/stable/transforms.html).

### Example

```python
# Define a custom transformation
class MyTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        return x

custom_train_trans = torchvision.transforms.Compose([
    MyTransform(),
    torchvision.transforms.RandomResizedCrop(32),
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize(mean, std)
])
custom_val_trans = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])

evaluator = DD_Ranking(
    ...
    custom_train_trans=custom_train_trans,
    custom_val_trans=custom_val_trans,
    ...
)
```
