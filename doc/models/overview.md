# Models

DD-Ranking provides a set of commonly used model architectures in existing dataset distillation methods. Users can flexibly use these models for main evaluation or cross-architecture evaluation. We will keep updating this section with more models.

- [ConvNet](convnet.md)
- [ResNet](resnet.md)
- [VGG](vgg.md)
- [LeNet](lenet.md)
- [AlexNet](alexnet.md)
- [MLP](mlp.md)

## Naming Convention

We use the following naming convention for models in DD-Ranking:

- `model name - model depth - norm type`

Model name and depth are required. When norm type is not specified, we use default normalization for the model. For example, `ResNet-18-BN` means ResNet18 with batch normalization. `ConvNet-4` means ConvNet with depth 4 and default instance normalization.

