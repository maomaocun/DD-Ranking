
import models.resnet as RN
import models.resnet_ap as RNAP
import models.densenet_cifar as DN
import models.convnet as CN
from efficientnet_pytorch import EfficientNet



def parse_model_name(model_name):
    try:
        depth = int(model_name.split("-")[1])
        if "BN" in model_name and len(model_name.split("-")) > 2 and model_name.split("-")[2] == "BN":
            batchnorm = True
        else:
            batchnorm = False
    except:
        raise ValueError("Model name must be in the format of <model_name>-<depth>-[<batchnorm>]")
    return model_name.split("-")[0].lower(),depth, batchnorm

def define_model(dataset, net_type, nclass, size, norm_type='instance'):
    model_name, depth, batchnorm = parse_model_name(net_type)
    if model_name == 'resnet':
        model = RN.ResNet(dataset.lower(), depth, nclass, norm_type=norm_type, size=size, nch=3)
    elif model_name == 'resnet_ap':
        model = RNAP.ResNetAP(dataset.lower(), depth, nclass, width=1.0, norm_type=norm_type, size=size, nch=3)
    elif model_name == 'efficient':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    elif model_name == 'densenet':
        model = DN.densenet_cifar(nclass)
    elif model_name == 'convnet':
        print(f"dataset: {dataset.lower()}, model_name: {model_name}, nclass: {nclass}, im_size: {(size, size)}")
        model = CN.ConvNet(nclass, net_norm=norm_type, net_depth=depth, net_width= int(128 * 1.0), channel=3, im_size=(size, size))
    else:
        raise Exception('unknown network architecture: {}'.format(model_name))

    return model
