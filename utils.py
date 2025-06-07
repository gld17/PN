import os
import torch
from torchvision import transforms
import torchvision
import torchvision.models as models

def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module

def get_module_by_name(model, module_name):
    """
    Get a module specified by its module name

    Parameters
    ----------
    model : pytorch model
        the pytorch model from which to get its module
    module_name : str
        the name of the required module

    Returns
    -------
    module, module
        the parent module of the required module, the required module
    """
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    leaf_module = getattr(model, name_list[-1])
    return model, leaf_module

def get_cnn_model(model_name):
    if model_name == 'vgg11':
        model = models.vgg11(pretrained=True)
        exclude_layers = ['classifier.6']
    elif model_name == 'vgg13':
        model = models.vgg13(pretrained=True)
        exclude_layers = ['classifier.6']
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        exclude_layers = ['classifier.6']
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        exclude_layers = ['classifier.6']
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        exclude_layers = ['fc']
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
        exclude_layers = ['fc']
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        exclude_layers = ['fc']
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        exclude_layers = ['fc']
    elif model_name == 'mobilenetv2':
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        exclude_layers = ['classifier.1']
    elif model_name == 'mobilenetv3s':
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        exclude_layers = ['classifier.3']
    elif model_name == 'mobilenetv3l':
        model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        exclude_layers = ['classifier.3']
    elif model_name == 'efficientnetv2s':
        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        exclude_layers = ['classifier.1']
    elif model_name == 'efficientnetv2m':
        model = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
        exclude_layers = ['classifier.1']
    elif model_name == 'efficientnetv2l':
        model = models.efficientnet_v2_l(weights='IMAGENET1K_V1')
        exclude_layers = ['classifier.1']
        
    return model, exclude_layers

def get_dataset(dataset_name, model_name):
    if dataset_name == 'cifar100':
        if 'vgg' in model_name:
            super_module, leaf_module = get_module_by_name(model, 'classifier.6')    
            classifier = torch.nn.Linear(in_features=leaf_module.in_features, out_features=100, bias=leaf_module.bias is not None).to(device)
            setattr(super_module, '6', classifier)
        elif 'resnet' in model_name:
            super_module, leaf_module = get_module_by_name(model, 'fc')    
            classifier = torch.nn.Linear(in_features=leaf_module.in_features, out_features=100, bias=leaf_module.bias is not None).to(device)
            setattr(super_module, 'fc', classifier)

        # Load test dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = torchvision.datasets.CIFAR100(root='/share/guolidong-local/CIFAR100', train=False, download=True, transform=transform)
    elif dataset_name == 'imagenet':
        testdir = os.path.join('/share/public-local/datasets/ILSVRC2012/', 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if model_name == 'efficientnetv2m':
            test_dataset = torchvision.datasets.ImageFolder(
                    testdir,
                    transforms.Compose([
                        transforms.Resize(480),
                        transforms.CenterCrop(480),
                        transforms.ToTensor(),
                        normalize,
                    ]))
        elif model_name in ['mobilenetv2', 'mobilenetv3l']:
            test_dataset = torchvision.datasets.ImageFolder(
                    testdir,
                    transforms.Compose([
                        transforms.Resize(232),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))
        else:
            test_dataset = torchvision.datasets.ImageFolder(
                    testdir,
                    transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))
    
    return test_dataset