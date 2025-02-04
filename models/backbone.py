import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet101_Weights

def get_resnet_backbone(model_name="resnet50", pretrained=True, freeze_until=0):
    """
    Get a ResNet backbone and optionally freeze the first 'freeze_until' child modules.
    Common ResNet structures: 
      children() => [Conv+BN+ReLU+Pool, layer1, layer2, layer3, layer4, avgpool, fc]
    Here, the final fc layer is removed, so only the modules from [0..-1) are kept.
    The 'freeze_until' parameter indicates the number of initial modules to freeze, for example:
      freeze_until=1 => Freeze the 0th module (i.e., Conv+BN+ReLU+Pool)
      freeze_until=2 => Also freeze layer1
    """
    if model_name.lower() == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet50(weights=weights)
        feature_dim = 2048
    elif model_name.lower() == "resnet101":
        weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet101(weights=weights)
        feature_dim = 2048
    else:
        raise NotImplementedError("Only resnet50/resnet101 are supported as examples. You can extend this yourself.")

    # Remove the final fc layer
    modules = list(net.children())[:-1]
    backbone = nn.Sequential(*modules)

    # Optionally freeze the first 'freeze_until' child modules
    if freeze_until > 0:
        child_list = list(backbone.children())
        # Prevent out-of-bounds error
        freeze_until = min(freeze_until, len(child_list))
        for idx, child in enumerate(child_list):
            if idx < freeze_until:
                for param in child.parameters():
                    param.requires_grad = False

    return backbone, feature_dim
