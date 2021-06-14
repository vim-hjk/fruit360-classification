import timm
import torch.nn as nn
# from pprint import pprint


# pprint(timm.list_models(pretrained=True))

class PretrainedModel(nn.Module):
    def __init__(self, model_arc='resnet18d', num_classes=131):
        super().__init__()
        self.net = timm.create_model(model_arc, pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.net(x)

        return x
