import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import resnext_50_32x4d_feature
from resnext_50_32x4d_feature import resnext_50_32x4d_feature
from collections import OrderedDict

__all__ = ['ResNeXt50_32x4d', 'resnext101_32x4d',
           'ResNeXt101_64x4d', 'resnext101_64x4d']

pretrained_settings = {
    'resnext50_32x4d': {
        'imagenet': {
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }
}

class ResNeXt50_32x4d(nn.Module):

    def __init__(self):
        super(ResNeXt50_32x4d, self).__init__()
        self.features = resnext_50_32x4d_feature

    def forward(self, input):
        x = self.features(input)
        return x

def resnext_50_32x4d_1(pretrained=False):
    model = ResNeXt50_32x4d()
    if pretrained:
        # model_state_dict = model.state_dict()
        # for k, v in model_state_dict.items():
        #     print ("model keys: ", k)
        # settings = pretrained_settings['resnext50_32x4d']['imagenet']
        checkpoint = torch.load('/home/eric/Disk100G/githubProject/AlignedReID-Re-Production-Pytorch/model/resnext_50_32x4d.pth')
        # checkpoint = torch.load('/home/eric/Disk100G/githubProject/AlignedReID-Re-Production-Pytorch/model/trainSet_conbined/ResneXt-50/GL-0.7_LL-0.3_NNF_TWGD_EP-200_LDOHS-true_CP-0.3_CR-0.7_staircase_warm_up/resume_ckpt.pth')
        # model.load_state_dict(checkpoint)
        new_state_dict = OrderedDict()
        state_dict = checkpoint
        for k, v in state_dict.items():
            # print ("ckpt keys: ", k)
            # name = k[9:]  # remove `module. or features.`
            name = "features." + k
            if k.startswith('10.'):
                continue
            else:
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        #model.input_space = settings['input_space']
        #model.input_size = settings['input_size']
        #model.input_range = settings['input_range']
        #model.mean = settings['mean']
        #model.std = settings['std']
    return model

