import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.autograd import Variable

# from aligned_reid.utils.utils import load_state_dict



class ResBottleBlock(nn.Module):

    def __init__(self, in_planes, bottleneck_width=4, stride=1, expansion=1):
        super(ResBottleBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, bottleneck_width, 1, stride=1, bias=False)
        self.bn0 = nn.BatchNorm2d(bottleneck_width)
        self.conv1 = nn.Conv2d(bottleneck_width, bottleneck_width, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_width)
        self.conv2 = nn.Conv2d(bottleneck_width, expansion * in_planes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(expansion * in_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or expansion != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * expansion, 1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock_C(nn.Module):
    """
    increasing cardinality is a more effective way of
    gaining accuracy than going deeper or wider
    """

    def __init__(self, in_planes, bottleneck_width=4, cardinality=32, stride=1, expansion=2):
        super(BasicBlock_C, self).__init__()
        inner_width = cardinality * bottleneck_width
        self.expansion = expansion
        self.basic = nn.Sequential(OrderedDict(
            [
                ('conv1_0', nn.Conv2d(in_planes, inner_width, 1, stride=1, bias=False)),
                ('bn1', nn.BatchNorm2d(inner_width)),
                ('act0', nn.ReLU()),
                ('conv3_0',
                 nn.Conv2d(inner_width, inner_width, 3, stride=stride, padding=1, groups=cardinality, bias=False)),
                ('bn2', nn.BatchNorm2d(inner_width)),
                ('act1', nn.ReLU()),
                ('conv1_1', nn.Conv2d(inner_width, inner_width * self.expansion, 1, stride=1, bias=False)),
                ('bn3', nn.BatchNorm2d(inner_width * self.expansion))
            ]
        ))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != inner_width * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, inner_width * self.expansion, 1, stride=stride, bias=False)
            )
        self.bn0 = nn.BatchNorm2d(self.expansion * inner_width)

    def forward(self, x):
        out = self.basic(x)
        out += self.shortcut(x)
        out = F.relu(self.bn0(out))
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, expansion=2, num_classes=1000):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.expansion = expansion

        self.conv0 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3)
        self.bn0 = nn.BatchNorm2d(self.in_planes)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.layer4 = self._make_layer(num_blocks[3], 2)
        # self.linear = nn.Linear(self.cardinality * self.bottleneck_width, num_classes)

    def forward(self, x):
        # out = F.relu(self.bn0(self.conv0(x)))
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu0(out)
        out = self.pool0(out)
        out = self.layer1(out)
        # print("out1 size is: {}".format(out.size()))
        out = self.layer2(out)
        # print("out2 size is: {}".format(out.size()))
        out = self.layer3(out)
        # print("out3 size is: {}".format(out.size()))
        out = self.layer4(out)
        # print("out4 size is: {}".format(out.size()))
        # out = F.avg_pool2d(out, 7)
        # print("out5 size is: {}".format(out.size()))
        # print("out52 size is: {}".format(out.size(0)))
        # out = out.view(out.size(0), -1)
        # print("out6 size is: {}".format(out.size()))
        # out = self.linear(out)
        return out

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_C(self.in_planes, self.bottleneck_width, self.cardinality, stride, self.expansion))
            self.in_planes = self.expansion * self.bottleneck_width * self.cardinality
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

def remove_fc(state_dict):
  """Remove the fc layer parameters from state_dict."""
  for key, value in state_dict.items():
    # if key.startswith('fc.'):
    if key.startswith('linear.'):
      del state_dict[key]
  return state_dict

def resnext26_2x64d(pretrained=False):
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=2, bottleneck_width=64)


def resnext26_4x32d(pretrained=False):
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=4, bottleneck_width=32)


def resnext26_8x16d(pretrained=False):
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=8, bottleneck_width=16)


def resnext26_16x8d(pretrained=False):
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=16, bottleneck_width=8)


def resnext26_32x4d(pretrained=False):
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=32, bottleneck_width=4)


def resnext26_64x2d(pretrained=False):
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=32, bottleneck_width=4)


def resnext50_2x64d(pretrained=False):
    return ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=2, bottleneck_width=64)


def resnext50_32x4d(pretrained=False):
    model = ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4)
    # print ("resnext model: ", model)
    if pretrained:
      checkpoint = torch.load('/home/eric/Disk100G/githubProject/AlignedReID-Re-Production-Pytorch/model/ResNext50_checkpoint_best.pth.tar')
      # print ("checkpoint: ", checkpoint)
      ## create new OrderedDict that does not contain `module.`
      new_state_dict = OrderedDict()
      state_dict = checkpoint['state_dict']
      for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.startswith('linear.'):
          continue
        else:
          new_state_dict[name] = v
      ## load params
      model.load_state_dict(new_state_dict)

      # model.load_state_dict((checkpoint['state_dict']))
    return model

    # return ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4)