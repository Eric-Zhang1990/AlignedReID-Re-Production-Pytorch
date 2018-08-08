import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import torch

class ResNeXtBottleneck(nn.Module):
  expansion = 4
  """
  RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
  """
  def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None):
    super(ResNeXtBottleneck, self).__init__()

    D = int(math.floor(planes * (base_width/64.0)))
    C = cardinality

    self.conv_reduce = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn_reduce = nn.BatchNorm2d(D*C)

    self.conv_conv = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
    self.bn = nn.BatchNorm2d(D*C)

    self.conv_expand = nn.Conv2d(D*C, planes*4, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn_expand = nn.BatchNorm2d(planes*4)

    self.downsample = downsample

  def forward(self, x):
    residual = x

    bottleneck = self.conv_reduce(x)
    bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

    bottleneck = self.conv_conv(bottleneck)
    bottleneck = F.relu(self.bn(bottleneck), inplace=True)

    bottleneck = self.conv_expand(bottleneck)
    bottleneck = self.bn_expand(bottleneck)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    return F.relu(residual + bottleneck, inplace=True)


class ResNeXt(nn.Module):
  """
  ResNext optimized for the Cifar dataset, as specified in
  https://arxiv.org/pdf/1611.05431.pdf
  """
  def __init__(self, block, depth, cardinality, base_width, layers):
    super(ResNeXt, self).__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    # assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
    # layer_blocks = (depth - 2) // 9

    self.cardinality = cardinality
    self.base_width = base_width
    # self.num_classes = num_classes

    # self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    # self.bn_1 = nn.BatchNorm2d(64)

    self.inplanes = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                        bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.stage_1 = self._make_layer(block, 64 , layers[0])
    self.stage_2 = self._make_layer(block, 128, layers[1], 2)
    self.stage_3 = self._make_layer(block, 256, layers[2], 2)
    self.stage_4 = self._make_layer(block, 512, layers[3], 2)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, self.cardinality, self.base_width))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.stage_4(x)

    return x

def remove_fc(state_dict):
  """Remove the fc layer parameters from state_dict."""
  for key, value in state_dict.items():
    if key.startswith('fc.'):
      del state_dict[key]
  return state_dict

def resnext50_32_4(pretrained=False):
  """Constructs a ResNeXt-50, 32*4d model
  
  Args:
    num_classes (uint): number of classes
  """
  model = ResNeXt(ResNeXtBottleneck, 50, 32, 4, [3, 4, 6, 3])
  print ("resnext model: ", model)
  if pretrained:
    # checkpoint = torch.load('/home/qwe/AlignedReID/AlignedReID-Re-Production-Pytorch/model/resnet50-19c8e357.pth')
    checkpoint = torch.load('/home/eric/Disk100G/githubProject/AlignedReID-Re-Production-Pytorch/model/resnext_50_32x4d.pth')
    print ("checkpoint: ", checkpoint)
    model.load_state_dict(remove_fc(checkpoint))
  return model

def resnext29_16_64(pretrained=False):
  """Constructs a ResNeXt-29, 16*64d model
  
  Args:
    num_classes (uint): number of classes
  """
  model = ResNeXt(ResNeXtBottleneck, 29, 16, 64, [3, 4, 6, 3])
  return model

def resnext29_8_64(pretrained=False):
  """Constructs a ResNeXt-29, 8*64d model
  
  Args:
    num_classes (uint): number of classes
  """
  model = ResNeXt(ResNeXtBottleneck, 29, 8, 64, [3, 4, 6, 3])
  return model