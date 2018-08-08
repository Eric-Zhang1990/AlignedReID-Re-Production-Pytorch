
import torch
import torch.nn as nn
import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable
from collections import OrderedDict

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt, self).__init__()
        self.layer0 = nn.Conv2d(3,64,(7, 7),(2, 2),(3, 3),1,1,bias=False)
        self.layer1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((3, 3),(2, 2),(1, 1))

        self.layer4 = nn.Sequential( # Sequential,
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(64,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(128),
							nn.ReLU(),
							nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(128),
							nn.ReLU(),
						),
						nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(256),
					),
					nn.Sequential( # Sequential,
						nn.Conv2d(64,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(256),
					),
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(256,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(128),
							nn.ReLU(),
							nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(128),
							nn.ReLU(),
						),
						nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(256),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(256,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(128),
							nn.ReLU(),
							nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(128),
							nn.ReLU(),
						),
						nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(256),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
		)

        self.layer5 = nn.Sequential( # Sequential,
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(256),
							nn.ReLU(),
							nn.Conv2d(256,256,(3, 3),(2, 2),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(256),
							nn.ReLU(),
						),
						nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
					),
					nn.Sequential( # Sequential,
						nn.Conv2d(256,512,(1, 1),(2, 2),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
					),
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(256),
							nn.ReLU(),
							nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(256),
							nn.ReLU(),
						),
						nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(256),
							nn.ReLU(),
							nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(256),
							nn.ReLU(),
						),
						nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(256),
							nn.ReLU(),
							nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(256),
							nn.ReLU(),
						),
						nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
		)

        self.layer6 = nn.Sequential( # Sequential,
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
							nn.Conv2d(512,512,(3, 3),(2, 2),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
						),
						nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					nn.Sequential( # Sequential,
						nn.Conv2d(512,1024,(1, 1),(2, 2),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
							nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
						),
						nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
							nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
						),
						nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
							nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
						),
						nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
							nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
						),
						nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
							nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
						),
						nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
		)

        self.layer7 = nn.Sequential( # Sequential,
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							# nn.Conv2d(1024,1024,(3, 3),(2, 2),(1, 1),1,32,bias=False),
							nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(2048),
					),
					nn.Sequential( # Sequential,
						# nn.Conv2d(1024,2048,(1, 1),(2, 2),(0, 0),1,1,bias=False),
						nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
						nn.BatchNorm2d(2048),
					),
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(2048,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(2048),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(2048,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,32,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(2048),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
		)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x

def resnext_50_32x4d(pretrained=False):
    model = ResNeXt()
    print ("use resneXt-50 model: ", model)
    if pretrained:
        checkpoint = torch.load('/home/eric/Disk100G/githubProject/AlignedReID-Re-Production-Pytorch/model/resnext_50_32x4d.pth')
        # # print ("checkpoint: ", checkpoint)
        # ## create new OrderedDict that does not contain `module.`
        # model_state_dict = model.state_dict()
        # for k, v in model_state_dict.items():
        #     print ("model keys: ", k)
        new_state_dict = OrderedDict()
        state_dict = checkpoint
        for k, v in state_dict.items():
            # print ("ckpt keys: ", k)
            # name = k[9:]  # remove `module. or features.`
            name = "layer" + k
            if k.startswith('10.'):
                continue
            else:
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        # model.load_state_dict((state_dict))
    return model