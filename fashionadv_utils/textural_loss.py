import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from collections import OrderedDict


# vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {'r11': F.relu(self.conv1_1(x))}
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, x):
        b, c, h, w = x.size()
        matrix_f = x.view(b, c, h * w)
        matrix_g = torch.bmm(matrix_f, F.transpose(1, 2))
        matrix_g.div_(h * w)
        return matrix_g


class GramMSELoss(nn.Module):
    def forward(self, x, target):
        out = torch.log(nn.MSELoss()(GramMatrix()(x), target))
        return out


class CrossGramMatrix(nn.Module):
    def forward(self, input1, input2):
        b1, c1, h1, w1 = input1.size()
        F1 = input1.view(b1, c1, h1 * w1)
        ms = torch.nn.Upsample(size=(h1, w1), mode='bilinear')
        input2 = ms(input2)
        b2, c2, h2, w2 = input2.size()
        F2 = input2.view(b2, c2, h2 * w2)

        G = torch.bmm(F1, F2.transpose(1, 2))
        G.div_(h1 * w1)
        return G


class CrossGramMSELoss(nn.Module):
    def forward(self, input1, input2, target):
        out = (nn.MSELoss()(CrossGramMatrix()(input1, input2), target))
        return out

vgg19 = torchvision.models.vgg19(pretrained=True, progress=True)
weight_vgg19 = vgg19.state_dict()

del weight_vgg19['classifier.0.weight']
del weight_vgg19['classifier.0.bias']
del weight_vgg19['classifier.3.weight']
del weight_vgg19['classifier.3.bias']
del weight_vgg19['classifier.6.weight']
del weight_vgg19['classifier.6.bias']

res = {}
res_trans = ["conv1_1.weight", "conv1_1.bias", "conv1_2.weight", "conv1_2.bias", "conv2_1.weight", "conv2_1.bias",
             "conv2_2.weight", "conv2_2.bias", "conv3_1.weight", "conv3_1.bias", "conv3_2.weight", "conv3_2.bias",
             "conv3_3.weight", "conv3_3.bias", "conv3_4.weight", "conv3_4.bias", "conv4_1.weight", "conv4_1.bias",
             "conv4_2.weight", "conv4_2.bias", "conv4_3.weight", "conv4_3.bias", "conv4_4.weight", "conv4_4.bias",
             "conv5_1.weight", "conv5_1.bias", "conv5_2.weight", "conv5_2.bias", "conv5_3.weight", "conv5_3.bias",
             "conv5_4.weight", "conv5_4.bias"]
i = 0
for key in weight_vgg19.keys():
    res[key] = res_trans[i]
    i += 1
vgg_rename = OrderedDict((res[k], weight_vgg19[k]) for k in weight_vgg19.keys())
torch.save(vgg_rename, 'vgg_conv.pth')

# get network
vgg = VGG()
vgg.load_state_dict(torch.load('vgg_conv.pth'))
vgg.cuda()
vgg.eval()

for param in vgg.parameters():
    param.requires_grad = False

# define layers, loss functions, weights and compute optimization targets
style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
content_layers = ['r42']
loss_layers = style_layers + content_layers
loss_fns = [CrossGramMSELoss()] * (len(style_layers) - 1) + [nn.MSELoss()] * len(content_layers)
loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
