import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.onnx as onnx
import os
import numpy as np
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
class broadcast_add(nn.Module):
    def __init__(self):
        super(broadcast_add, self).__init__()
        self.conv1 = conv_bn(3,128,1)
        self.conv_tr = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.poo1 = nn.AvgPool2d(kernel_size=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv_tr(x1)
        x1 = self.relu(self.bn1(x1))
        # x2 = self.poo1(x1)
        out = x1
        return out

def export(dir):
    dummy_input = Variable(torch.randn(1, 3, 100, 100))
    model = broadcast_add()
    model.eval()
    torch.save(model.state_dict(),os.path.join(dir,"broadcast_add.pth"))
    onnx.export(model, dummy_input,os.path.join(dir,"broadcast_add.onnx"), verbose=True)

def get_model_and_input(model_save_dir):
    model = broadcast_add()
    model.cpu()
    model_path = os.path.join(model_save_dir,'broadcast_add.pth')
    model.load_state_dict(torch.load(model_path))
    model.cpu()
    model.eval()
    batch_size = 1
    channels = 3
    height = 100
    width = 100
    images = Variable(torch.ones(batch_size, channels, height, width))
    return images,model