import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.onnx as onnx
import os
import numpy as np
import sys
sys.path.insert(0, "/home/chenjun/6_caffe/onnx2caffe")

from onnx2caffe.convertCaffe import convertToCaffe, getGraph

from res import ResNet, new_model


def conv_bn(inp, oup, ks, pad, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, ks, stride, pad, stride, bias=True),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.conv1 = conv_bn(3, 64, 7, 3, 2)

        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        return x1
    

def check():
    onnx_path = 'output/resnet18.onnx'
    prototxt_path = 'output/resnet18.prototxt'
    caffemodel_path = 'output/resnet18.caffemodel'

    ## pytorch infer
    dummy_input = Variable(torch.randn(1, 3, 448, 672))
    # model = resnet18()
    model = new_model()
    model.eval()
    pt_out = model(dummy_input)
    if isinstance(pt_out, tuple):
        pt_out = pt_out[0]
    pt_out = pt_out.detach().numpy()
    onnx.export(model, dummy_input, onnx_path, verbose=True)

    ## convert caffe
    graph = getGraph(onnx_path)
    caffe_model = convertToCaffe(graph, prototxt_path, caffemodel_path)

    ## caffe infer
    input_name = str(graph.inputs[0][0])
    output_name = str(graph.outputs[0][0])

    caffe_model.blobs[input_name].data[...] = dummy_input.numpy()
    net_output = caffe_model.forward()
    caffe_out = net_output[output_name]

    ## check
    minus_result = caffe_out - pt_out
    mse = np.sum(minus_result*minus_result)
    print(f"mse loss is: {mse}")


if __name__ == '__main__':
    check()