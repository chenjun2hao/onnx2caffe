# Convert pytorch to Caffe by ONNX
This tool converts [pytorch](https://github.com/pytorch/pytorch) model to [Caffe](https://github.com/BVLC/caffe) model by [ONNX](https://github.com/onnx/onnx)  
only use for inference


### Update
20230315: change pool roundmethod to floor, which pytorch used. add centernet example, pretrained from [mmdetection](https://github.com/open-mmlab/mmdetection).


### Dependencies
* caffe (with python support)
* pytorch 0.4 (optional if you only want to convert onnx)
* onnx  


### install
```
git clone https://github.com/chenjun2hao/onnx2caffe.git
python setup.py install

or install on develop mode
python setup.py develop
```


### How to use
just run in terminal
```
onnx2caffe ./model/MobileNetV2.onnx ./model/MobileNetV2.prototxt ./model/MobileNetV2.caffemodel
```


### Current support operation
* Conv
* ConvTranspose
* BatchNormalization
* MaxPool
* AveragePool
* Relu
* Sigmoid
* Dropout
* Gemm (InnerProduct only)
* Add
* Mul
* Reshape
* Upsample
* Concat
* Flatten

### TODO List
 - [ ] support all onnx operations (which is impossible)
 - [ ] merge batchnormization to convolution
 - [ ] merge scale to convolution
