This is a playground for pytorch beginners, which contains predefined models on popular dataset. Currently we support 
- mnist, svhn
- cifar10, cifar100
- stl10
- alexnet
- vgg16, vgg16_bn, vgg19, vgg19_bn
- resent18, resent34, resnet50, resnet101, resnet152
- squeezenet_v0, squeezenet_v1
- inception_v3

Here is an example for MNIST dataset. This will download the dataset and pre-trained model automatically.
```
import torch
from torch.autograd import Variable
from utee import selector
model_raw, ds_fetcher, is_imagenet = selector.select('mnist')
ds_val = ds_fetcher(batch_size=10, train=False, val=True)
for idx, (data, target) in enumerate(ds_val):
    data =  Variable(torch.FloatTensor(data)).cuda()
    output = model_raw(data)
```

Also, if want to train the MLP model on mnist, simply run `python mnist/train.py`


# Install
- pytorch (>=0.1.11) and torchvision from [official website](http://pytorch.org/), for example, cuda8.0 for python3.5
    - `pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl`
    - `pip install torchvision`
- tqdm
    - `pip install tqdm`
- OpenCV
    - `conda install -c menpo opencv3`
- Setting PYTHONPATH
    - `export PYTHONPATH=/path/to/pytorch-playground:$PYTHONPATH`

# ImageNet dataset
We provide precomputed imagenet validation dataset with 224x224x3 size. We first resize the shorter size of image to 256, then we crop 224x224 image in the center. Then we encode the cropped images to jpg string and dump to pickle. 
- `cd script`
- Download the [val224_compressed.pkl](http://ml.cs.tsinghua.edu.cn/~chenxi/dataset/val224_compressed.pkl) 
    - `axel http://ml.cs.tsinghua.edu.cn/~chenxi/dataset/val224_compressed.pkl`
- `python convert.py`


# Quantization
We also provide a simple demo to quantize these models to specified bit-width with several methods, including linear method, minmax method and non-linear method.

`python quantize.py --type cifar10 --quant_method linear --param_bits 8 --fwd_bits 8 --bn_bits 8 --ngpu 1`
   
## Top1 Accuracy
We evaluate the performance of popular dataset and models with linear quantized method. The bit-width of running mean and running variance in BN are 10 bits for all results.


|Model|32-float  |12-bit  |10-bit |8-bit  |6-bit  |
|:----|:--------:|:------:|:-----:|:-----:|:-----:|
|[MNIST](http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth)|98.42%|98.43%|98.44%|98.44%|98.32|
|[SVHN](http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/svhn-f564f3d8.pth)|96.03%|96.03%|96.04%|96.02%|95.46%|
|[CIFAR10](http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar10-d875770b.pth)|93.78%|93.79%|93.80%|93.58%|90.86%|
|[CIFAR100](http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar100-3a55a987.pth)|74.27%|74.21%|74.19%|73.70%|66.32%|
|[STL10](http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/stl10-866321e9.pth)|77.59%|77.65%|77.70%|77.59%|73.40%|
|[AlexNet](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth)|55.70%|55.66%|55.54%|54.17%|18.19%|
|[VGG16](https://download.pytorch.org/models/vgg16-397923af.pth)|70.44%|70.45%|70.44%|69.99%|53.33%|
|[VGG19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)|71.36%|71.35%|71.34%|70.88%|56.00%|
|[ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)|68.63%|68.62%|68.49%|66.80%|19.14%|
|[ResNet34](https://download.pytorch.org/models/resnet34-333f7ec4.pth)|72.50%|72.46%|72.45%|71.47%|32.25%|
|[ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth)|74.98%|74.94%|74.91%|72.54%|2.43%|
|[ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)|76.69%|76.66%|76.22%|65.69%|1.41%|
|[ResNet152](https://download.pytorch.org/models/resnet152-b121ed2d.pth)|77.55%|77.51%|77.40%|74.95%|9.29%|
|[SqueezeNetV0](https://download.pytorch.org/models/squeezenet1_0-a815701f.pth)|56.73%|56.75%|56.70%|53.93%|14.21%|
|[SqueezeNetV1](https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth)|56.52%|56.52%|56.24%|54.56%|17.10%|
|[InceptionV3](https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth)|76.41%|76.43%|76.44%|73.67%|1.50%|

**Note: ImageNet 32-float models are directly from torchvision**


## Selected Arguments
Here we give an overview of selected arguments of `quantize.py`

|Flag                          |Default value|Description & Options|
|:-----------------------------|:-----------------------:|:--------------------------------|
|type|cifar10|mnist,svhn,cifar10,cifar100,stl10,alexnet,vgg16,vgg16_bn,vgg19,vgg19_bn,resent18,resent34,resnet50,resnet101,resnet152,squeezenet_v0,squeezenet_v1,inception_v3|
|quant_mothod|linear|quantization method:linear,minmax,log,tanh|
|param_bits|8|bit-width of weights and bias|
|fwd_bits|8|bit-width of activation|
|bn_bits|32|bit-width of running mean and running vairance|
|overflow_rate|0.0|overflow rate threshold for linear quantization method|
|n_samples|20|number of samples to make statistics for activation|
