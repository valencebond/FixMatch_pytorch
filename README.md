
## FixMatch

The unofficial reimplementation of [fixmatch](https://arxiv.org/abs/2001.07685) with RandomAugment. 

## Overview


|repo|using EMA model to evaluate|using EMA model to train|update parameters|update buffer|
|:---|:---:|:---:|:---:|:---:|
|ours| &check;|-| &check;|-|
|mdiephuis| &check;| &check;| &check;|-|
|kekmodel| &check;|-|-| &check;|


2020-03-30_18:07:08.log : annotation decay and add  classifier.bias

2020-03-31_09:51:38.log : add interleave and run model once

## Dependencies

- python 3.6
- pytorch 1.3.1
- torchvision 0.2.1

The other packages and versions are listed in ```requirements.txt```. 
You can install them by ```pip install -r requirements.txt```.


## Dataset
download cifar-10 dataset: 
```
    $ mkdir -p dataset && cd data
    $ wget -c http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    $ tar -xzvf cifar-10-python.tar.gz
```

download cifar-100 dataset: 
```
    $ mkdir -p dataset && cd data
    $ wget -c http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    $ tar -xzvf cifar-100-python.tar.gz
```

## Train the model

To train the model on CIFAR10 with 40 labeled samples, you can run the script: 
```
    $ CUDA_VISIBLE_DEVICES='0' python train.py --dataset CIFAR10 --n-labeled 40 
```
To train the model on CIFAR100 with 400 labeled samples, you can run the script: 
```
    $ CUDA_VISIBLE_DEVICES='0' python train.py --dataset CIFAR100 --n-labeled 400 
```


## Results


### CIFAR10
| #Labels | 40 | 250 | 4000 |
|:---|:---:|:---:|:---:|
|Paper (RA) | 86.19 ± 3.37 | 94.93 ± 0.65 | 95.74 ± 0.05 |
|ours| 89.63(85.65) | 93.0832 |94.7154|

### CIFAR100

| #Labels | 400 | 2500 | 10000 |
|:---|:---:|:---:|:---:|
|Paper (RA) | 51.15 ± 1.75 | 71.71 ± 0.11 | 77.40 ± 0.12 |
|ours | 53.74 | 67.3169 | 73.26 |


### References
- https://github.com/CoinCheung/fixmatch
- https://github.com/kekmodel/FixMatch-pytorch
- official implement https://github.com/google-research/fixmatch
