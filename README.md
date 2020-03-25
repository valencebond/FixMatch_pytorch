
# FixMatch

This is my implementation of the experiment in the paper of [fixmatch](https://arxiv.org/abs/2001.07685). 


## Environment setup

My platform is: 
* 2080ti gpu
* ubuntu-16.04
* python3.6.9
* pytorch-1.3.1 installed via conda
* cudatoolkit-10.1.243 
* cudnn-7.6.3 in /usr/lib/x86_64-linux-gpu


## Dataset
download cifar-10 dataset: 
```
    $ mkdir -p dataset && cd dataset
    $ wget -c http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    $ tar -xzvf cifar-10-python.tar.gz
```

## Train the model

To train the model with 40 labeled samples, you can run the script: 
```
    $ python train.py --n-labeled 40 
```
where `40` is the number of labeled sample during training.


## Results
After training the model with 40 labeled samples for 5 times with the command:
```
    $ python train.py --n-labeled 40 
```
I observed top-1 accuracy like this:  

| #No. | 1 | 2 | 3 | 4 | 5 |
|:---|:---:|:---:|:---:|:---:|:---:|
|acc | 91.81 | 91.29 | 89.51 | 91.32 | 79.42 |


Note: 
I only implemented experiements on cifar-10 dataset without CTAugment.
