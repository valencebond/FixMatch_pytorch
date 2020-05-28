import os.path as osp
import pickle
import numpy as np
import os

import torch
import pandas as pd
from torch.utils.data import Dataset
from datasets import transform as T
from torchvision import transforms

from datasets.randaugment import RandomAugment
from datasets.sampler import RandomSampler, BatchSampler

class_to_idx = {
    "chinee apple": 0,
    "lantana": 1,
    "parkinsonia": 2,
    "parthenium": 3,
    "prickly acacia": 4,
    "rubber vine": 5,
    "siam weed": 6,
    "snake weed": 7,
    "negative": 8,
}

def cleanup(files):
    if ".DS_Store" in files:
        files.remove(".DS_Store")

    return files

def load_data_train(L=250, dataset='CIFAR10', dspth='./data'):
    if dataset == 'CIFAR10':
        datalist = [
            osp.join(dspth, 'cifar-10-batches-py', 'data_batch_{}'.format(i + 1))
            for i in range(5)
        ]
        n_class = 10
        assert L in [40, 250, 4000]
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar-100-python', 'train')]
        n_class = 100
        assert L in [400, 2500, 10000]

    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    n_labels = L // n_class
    data_x, label_x, data_u, label_u = [], [], [], []
    for i in range(n_class):
        indices = np.where(labels == i)[0]
        np.random.shuffle(indices)
        inds_x, inds_u = indices[:n_labels], indices[n_labels:]
        data_x += [
            data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            for i in inds_x
        ]
        label_x += [labels[i] for i in inds_x]
        data_u += [
            data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            for i in inds_u
        ]
        label_u += [labels[i] for i in inds_u]
    return data_x, label_x, data_u, label_u


# def load_data_train(L=250, dspth='./data'):
#     datalist = [
#         osp.join(dspth, 'cifar-10-batches-py', 'data_batch_{}'.format(i + 1))
#         for i in range(5)
#     ]
#     data, labels = [], []
#     for data_batch in datalist:
#         with open(data_batch, 'rb') as fr:
#             entry = pickle.load(fr, encoding='latin1')
#             lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
#             data.append(entry['data'])
#             labels.append(lbs)
#     data = np.concatenate(data, axis=0)
#     labels = np.concatenate(labels, axis=0)
#     n_labels = L // 10
#     data_x, label_x, data_u, label_u = [], [], [], []
#     for i in range(10):
#         indices = np.where(labels == i)[0]
#         np.random.shuffle(indices)
#         inds_x, inds_u = indices[:n_labels], indices[n_labels:]
#         data_x += [
#             data[i].reshape(3, 32, 32).transpose(1, 2, 0)
#             for i in inds_x
#         ]
#         label_x += [labels[i] for i in inds_x]
#         data_u += [
#             data[i].reshape(3, 32, 32).transpose(1, 2, 0)
#             for i in inds_u
#         ]
#         label_u += [labels[i] for i in inds_u]
#     return data_x, label_x, data_u, label_u


def load_data_val(dataset, dspth='./data'):
    if dataset == 'CIFAR10':
        datalist = [
            osp.join(dspth, 'cifar-10-batches-py', 'test_batch')
        ]
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar-100-python', 'test')
        ]

    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = [
        el.reshape(3, 32, 32).transpose(1, 2, 0)
        for el in data
    ]
    return data, labels


def compute_mean_var():
    data_x, label_x, data_u, label_u = load_data_train()
    data = data_x + data_u
    data = np.concatenate([el[None, ...] for el in data], axis=0)

    mean, var = [], []
    for i in range(3):
        channel = (data[:, :, :, i].ravel() / 127.5) - 1
        #  channel = (data[:, :, :, i].ravel() / 255)
        mean.append(np.mean(channel))
        var.append(np.std(channel))

    print('mean: ', mean)
    print('var: ', var)

class CustomDataset(Dataset):
    def __init__(self, im_folder, label_file):
        self.im_folder = im_folder
        self.labels_df = pd.read_csv(label_file)
        self.total_imgs = cleanup(sorted(os.listdir(self.im_folder)))
        self.labels = [
            class_to_idx[label.lower()] for label in self.labels_df["Species"].to_list()
        ]

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.im_folder, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        label = torch.as_tensor(self.labels[idx])
        return image, label

class DeepWeeds(CustomDataset):
    def __init__(self, im_folder, label_file, is_train=True):
        super(DeepWeeds, self).__init__(im_folder, label_file)
        self.is_train = is_train
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        if is_train:
            self.trans_weak = T.Compose([
                T.Resize((224, 224)),
                T.PadandRandomCrop(border=4, cropsize=(224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            self.trans_strong = T.Compose([
                T.Resize((224, 224)),
                T.PadandRandomCrop(border=4, cropsize=(224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
        else:
            self.trans = T.Compose([
                T.Resize((224, 224)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])

class Cifar(Dataset):
    def __init__(self, dataset, data, labels, is_train=True):
        super(Cifar, self).__init__()
        self.data, self.labels = data, labels
        self.is_train = is_train
        assert len(self.data) == len(self.labels)
        if dataset == 'CIFAR10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        elif dataset == 'CIFAR100':
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        if is_train:
            self.trans_weak = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            self.trans_strong = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
        else:
            self.trans = T.Compose([
                T.Resize((32, 32)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        if self.is_train:
            return self.trans_weak(im), self.trans_strong(im), lb
        else:
            return self.trans(im), lb

    def __len__(self):
        leng = len(self.data)
        return leng


def get_train_loader(dataset, batch_size, mu, n_iters_per_epoch, L, root='data'):
    if dataset.startswith("CIFAR"):
        data_x, label_x, data_u, label_u = load_data_train(L=L, dataset=dataset, dspth=root)

        ds_x = Cifar(
            dataset=dataset,
            data=data_x,
            labels=label_x,
            is_train=True
        )  # return an iter of num_samples length (all indices of samples)
        sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
        batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
        dl_x = torch.utils.data.DataLoader(
            ds_x,
            batch_sampler=batch_sampler_x,
            num_workers=2,
            pin_memory=True
        )
        ds_u = Cifar(
            dataset=dataset,
            data=data_u,
            labels=label_u,
            is_train=True
        )
        sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
        batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
        dl_u = torch.utils.data.DataLoader(
            ds_u,
            batch_sampler=batch_sampler_u,
            num_workers=2,
            pin_memory=True
        )
    else:
        ds = DeepWeeds("/home/ubuntu/Home/data/blueriver/DeepWeeds/images", "/home/ubuntu/Home/data/blueriver/DeepWeeds/labels/labels.csv", is_train=True)
        size = len(ds)
        
        ds_u, ds_x = torch.utils.data.random_split(ds, [round(size * 0.05), round(size * 0.95)])

        sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
        batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)

        dl_x = torch.utils.data.DataLoader(
            ds_x,
            batch_sampler=batch_sampler_x,
            num_workers=2,
            pin_memory=True
        )

        sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
        batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)

        dl_u = torch.utils.data.DataLoader(
            ds_u,
            batch_sampler=batch_sampler_u,
            num_workers=2,
            pin_memory=True
        )

    return dl_x, dl_u


def get_val_loader(dataset, batch_size, num_workers, pin_memory=True):
    data, labels = load_data_val(dataset)
    ds = Cifar(
        dataset=dataset,
        data=data,
        labels=labels,
        is_train=False
    )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dl


class OneHot(object):
    def __init__(
            self,
            n_labels,
            lb_ignore=255,
    ):
        super(OneHot, self).__init__()
        self.n_labels = n_labels
        self.lb_ignore = lb_ignore

    def __call__(self, label):
        N, *S = label.size()
        size = [N, self.n_labels] + S
        lb_one_hot = torch.zeros(size)
        if label.is_cuda:
            lb_one_hot = lb_one_hot.cuda()
        ignore = label.data.cpu() == self.lb_ignore
        label[ignore] = 0
        lb_one_hot.scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(self.n_labels), *b]] = 0

        return lb_one_hot


if __name__ == "__main__":
    compute_mean_var()
    #  dl_x, dl_u = get_train_loader(64, 250, 2, 2)
    #  dl_x2 = iter(dl_x)
    #  dl_u2 = iter(dl_u)
    #  ims, lb = next(dl_u2)
    #  print(type(ims))
    #  print(len(ims))
    #  print(ims[0].size())
    #  print(len(dl_u2))
    #  for i in range(1024):
    #      try:
    #          ims_x, lbs_x = next(dl_x2)
    #          #  ims_u, lbs_u = next(dl_u2)
    #          print(i, ": ", ims_x[0].size())
    #      except StopIteration:
    #          dl_x2 = iter(dl_x)
    #          dl_u2 = iter(dl_u)
    #          ims_x, lbs_x = next(dl_x2)
    #          #  ims_u, lbs_u = next(dl_u2)
    #          print('recreate iterator')
    #          print(i, ": ", ims_x[0].size())
