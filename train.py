from __future__ import print_function
import random

import time
import argparse
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import WideResnet
from datasets.cifar import get_train_loader, get_val_loader
from label_guessor import LabelGuessor
from lr_scheduler import WarmupCosineLrScheduler
from models.ema import EMA

from utils import accuracy, setup_default_logging, interleave, de_interleave

from utils import AverageMeter
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def set_model(args):
    model = WideResnet(n_classes=10 if args.dataset == 'CIFAR10' else 100,
                       k=args.wresnet_k, n=args.wresnet_n)  # wresnet-28-2

    model.train()
    model.to(device)
    criteria_x = nn.CrossEntropyLoss().to(device)
    criteria_u = nn.CrossEntropyLoss(reduction='none').to(device)
    return model, criteria_x, criteria_u


def train_one_epoch(epoch,
                    model,
                    criteria_x,
                    criteria_u,
                    optim,
                    lr_schdlr,
                    ema,
                    dltrain_x,
                    dltrain_u,
                    lb_guessor,
                    lambda_u,
                    n_iters,
                    logger,
                    ):
    model.train()
    # loss_meter, loss_x_meter, loss_u_meter, loss_u_real_meter = [], [], [], []
    loss_meter = AverageMeter()
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()
    loss_u_real_meter = AverageMeter()
    # the number of correctly-predicted and gradient-considered unlabeled data
    n_correct_u_lbs_meter = AverageMeter()
    # the number of gradient-considered strong augmentation (logits above threshold) of unlabeled samples
    n_strong_aug_meter = AverageMeter()
    mask_meter = AverageMeter()

    epoch_start = time.time()  # start time
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    for it in range(n_iters):
        ims_x_weak, ims_x_strong, lbs_x = next(dl_x)
        ims_u_weak, ims_u_strong, lbs_u_real = next(dl_u)

        lbs_x = lbs_x.to(device)
        lbs_u_real = lbs_u_real.to(device)

        # --------------------------------------

        bt = ims_x_weak.size(0)
        mu = int(ims_u_weak.size(0) // bt)
        imgs = torch.cat([ims_x_weak, ims_u_weak, ims_u_strong], dim=0).to(device)
        imgs = interleave(imgs, 2 * mu + 1)
        logits = model(imgs)
        logits = de_interleave(logits, 2 * mu + 1)

        logits_x = logits[:bt]
        logits_u_w, logits_u_s = torch.split(logits[bt:], bt * mu)

        loss_x = criteria_x(logits_x, lbs_x)

        with torch.no_grad():
            probs = torch.softmax(logits_u_w, dim=1)
            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(0.95).float()

        loss_u = (criteria_u(logits_u_s, lbs_u_guess) * mask).mean()
        loss = loss_x + lambda_u * loss_u
        loss_u_real = (F.cross_entropy(logits_u_s, lbs_u_real) * mask).mean()

        # --------------------------------------


        # mask, lbs_u_guess = lb_guessor(model, ims_u_weak.cuda())
        # n_x = ims_x_weak.size(0)
        # ims_x_u = torch.cat([ims_x_weak, ims_u_strong]).cuda()
        # logits_x_u = model(ims_x_u)
        # logits_x, logits_u = logits_x_u[:n_x], logits_x_u[n_x:]
        # loss_x = criteria_x(logits_x, lbs_x)
        # loss_u = (criteria_u(logits_u, lbs_u_guess) * mask).mean()
        # loss = loss_x + lambda_u * loss_u
        # loss_u_real = (F.cross_entropy(logits_u, lbs_u_real) * mask).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        ema.update_params()
        lr_schdlr.step()

        loss_meter.update(loss.item())
        loss_x_meter.update(loss_x.item())
        loss_u_meter.update(loss_u.item())
        loss_u_real_meter.update(loss_u_real.item())
        mask_meter.update(mask.mean().item())

        corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_strong_aug_meter.update(mask.sum().item())

        if (it + 1) % 512 == 0:
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)

            logger.info("epoch:{}, iter: {}. loss: {:.4f}. loss_u: {:.4f}. loss_x: {:.4f}. loss_u_real: {:.4f}. "
                        "n_correct_u: {:.2f}/{:.2f}. Mask:{:.4f} . LR: {:.4f}. Time: {:.2f}".format(
                epoch, it + 1, loss_meter.avg, loss_u_meter.avg, loss_x_meter.avg, loss_u_real_meter.avg,
                n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg, mask_meter.avg, lr_log, t))

            epoch_start = time.time()

    ema.update_buffer()
    return loss_meter.avg, loss_x_meter.avg, loss_u_meter.avg, loss_u_real_meter.avg, mask_meter.avg


def evaluate(ema, dataloader, criterion):
    # using EMA params to evaluate performance
    ema.apply_shadow()
    ema.model.eval()
    ema.model.to(device)

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    # matches = []
    with torch.no_grad():
        for ims, lbs in dataloader:
            ims = ims.to(device)
            lbs = lbs.to(device)
            logits = ema.model(ims)
            loss = criterion(logits, lbs)
            scores = torch.softmax(logits, dim=1)
            top1, top5 = accuracy(scores, lbs, (1, 5))
            loss_meter.update(loss.item())
            top1_meter.update(top1.item())
            top5_meter.update(top5.item())

    # note roll back model current params to continue training
    ema.restore()
    return top1_meter.avg, top5_meter.avg, loss_meter.avg


def main():
    parser = argparse.ArgumentParser(description=' FixMatch Training')
    parser.add_argument('--wresnet-k', default=2, type=int,
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int,
                        help='depth of wide resnet')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='number of classes in dataset')
    # parser.add_argument('--n-classes', type=int, default=100,
    #                     help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=40,
                        help='number of labeled samples for training')
    parser.add_argument('--n-epoches', type=int, default=1024,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='train batch size of labeled samples')
    parser.add_argument('--mu', type=int, default=7,
                        help='factor of train batch size of unlabeled samples')
    parser.add_argument('--thr', type=float, default=0.95,
                        help='pseudo label threshold')
    parser.add_argument('--n-imgs-per-epoch', type=int, default=64 * 1024,
                        help='number of training images for each epoch')
    parser.add_argument('--lam-u', type=float, default=1.,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--ema-alpha', type=float, default=0.999,
                        help='decay rate for ema module')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=-1,
                        help='seed for random behaviors, no seed if negtive')

    args = parser.parse_args()

    logger, writer = setup_default_logging(args)
    logger.info(dict(args._get_kwargs()))

    # global settings
    #  torch.multiprocessing.set_sharing_strategy('file_system')
    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        # torch.backends.cudnn.deterministic = True

    n_iters_per_epoch = args.n_imgs_per_epoch // args.batchsize  # 1024
    n_iters_all = n_iters_per_epoch * args.n_epoches  # 1024 * 1024

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.n_labeled}")
    logger.info(f"  Num Epochs = {n_iters_per_epoch}")
    logger.info(f"  Batch size per GPU = {args.batchsize}")
    # logger.info(f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {n_iters_all}")

    model, criteria_x, criteria_u = set_model(args)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    dltrain_x, dltrain_u = get_train_loader(
        args.dataset, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled)
    dlval = get_val_loader(dataset=args.dataset, batch_size=64, num_workers=2)

    lb_guessor = LabelGuessor(thresh=args.thr)

    ema = EMA(model, args.ema_alpha)

    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        # if len(param.size()) == 1:
        if 'bn' in name:
            non_wd_params.append(param)  # bn.weight, bn.bias and classifier.bias
            # print(name)
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay,
                            momentum=args.momentum, nesterov=True)
    lr_schdlr = WarmupCosineLrScheduler(
        optim, max_iter=n_iters_all, warmup_iter=0
    )

    train_args = dict(
        model=model,
        criteria_x=criteria_x,
        criteria_u=criteria_u,
        optim=optim,
        lr_schdlr=lr_schdlr,
        ema=ema,
        dltrain_x=dltrain_x,
        dltrain_u=dltrain_u,
        lb_guessor=lb_guessor,
        lambda_u=args.lam_u,
        n_iters=n_iters_per_epoch,
        logger=logger
    )
    best_acc = -1
    best_epoch = 0
    logger.info('-----------start training--------------')
    for epoch in range(args.n_epoches):
        train_loss, loss_x, loss_u, loss_u_real, mask_mean = train_one_epoch(epoch, **train_args)
        # torch.cuda.empty_cache()

        top1, top5, valid_loss = evaluate(ema, dlval, criteria_x)

        writer.add_scalars('train/1.loss', {'train': train_loss,
                                            'test': valid_loss}, epoch)
        writer.add_scalar('train/2.train_loss_x', loss_x, epoch)
        writer.add_scalar('train/3.train_loss_u', loss_u, epoch)
        writer.add_scalar('train/4.train_loss_u_real', loss_u_real, epoch)
        writer.add_scalar('train/5.mask_mean', mask_mean, epoch)
        writer.add_scalars('test/1.test_acc', {'top1': top1, 'top5': top5}, epoch)
        # writer.add_scalar('test/2.test_loss', loss, epoch)

        # best_acc = top1 if best_acc < top1 else best_acc
        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch

        logger.info("Epoch {}. Top1: {:.4f}. Top5: {:.4f}. best_acc: {:.4f} in epoch{}".
                    format(epoch, top1, top5, best_acc, best_epoch))

    writer.close()


if __name__ == '__main__':
    main()
