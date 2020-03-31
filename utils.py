import datetime
import logging
import os
import sys
import torch

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter


def interleave(x, bt):
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([-1, bt] + s[1:]), 1, 0), [-1] + s[1:])


def de_interleave(x, bt):
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([bt, -1] + s[1:]), 1, 0), [-1] + s[1:])


def setup_default_logging(args, default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s"):
    output_dir = os.path.join(args.dataset, f'x{args.n_labeled}')
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(comment=f'{args.dataset}_{args.n_labeled}')

    logger = logging.getLogger('train')

    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(output_dir, f'{time_str()}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    # print
    # file_handler = logging.FileHandler()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)

    return logger, writer


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)  # return value, indices
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value

    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / (self.count + 1e-20)
        self.avg = self.sum / self.count


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.datetime.today().strftime(fmt)


if __name__ == '__main__':

    a = torch.tensor(range(30))
    a_ = interleave(a, 15)
    a__ = de_interleave(a_, 15)
    print(a, a_, a__)
