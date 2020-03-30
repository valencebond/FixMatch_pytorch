import torch
import torch.distributed as dist

"""
为什么 param 和 buffer 要采用不同的的更新策略
param 是 指数移动平均数，buffer 不是
"""


class EMA(object):
    def __init__(self, model, alpha=0.999):
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        # num_batches_tracked, running_mean, running_var in bn
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        # decay = min(self.alpha, (self.step + 1) / (self.step + 10))  # ????
        decay = self.alpha
        state = self.model.state_dict()  # current params
        for name in self.param_keys:
            self.shadow[name].copy_(
                decay * self.shadow[name] + (1 - decay) * state[name]
            )
        # for name in self.buffer_keys:
        #     self.shadow[name].copy_(
        #         decay * self.shadow[name]
        #         + (1 - decay) * state[name]
        #     )

        self.step += 1

    def update_buffer(self):
        # without EMA
        state = self.model.state_dict()
        for name in self.buffer_keys:
            self.shadow[name].copy_(state[name])

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }


if __name__ == '__main__':
    print('=====')
    model = torch.nn.BatchNorm1d(5)
    ema = EMA(model, 0.9, 0.02, 0.002)
    inten = torch.randn(10, 5)
    out = model(inten)
    ema.update_params()
    print(model.state_dict())
    ema.update_buffer()
    print(model.state_dict())
