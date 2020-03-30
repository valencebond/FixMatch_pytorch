import torch


class LabelGuessor(object):

    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, model, ims):
        org_state = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }
        is_train = model.training
        with torch.no_grad():
            model.train()
            logits = model(ims)
            probs = torch.softmax(logits, dim=1)
            scores, lbs = torch.max(probs, dim=1)
            mask = scores.ge(self.thresh).float()

        # note it is necessary to keep org_state! especially for bn layer
        # for k, v in org_state.items():
        #     if not all((model.state_dict()[k] == v).reshape(-1)):
        #         print(f'{k} diff')

        model.load_state_dict(org_state)
        if is_train:
            model.train()
        else:
            model.eval()
        return mask, lbs
