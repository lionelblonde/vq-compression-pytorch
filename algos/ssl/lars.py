import torch


class LARSWrapper(object):
    """
    Flavor of LARC optimizer
    https://github.com/NVIDIA/apex/blob/d74fda260c403f775817470d87f810f816f3d615/apex/parallel/LARC.py

    to align with the official implementation
    https://github.com/google-research/simclr/blob/master/lars_optimizer.py

    Args:
        `optimizer`:
            Pytorch optimizer to wrap and adaptively modify the learning rate of
        `trust_coeff`:
            Trust coefficient for calculating the adaptive lr
            https://arxiv.org/abs/1708.03888

    this is itself built from
    https://github.com/AndrewAtanov/simclr-pytorch/blob/master/utils/lars_optimizer.py
    """

    def __init__(self, opt, trust_coeff=1e-3):
        self.opt = opt
        self.trust_coeff = trust_coeff

    @property
    def param_groups(self):
        return self.opt.param_groups

    def __getstate__(self):
        return self.opt.__getstate__()

    def __setstate__(self, state):
        self.opt.__setstate__(state)

    def __repr__(self):
        return self.opt.__repr__()

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)

    def zero_grad(self):
        self.opt.zero_grad()

    def add_param_group(self, param_group):
        self.opt.add_param_group(param_group)

    def step(self):

        with torch.no_grad():  # just doing arithmetics here

            # we do the same as in the util:
            # we hijack the weight decay handling from the wrapee opt
            decay = []

            for group in self.opt.param_groups:

                # hijack the decay handling and zero it out
                gwd = group['weight_decay']
                decay.append(gwd)
                group['weight_decay'] = 0.

                for p in group['params']:

                    if p.grad is None:
                        continue

                    if gwd != 0.:
                        # decay the gradient with wd
                        p.grad += gwd * p

                    # compute the adaptive lr
                    p_norm = p.norm()
                    g_norm = p.grad.norm()
                    adaptive_lr = 1.
                    if p_norm != 0. and g_norm != 0. and group['lars']:
                        # we defined ourselves the 'lars' key in `model_util.py`
                        adaptive_lr = self.trust_coeff * p_norm / (g_norm + 1e-9)

                    # add the computed lr to the grad
                    p.grad *= adaptive_lr

        # OUTSIDE the no_grad context, carry out the opt step (ignoring its own wd settings)
        self.opt.step()

        # give back wd handling to the wrapped optimizer
        for i, group in enumerate(self.opt.param_groups):
            group['weight_decay'] = decay[i]

