class OptimizerWrapper(object):

    def __init__(self, optimizer):
        self.optimizer = optimizer
        # self.step_num = 0
        # self.lr = 0.1

    def clip_gradient(self, clip_val):
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-clip_val, clip_val)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        # self._update_lr()
        self.optimizer.step()

    def adjust_lr(self, lr):
        for param in self.optimizer.param_groups:
            param['lr'] = lr
