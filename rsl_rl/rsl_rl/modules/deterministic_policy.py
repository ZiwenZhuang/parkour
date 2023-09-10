import numpy as np

import torch
import torch.nn as nn

class DeterministicPolicyMixin:
    def act(self, *args, **kwargs):
        return_ = super().act(*args, **kwargs)
        return self.action_mean
