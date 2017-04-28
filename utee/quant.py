from torch.autograd import Variable
import torch
from torch import nn
from collections import OrderedDict
import math

def approximate_log2_pytorch(input, overflow_rate=0.0):
    from IPython import embed
    assert isinstance(input, Variable), type(input)
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    sf = None
    try:
        v = v.data.cpu().numpy()[0]
        sf = math.ceil(math.log2(v))
    except Exception as e:
        embed()
    return sf

def linear_quantize_pytorch(input, sf=None, bits=8., **kwargs):
    assert isinstance(input, Variable), type(input)
    if sf is None:
        sf = bits - 1. - approximate_log2_pytorch(input, **kwargs)
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)

    mask = rounded.gt(max_val).float() + rounded.lt(min_val).float()
    overflow_rate_real = mask.mean().data.cpu().numpy()[0]

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value, overflow_rate_real, sf


class Quant(nn.Module):
    def __init__(self, name, bits=8.0, sf=None, overflow_rate=0.0, counter=10):
        super(Quant, self).__init__()
        self.name = name
        self.update_sf = True
        self.counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate
        self.overflow_rate_real = -1

    def on(self):
        self.update_sf= True

    def off(self):
        self.update_sf = False

    def forward(self, input):
        if self.update_sf:
            sf_new = self.bits - 1 - approximate_log2_pytorch(input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            self.counter -= 1
            if self.counter <= 0:
                self.update_sf= False
        else:
            assert self.sf is not None, 'sf should not be None'

        output, overflow_rate_real, _ = linear_quantize_pytorch(input, sf=self.sf, bits=self.bits, overflow_rate=self.overflow_rate)
        self.overflow_rate_real = overflow_rate_real
        return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, overflow_rate_real={:.3f}, update_sf={}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.overflow_rate_real,
            self.update_sf, self.counter)

def duplicate_model_with_quant(model, bits, overflow_rate=0.0, counter=10):
    """assume that original model has at least a nn.Sequential"""
    if isinstance(model, nn.Sequential):
        l = OrderedDict()
        for k, v in model._modules.items():
            if isinstance(v, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.AvgPool2d)):
                l[k] = v
                l['{}_quant'.format(k)] = Quant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
            else:
                l[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter)
        m = nn.Sequential(l)
        return m
    else:
        for k, v in model._modules.items():
            # assert isinstance(v, nn.Sequential), type(v)
            model._modules[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter)
        return model

