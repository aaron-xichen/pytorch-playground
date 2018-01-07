import torch.nn as nn

class mIdentity(nn.Module):
    def forward(self, input):
        return input

