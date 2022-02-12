import math
import torch
import numpy
from torch import nn
from torch import optim
from torch._C import device
from torch.nn import functional
from torch.nn.modules.utils import _single

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000, freq=10000.):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(freq) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div) # Even terms
        pe[:, 1::2] = torch.cos(position * div) # Odd terms
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class CausalConv1d(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride=1, dilation=1):
        super().__init__()
        
        self.pad = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        input = functional.pad(input, (self.pad, 0))        #left pad
        x = self.conv(input)
        return x
    
def mu_law_encode(input):
    out = numpy.empty_like(input.cpu(), dtype=numpy.float16)

    for x in range(len(input)):
        if abs(input[x]) > 1.:
            raise ValueError
    
        sign = 1
        if input[x] < 0:
            sign = -1

        out[x] = sign * math.log(1 + (255 * abs(input[x]))) / math.log(256)

    return torch.tensor(((out + 1) / 2) * 255, dtype=torch.uint8)

def mu_law_decode(input):
    out = numpy.empty_like(input, dtype=numpy.float16)

    input = numpy.asarray(((input / 255) * 2) - 1, dtype=numpy.float16)

    for x in range(len(input)):
        if abs(input[x]) > 1.:
            raise ValueError
    
        sign = 1
        if input[x] < 0:
            sign = -1

        out[x] = sign * (1 / 255) * (256 ** abs(input[x]) - 1)

    return torch.tensor(out)
