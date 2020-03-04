import torch

import torchtext
from torchtext.data import Field
tokenize = lambda x: x.split()


SeqText = Field(sequential = True, tokenize = tokenize, lower = True, init_token = '<sos>', eos_token = '<eos>')
