import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch import Tensor

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator 


import spacy

from typing import Tuple
import random
import math
import time

from attention import * 
from encoder import * 
from decoder import * 
from modifiedattention import * 
from seq2seq import *

#Set the seed for the reproducibility
SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
SRC = Field(tokenize= "spacy", 
            tokenizer_language="de", 
            init_token='<sos>', 
            eos_token = '<eos>',
            lower = True)

TRG = Field(tokenize = "spacy", 
            tokenizer_language="en", 
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),fields = (SRC, TRG))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(datasets = (train_data, valid_data, test_data), 
                                                                     batch_size = BATCH_SIZE,
                                                                     device = device)


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ATTN_DIM = 64
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

mod_attn = ModifiedAttention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec_mod_attn = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, mod_attn)

model_mod_attn = Seq2Seq(enc, dec_mod_attn, device).to(device)

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)
model_mod_attn.apply(init_weights)