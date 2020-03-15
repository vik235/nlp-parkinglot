import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

from typing import Tuple
import random
import math
import time
import os
import csv

from encoder import Encoder
from encoderlayer import EncoderLayer
from decoder import Decoder
from decoderlayer import DecoderLayer
from feedforward import PositionwiseFeedforwardLayer
from multiheadattention import MultiHeadAttentionLayer
from seq2seq import Seq2Seq

from train import * 
from utils import *


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


spacy_en = spacy.load('en')
spacy_de = spacy.load('de')

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

SRC = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True,
            batch_first= True)

TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first= True)



train_data = TranslationDataset(path = './data/functions.train', exts = ('.src', '.tgt'), fields = (SRC, TRG))
test_data = TranslationDataset(path = './data/functions.test', exts = ('.src', '.tgt'), fields = (SRC, TRG))
valid_data = TranslationDataset(path = './data/functions.valid', exts = ('.src', '.tgt'), fields = (SRC, TRG))
'''

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                    fields = (SRC, TRG))
'''
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

print('Built the Vocab of SRC and TRG')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

BATCH_SIZE = 256
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 2
DEC_LAYERS = 2
ENC_HEADS = 2
DEC_HEADS = 2
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
LEARNING_RATE = 0.0005
N_EPOCHS = 5
CLIP = 1
MODEL_NAME = 'deepSumm-model.pt'

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)


enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)
dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)


print(f'The model has {count_parameters(model):,} trainable parameters')

model.apply(initialize_weights)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
train_losses = []
valid_losses = []

best_valid_loss = float('inf')
time_started = time.time()
for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    eval_context = 'Train\Validation set'
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion, eval_context)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    start_mins, start_secs = epoch_time(time_started, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        print('\t','-'*5, f'Saving the model. Valid Loss - {valid_loss} and Best valid loss so far {best_valid_loss}','-'*5)
        torch.save(model.state_dict(), MODEL_NAME)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s ||  Time Since Start: {start_mins}m {start_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load(MODEL_NAME))

with open(file = './output/train_output.csv', mode ='w') as f:
            writer = csv.writer(f, delimiter='\n', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows([train_losses])
with open(file = './output/valid_output.csv', mode ='w') as f:
            writer = csv.writer(f, delimiter='\n', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows([valid_losses])            

eval_context = 'Test set'
test_loss = evaluate(model, test_iterator, criterion, eval_context)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)

print(f'BLEU score = {bleu_score*100:.2f}')