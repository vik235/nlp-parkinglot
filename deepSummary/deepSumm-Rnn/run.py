import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor


import torchtext
from torchtext.data import Field

import spacy
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

from typing import Tuple
import random
import math
import time
import os
import csv

from train import *
from utils import *
from encoder import Encoder
from attention import Attention
from decoder import Decoder
from seq2seq import Seq2Seq


spacy_en = spacy.load('en')

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True,
            include_lengths = True)

TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)


train_data = TranslationDataset(path = './data/functions.train', exts = ('.src', '.tgt'), fields = (SRC, TRG))
test_data = TranslationDataset(path = './data/functions.test', exts = ('.src', '.tgt'), fields = (SRC, TRG))
valid_data = TranslationDataset(path = './data/functions.valid', exts = ('.src', '.tgt'), fields = (SRC, TRG))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

print('Built the Vocab of SRC and TRG')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




BATCH_SIZE = 128
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.3
DEC_DROPOUT = 0.3
LEARNING_RATE = 0.001
N_EPOCHS = 10
CLIP = 1

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(  
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)

print('Vocab SRC Dim', INPUT_DIM)
print('Vocab TRG Dim', OUTPUT_DIM)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)

#model = Seq2Seq(enc, dec, device).to(device)
#mod_attn = ModifiedAttention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
#dec_mod_attn = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, mod_attn)
#model_mod_attn = Seq2Seq(enc, dec_mod_attn, device).to(device)
            
print(model.apply(init_weights))
#print(model_mod_attn.apply(init_weights))

print(f'The model has {count_parameters(model):,} trainable parameters')
#print(f'The model has {count_parameters(model_mod_attn):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
#optimizer_mod_attn = optim.Adam(model_mod_attn.parameters())

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

MODEL_PATH = 'deepSummary-model.pt'
MODEL_PATH_MOD_ATTN = 'deepSummary-model-modified_attn.pt'



best_valid_loss = float('inf')
time_started = time.time()
for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    torch.cuda.empty_cache()
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    start_mins, start_secs = epoch_time(time_started, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_PATH)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s ||  Time Since Start: {start_mins}m {start_secs}s  ')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load(MODEL_PATH))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)

print(f'BLEU score = {bleu_score*100:.2f}')