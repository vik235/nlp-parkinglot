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

from atttention import *
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
print(model_mod_attn.apply(init_weights))
def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
print(f'The model has {count_parameters(model_mod_attn):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

optimizer_mod_attn = optim.Adam(model_mod_attn.parameters())

PAD_IDX = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)


def train(model: nn.Module, 
          iterator: BucketIterator, 
          optimizer: optim.Adam, 
          criterion: nn.modules.loss.CrossEntropyLoss, 
          clip: float):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]
        
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model: nn.Module, 
        iterator: BucketIterator, 
        criterion: nn.modules.loss.CrossEntropyLoss):    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            #trg = [(trg sent len - 1) * batch size]
            #output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time: int, 
            end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

    
MODEL_PATH = 'tut3-model.pt'
MODEL_PATH_MOD_ATTN = 'tut3-model-modified_attn.pt'



N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model_mod_attn, train_iterator, optimizer_mod_attn, criterion, CLIP)
    valid_loss = evaluate(model_mod_attn, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model_mod_attn.state_dict(), MODEL_PATH_MOD_ATTN)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


model_mod_attn.load_state_dict(torch.load(MODEL_PATH_MOD_ATTN))

test_loss = evaluate(model_mod_attn, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')