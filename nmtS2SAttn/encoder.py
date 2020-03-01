import torch 
import torch.nn as nn 
from typing import Tuple

class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim*2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor]:
        # src : (src_sent_len , batch_size)
        embedded = self.dropout(self.embedding(src))
        #embedded = (src_sent_length, batch_size, emb_dim)
        outputs, hidden = self.rnn(embedded)
        #outputs = (src_sent_length, batch_size, dec_hid_dim*2), needed for calculating attentions 
        #hidden  = (2 - # of directions, batch_size, hidden_size))

        #final layer output of the RNN is the initial hidden for the decoder. We cat forward 
        # and backward states of the final layer, send it through a linear layer and a tanh 
        # activation before feeding it to bootstrap the decoder. 
        # torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1) returns (batch_size, enc_hid_dim * 2)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return outputs, hidden 
