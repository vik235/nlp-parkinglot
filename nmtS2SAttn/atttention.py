import torch 
import torch.nn as nn 
from typing import Tuple
from torch import Tensor
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int):
        super(Attention, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        
        self.attn = nn.Linear(self.attn_in, attn_dim)
        self.v = nn.Parameter(torch.rand(attn_dim))
        
    def forward(self, 
                decoder_hidden: Tensor, 
                encoder_outputs: Tensor) -> Tensor:
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #decoder_hidden = [batch size, src sent len, dec hid dim]
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        
        # Step 1: to enable feeding through "self.attn" pink box above, concatenate 
        # `repeated_decoder_hidden` and `encoder_outputs`:
        # torch.cat((hidden, encoder_outputs), dim = 2) has shape 
        # [batch_size, seq_len, enc_hid_dim * 2 + dec_hid_dim]
        
        # Step 2: feed through self.attn to end up with:
        # [batch_size, seq_len, attn_dim]
        
        # Step 3: feed through tanh       
        
        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden, 
            encoder_outputs), 
            dim = 2))) 
        
        #energy = [batch size, src sent len, attn_dim]
        
        energy = energy.permute(0, 2, 1)
        
        #energy = [batch size, attn_dim, src sent len]
        
        #v = [attn_dim]
        
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        
        #v = [batch size, 1, attn_dim]
        
        # High level: energy a function of both encoder element outputs and most recent decoder hidden state,
        # of shape attn_dim x enc_seq_len for each observation
        # v, being 1 x attn_dim, transforms this into a vector of shape 1 x enc_seq_len for each observation
        # Then, we take the softmax over these to get the output of the attention function

        attention = torch.bmm(v, energy).squeeze(1)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)
