import torch 
import torch.nn as nn 
from typing import Tuple
from torch import Tensor
import torch.nn.functional as F

class ModifiedAttention(nn.Module):
    def __init__(self, 
                 enc_hid_dim: int, 
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        
        self.attn = nn.Linear(self.attn_in, attn_dim)
        
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

        attention = torch.sum(energy, dim=2)
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)