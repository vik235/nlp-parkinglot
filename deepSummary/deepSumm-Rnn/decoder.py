import torch 
import torch.nn as nn 
from typing import Tuple
from torch import Tensor
import torch.nn.functional as F

class Decoder1(nn.Module):
    def __init__(self, 
                 output_dim: int, 
                 emb_dim: int, 
                 enc_hid_dim: int, 
                 dec_hid_dim: int, 
                 dropout: int, 
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention
        # Note: from Attention: self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, attn_dim)
        
        # Note: `output_dim` same as `vocab_size`
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        
    def _weighted_encoder_rep(self, 
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:
        
        # Attention, at a high level, takes in:
        # The decoder hidden state
        # All the "seq_len" encoder outputs
        # Outputs a vector summing to 1 of length seq_len for each observation
        a = self.attention(decoder_hidden, encoder_outputs)

        #a = [batch size, src len]

        a = a.unsqueeze(1)

        #a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        #weighted_encoder_rep = [batch size, 1, enc hid dim * 2]

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        #weighted_encoder_rep = [1, batch size, enc hid dim * 2]
        
        return weighted_encoder_rep
        
        
    def forward(self, 
                input: Tensor, 
                decoder_hidden: Tensor, 
                encoder_outputs: Tensor) -> Tuple[Tensor]:
             
        #input = [batch size] Note: "one character at a time"
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden, 
                                                          encoder_outputs)
        
        # Then, the input to the decoder _for this character_ is a concatenation of:
        # This weighted attention
        # The embedding itself
        
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
        
        #output = [sent len, batch size, dec hid dim * n directions]
        #decoder_hidden = [n layers * n directions, batch size, dec hid dim]
        
        #sent len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == decoder_hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
        
        output = self.out(torch.cat((output, 
                                     weighted_encoder_rep, 
                                     embedded), dim = 1))
        
        #output = [bsz, output dim]
        
        return output, decoder_hidden.squeeze(0)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        #mask = [batch size, src len]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs, mask)
                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0), a.squeeze(1)  