import torch 
import torch.nn as nn 
from torch import optim 
import torch.nn.functional as F 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size): # To init the decoder we only need the hidden size (coming from the encoder in conditional seq2seq) and its output_size 
        super(DecoderRNN, self).__init__()
        self.hidden_size= hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        relu = F.relu(embedded)
        output, hidden = self.gru(relu, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


