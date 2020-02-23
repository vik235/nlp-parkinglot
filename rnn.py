import torch
import torch.nn as nn 

class RNN(nn.Module):
    def __init__(self, dropout_rate, n_categories, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.n_categories = n_categories
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2o = nn.Linear(self.n_categories + self.input_size + self.hidden_size, self.output_size)
        self.i2h = nn.Linear(self.n_categories + self.input_size + self.hidden_size, self.hidden_size)
        self.o2o = nn.Linear(self.output_size + self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        output = self.i2o(input_combined)
        hidden = self.i2h(input_combined)
        combined_output = torch.cat((output, hidden), dim = 1)
        output = self.o2o(combined_output)
        output = self.dropout(combined_output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)    

        
