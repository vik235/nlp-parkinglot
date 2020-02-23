import torch
import torch.nn as nn

criterion  = nn.NLLLoss()
learning_rate = 0.0005

def train(category_tensor, input_line_tensor, target_line_tensor, rnn):
    target_line_tensor = target_line_tensor.unsqueeze(-1)
    hidden = rnn.initHidden()
    rnn.zero_grad()

    loss = 0 

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad_data)

    return output, loss.item() / input_line_tensor.size(0)