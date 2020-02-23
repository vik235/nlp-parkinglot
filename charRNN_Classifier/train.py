import torch 
from util import *
from network import *
import random
import time 
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def timeSince(since):
    now = time.time()
    s = now - since 
    m = math.floor(s / 60.0)
    s -= m*60
    return '%dm %ds' % (m,s)

def categoryToTensor(category):
    tensor = torch.zeros(1, n_categories)
    tensor[0][category] = 1
    return tensor

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    
    return output, loss.item()


all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)

category_lines = {} 
all_categories = []
n_hidden = 128
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

for filename in findFiles('names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
print(n_categories)

print(letterToTensor('J'))
print(lineToTensor('Jonas').size())

rnn = RNN(n_letters, n_hidden, n_categories) 
input = lineToTensor('Albert')
hidden =torch.zeros(1, n_hidden)
output, next_hidden = rnn(input[0], hidden)

print(all_letters)

## training 
criterion = nn.NLLLoss() 
learning_rate = 0.05
current_loss = 0
all_losses = []

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '|| line =', line, '|| categor tensor', category_tensor)

n_iters = 10000
print_every = 100
plot_every = 10

current_loss = 0
all_loss = []

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


plt.figure()
plt.plot(all_losses)