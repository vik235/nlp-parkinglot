from io import open
import glob
import os
import unicodedata
import string
import random

import time
import math

import torch 
import torch.nn as nn 
import torch.optim as optim
from rnn import *


all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker
print(all_letters)

'''

'''
def findFiles(path):
    return glob.glob(path)

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename):
    '''
    Read a file and returns List[List[string]]

    '''
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# INPUT ::: One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# TARGET(Seq) ::: LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)    

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def sample(category, startingLetter, max_length, rnn): 
    with torch.no_grad(): 
        category_tensor = categoryTensor(category)
        input_tensor = inputTensor(startingLetter)
        input_tensor= input_tensor.squeeze(dim = 0)
        hidden = rnn.initHidden()
        output_name = startingLetter

        for i in range(max_length): 
            output, hidden = rnn(category_tensor, input_tensor, hidden )
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)
        return output_name


def samples(category, start_letters, max_length, rnn):
    for start_letter in start_letters:
        print(sample(category, start_letter, max_length, rnn))


def train(category_tensor, input_line_tensor, target_line_tensor, rnn, optimizer):
    target_line_tensor = target_line_tensor.unsqueeze(-1)
    hidden = rnn.initHidden()
    optimizer.zero_grad()

    loss = 0 

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    optimizer.step()

    return output, loss.item() / input_line_tensor.size(0)

category_lines = {}
all_categories = []

n_iters = 10#50000
print_every = 1#5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters
criterion  = nn.NLLLoss()
learning_rate = 0.0005
dropout_rate = 0.3
hidden_size = 128
input_size = n_letters
output_size = n_letters
start = time.time()
max_length = 20

for filename in findFiles('names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')



rnn = RNN(dropout_rate, n_categories, input_size, hidden_size, output_size)
optimizer = optim.Adam(rnn.parameters(), lr = learning_rate)

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample(), rnn, optimizer)
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

samples('English', 'ABCXYZ', max_length, rnn)
samples('Russian', 'ABCRUSX', max_length, rnn)
samples('Spanish', 'ABCXYZSPA', max_length, rnn)
samples('Chinese', 'CHI', max_length, rnn)
