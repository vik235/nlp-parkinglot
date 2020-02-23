# This is being used as is in run.py directly.
import torch
import torch.nn 
from run import * 
from rnn.py import * 
from train.py import *


def sample(category, startingLetter, max_length, rnn): 
    with torch.no_grad(): 
        category_tensor = categoryTensor(category)
        input_tensor = inputTensor(startingLetter)
        hidden = rnn.initHidden()
        output_name = start_letter

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


def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

category_lines = {}
all_categories = []

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

