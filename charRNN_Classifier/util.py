from __future__ import unicode_literals, print_function, division
from io import open 
import glob
import os 

import unicodedata
import string 

import torch
import random

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def findFiles(path): 
    return glob.glob(path)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for l, letter in enumerate(line):
        tensor[l][0][letterToIndex(letter)] = 1
    return tensor

