from lang import * 
from encoder import * 
from decoder import * 
from util import *

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentences(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromPair(pair):
    input_tensor = tensorFromSentences(input_lang, pair[0])
    target_tensor = tensorFromSentences(output_lang, pair[1])
    return (input_tensor, target_tensor)