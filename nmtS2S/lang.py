'''
Lang : helper class to keep track of the words to be used in the vocab of soruce and target in nmt system 
Attributes : 
    word2index
    word2count
    index2word
    n_words 

Methods : 
    addSentence : consumes a List[List[str]] and populates the attribute dicts and scalars.
    addWord : consumes a str and populates the attribute dicts and scalars.
'''

SOS_token = 0 
EOS_token = 1

class Lang: 
    def __init__(self, name):
        self.name = name 
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0 : "SOS", 1 : "EOS"}
        self.n_words = 2 # init with SOS and EOS

    def addWord(self, word): 
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else: 
            self.word2count[word] += 1
    
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    



