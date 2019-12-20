import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = int(word)
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.dictionary.add_word('1')
        self.train = self.tokenize(os.path.join(path, 'train.txt.r'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt.r'))
        self.test = self.tokenize(os.path.join(path, 'test.txt.r'))
        magic_size = 50265
        counter = 0
        while len(self.dictionary) < magic_size:
            self.dictionary.add_word(str(counter))
            counter += 1

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.strip().split(' ')
                for word in words:
                    self.dictionary.add_word(word)
        print(f'dict length: {len(self.dictionary)}')
        max_len=800
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                # words = line.strip().split(' ') + ['<eos>']
                words = line.strip().split(' ')
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                remaining = [self.dictionary.word2idx['1']] * (max_len - len(ids))
                idss.append(torch.tensor(ids+remaining).type(torch.int64))
            ids = torch.cat(idss)

        return ids
