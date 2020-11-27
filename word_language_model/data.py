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
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    padding = '<pad>'
    bos = '<bos>'
    eos = '<eos>'
    def __init__(self, path):
        self.dictionary = Dictionary()
        for w in [Corpus.padding, Corpus.bos, Corpus.eos]:
            self.dictionary.add_word(w)
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

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
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:

                # words = line.strip().split(' ') + ['<eos>']
                stripped = line.strip().split(' ')
                if len(stripped) > (max_len - 2) or stripped[0] == '=':
                    continue
                words = ['<bos>'] + stripped + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])

                assert len(ids) <= max_len - 2
                remaining = [self.dictionary.word2idx[Corpus.padding]] * (max_len - len(ids))
                idss.append(torch.tensor(ids+remaining).type(torch.long))
            ids = torch.cat(idss)

        return ids
