import unicodedata
import re

UNK_token = 3
SOS_token = 2
EOS_token = 1
PAD_token = 0


class Lang:
    def __init__(self, name):
        self.name = name
        self.unk = '未'
        self.emp = '空'
        self.sos = '始'
        self.eos = '末'

        self.word2index = {'始': 2, '末': 1, '空': 0, self.unk: 3}
        self.index2word = {2: '始', 1: '末', 0: '空', 3: self.unk}
        self.vocab_size = 4
        self.word_cnt = {}


    def scan_sentence(self, sentence):
        for c in sentence:
            self.scan_word(c)

    def scan_word(self, c):
        if self.name == 'th':
            self.langVocab = 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืฺุูเแโใไๅๆ็่้๊๋์ํ๎'

        if c in self.langVocab:
            if c not in self.word2index.keys():
                self.word2index[c] = self.vocab_size
                self.index2word[self.vocab_size] = c
                self.word_cnt[c] = 1
                self.vocab_size += 1
            else:
                self.word_cnt[c] += 1




def unicode2ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def normalize_two_sentence(s1, s2):
    return [normalize_string(s1), normalize_string(s2)]
