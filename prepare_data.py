from utils.read_data import *
from utils.lang import *
from device import device
import torch
from torch.utils.data import RandomSampler, DataLoader, TensorDataset

BATCH_SIZE = 768
MAX_LENGTH = 32


def prepare_data(ratio, cap):
    #input_lang, output_lang, pairs = read_data()
    #dev_pairs, test_pairs = read_dev_test()
    # eg: ratio = 0.8: train 0.8, dev 0.1, test 0.1
    input_lang, output_lang, pairs, dev_pairs, test_pairs = read_all(ratio, cap, MAX_LENGTH)
    print('Read {} train sentence pairs'.format(len(pairs)))
    print('Read {} dev and {} test sentence pairs'.format(len(dev_pairs), len(test_pairs)))
    print('Counting words...')

    for pair in pairs:
        input_lang.scan_sentence(pair[0])
        output_lang.scan_sentence(pair[1])
    for pair in dev_pairs:
        input_lang.scan_sentence(pair[0])
        output_lang.scan_sentence(pair[1])
    for pair in test_pairs:
        input_lang.scan_sentence(pair[0])
        output_lang.scan_sentence(pair[1])

    print('Vocab size:')
    print(input_lang.name, ':', input_lang.vocab_size)
    print(output_lang.name, ':', output_lang.vocab_size)

    return input_lang, output_lang, pairs, dev_pairs, test_pairs


def indexes_from_sentence(lang, sentence):
    ids = []
    for c in sentence:
        if c in lang.word2index.keys():
            ids.append(lang.word2index[c])
        else:
            ids.append(lang.word2index[lang.unk])
    return pad_to_max(ids)


def dataloader_from_pairs(pairs, input_lang, output_lang):
    input_ids = []
    target_ids = []
    for pair in pairs:
        input = indexes_from_sentence(input_lang, pair[0])
        target = indexes_from_sentence(output_lang, pair[1])
        input_ids.append(input)
        target_ids.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    target_ids = torch.tensor(target_ids, dtype=torch.long, device=device)
    return train_dataloader(input_ids, target_ids)


def train_dataloader(train, target):
    dataset = TensorDataset(train, target)
    return DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=BATCH_SIZE)


def pad_to_max(ids):
    new_ids = ids
    new_ids.append(EOS_token)
    while len(new_ids) < MAX_LENGTH:
        new_ids.append(PAD_token)
    return new_ids
