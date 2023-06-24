from utils.lang import *


def read_all(ratio, cap, MAX_LENGTH):
    print('Reading ...')

    ori_lines_all = open('./data/spellCorrect/th-ori.txt', encoding='utf-8').read().strip().split('\n')
    cor_lines_all = open('./data/spellCorrect/th-cor.txt', encoding='utf-8').read().strip().split('\n')

    ori_lines_all = [ ori_line for ori_line in ori_lines_all if len(ori_line) < MAX_LENGTH]
    cor_lines_all = [ cor_line for cor_line in cor_lines_all if len(cor_line) < MAX_LENGTH]

    cap_n = int(len(ori_lines_all) * cap)
    ori_lines_all = ori_lines_all[:cap_n]
    cor_lines_all = cor_lines_all[:cap_n]
    
    split_n = int(len(ori_lines_all) * ratio)
    ori_lines_train = ori_lines_all[:split_n]
    cor_lines_train = cor_lines_all[:split_n]
    ori_lines_nonTrain = ori_lines_all[split_n:]
    cor_lines_nonTrain = cor_lines_all[split_n:]

    split_n = int(len(ori_lines_nonTrain) * 0.5)
    ori_lines_dev = ori_lines_nonTrain[:split_n]
    cor_lines_dev = cor_lines_nonTrain[:split_n]
    ori_lines_test = ori_lines_nonTrain[split_n:]
    cor_lines_test = cor_lines_nonTrain[split_n:]


    input_lang = Lang('th')
    output_lang = Lang('th')

    pairs = []
    for i in range(len(ori_lines_train)):
        pairs.append([ori_lines_train[i], cor_lines_train[i]])

    dev_pairs = []
    for i in range(len(ori_lines_dev)):
        dev_pairs.append((ori_lines_dev[i], cor_lines_dev[i]))

    test_pairs = []
    for i in range(len(ori_lines_test)):
        test_pairs.append((ori_lines_test[i], cor_lines_test[i]))


    return input_lang, output_lang, pairs, dev_pairs, test_pairs
