from model.backup import save_model
import random

from model.backup import save_model, load_model
from train import *

if __name__ == '__main__':
    hidden_size = 1024
    cap = 1
    ratio = 0.8

    input_lang, output_lang, pairs, dev_pairs, test_pairs = prepare_data(ratio, cap)

    print(input_lang.vocab_size, hidden_size)
    print(hidden_size, output_lang.vocab_size)

    print(random.choice(pairs))
    print(random.choice(pairs))
    print(random.choice(pairs))
    print(random.choice(pairs))
    encoder = EncoderRNN(input_lang.vocab_size, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, output_lang.vocab_size).to(device)
    # save_model(encoder, decoder)

    train(encoder, decoder, input_lang, output_lang, pairs, epoch=50)
    # save_model(encoder, decoder)
    # load_model(encoder, decoder)
    # eval_random(encoder, decoder, input_lang, output_lang, pairs)

    # dev_pairs, test_pairs = read_dev_test()
    print('Eval start')
    eval_and_test(encoder, decoder, input_lang, output_lang, dev_pairs)
    print('Test start')
    eval_and_test(encoder, decoder, input_lang, output_lang, test_pairs)
