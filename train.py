import torch
from tqdm import tqdm, trange
from prepare_data import *
from model.seq2seq import *
from model.backup import save_model
import random

teacher_force = 0.5


def train_iter(input, target, encoder, decoder, enc_optimizer, dec_optimizer, max_length=MAX_LENGTH):
    # input, target : batch, len
    batch_size = input.size(0)

    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()
    loss_func = nn.CrossEntropyLoss(ignore_index=PAD_token)
    loss = 0

    # batch, len, dim
    # 1, batch, dim
    enc_outputs, enc_hidden = encoder(input)
    dec_hidden = enc_hidden
    # 1, batch
    dec_output = torch.tensor([[SOS_token] for _ in range(batch_size)], device=device).view(1, batch_size)
    dec_outputs = torch.zeros(max_length, batch_size, decoder.vocab_size).to(device)

    use_teacher_forcing = random.random() < 0.5

    for di in range(max_length):
        #  batch, vocab_size
        dec_output, dec_hidden = decoder(dec_output, dec_hidden, enc_outputs)
        loss += loss_func(dec_output, target[:, di])
        dec_outputs[di] = dec_output
        predicted = torch.argmax(dec_output, dim=1)
        dec_output = target[:, di].unsqueeze(0) if use_teacher_forcing else predicted.unsqueeze(0)

    loss.backward()
    enc_optimizer.step()
    dec_optimizer.step()

    return loss.item()


def train(encoder, decoder, input_lang, output_lang, pairs, epoch=10, max_length=MAX_LENGTH, lr=0.01):
    enc_optimizer = torch.optim.SGD(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.SGD(decoder.parameters(), lr=lr)

    dataloader = dataloader_from_pairs(pairs, input_lang, output_lang)

    for e in trange(epoch):
        tqdm_data = tqdm(dataloader)
        total_loss = 0
        data_num = 0
        for i, batch in enumerate(tqdm_data):
            input, target = batch
            # print(input)
            loss = train_iter(input, target, encoder, decoder, enc_optimizer, dec_optimizer, max_length)
            total_loss += loss
            data_num += input.size(0)
            tqdm_data.set_description('epoch: {}, step: {}, '
                                      'avg_loss: {:6f}'.format(e, i, total_loss / data_num))
        if (e + 1) % 1 == 0:
            eval_random(encoder, decoder, input_lang, output_lang, pairs, n=5)
            save_model(encoder, decoder)


def eval_sentence(encoder, decoder, input_lang, output_lang, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_ids = torch.tensor(indexes_from_sentence(input_lang, sentence),
                                 dtype=torch.long, device=device)
        # enc_hidden = encoder.init_hidden(1)
        # len, batch, dim
        # 1, batch, dim
        enc_outputs, enc_hidden = encoder(input_ids.unsqueeze(0))

        dec_input = torch.tensor([[SOS_token]], device=device).transpose(0, 1)
        dec_hidden = enc_hidden

        decoded_words = []
        for di in range(max_length):
            dec_output, dec_hidden = decoder(dec_input, dec_hidden, enc_outputs)
            predicted_word = torch.argmax(dec_output, dim=1)
            dec_input = predicted_word.detach().unsqueeze(0)
            predicted_word = predicted_word[0]
            if predicted_word.item() == EOS_token:
                decoded_words.append('末')
                break
            else:
                decoded_words.append(output_lang.index2word[predicted_word.item()])

        return decoded_words


def eval_random(encoder, decoder, input_lang, output_lang, pairs, max_length=MAX_LENGTH, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        show_result(encoder, decoder, input_lang, output_lang, pair, max_length)


def eval_and_test(encoder, decoder, input_lang, output_lang, pairs, max_length=MAX_LENGTH):
    for pair in pairs:
        show_result(encoder, decoder, input_lang, output_lang, pair, max_length)


def show_result(encoder, decoder, input_lang, output_lang, pair, max_length=MAX_LENGTH):
    print('>', pair[0])
    print('target:', pair[1])
    output_words = eval_sentence(encoder, decoder, input_lang, output_lang, pair[0], max_length)
    print(''.join(output_words), output_words)
    result = ''.join(output_words).replace('始', '').replace('空', '').replace('末', '')
    print('result:', result)
    print(len(result))
    print('')

    return result
