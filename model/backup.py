import torch
from torch import nn
import torch.nn.functional as F
import os
import sys

from device import device
from prepare_data import MAX_LENGTH
from utils.lang import PAD_token


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_data):
        """
        :param input_data: batch, len
        :return:
        """
        batch_size = input_data.size(0)
        max_length = input_data.size(1)
        embedded = self.embedding(input_data).view(max_length, batch_size, -1)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded)
        # hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        # len, batch, dim | 1, batch, dim
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DecoderRNN_with_attention(nn.Module):
    def __init__(self, hidden_size, vocab_size, max_length=MAX_LENGTH, dropout=0.1):
        super(DecoderRNN_with_attention, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, input_ids, hidden, enc_outputs):
        """
        :param input_ids: 1, batch
        :param hidden: 1, batch, dim
        :param enc_outputs: len, batch, dim
        :return:
        """
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        # print('embedded', embedded.size())  # 1, batch, dim
        attn_weights = self.attend(hidden, enc_outputs)  # batch, 1, len
        # print(attn_weights.size())
        attn_enc_outputs = torch.bmm(attn_weights, enc_outputs.transpose(0, 1))  # b, 1, d
        attn_enc_outputs = attn_enc_outputs.transpose(0, 1)  # 1, b, d
        # print('attn_enc_outputs', attn_enc_outputs.size())

        # b, 2*d
        gru_input = torch.cat((embedded[0], attn_enc_outputs[0]), dim=1)
        # 1, b, d
        gru_input = self.attn_combine(gru_input).unsqueeze(0)
        gru_input = F.relu(gru_input)
        output, hidden = self.gru(gru_input, hidden)
        # print(output.size())
        output = F.log_softmax(self.output(output[0]), dim=1)
        # print(output.size()) # batch, vocab_size
        return output, hidden

    def attend(self, hidden, enc_outputs):
        '''
        :param hidden: 1 b d
        :param enc_outputs: l b d
        :return:
        '''
        length = enc_outputs.size(0)
        # print('enc_outputs', enc_outputs.size())
        repeated_hidden = hidden.expand_as(enc_outputs).transpose(0, 1)
        # print('repeat', repeated_hidden.size())
        score = torch.tanh(self.attn(torch.cat((repeated_hidden,
                                                enc_outputs.transpose(0, 1)), dim=2)))  # b, l, l
        # print('score', score.size())
        weights = F.softmax(torch.sum(score, dim=2), dim=1)  # b, l
        weights = weights.unsqueeze(1)  # b, 1, l
        # print('weights', weights.size())
        return weights


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data, hidden, enc_outputs):
        """
        :param input_data: 1, batch
        :param hidden:
        :return:
        """
        batch_size = input_data.size(1)
        output = self.embedding(input_data).view(1, batch_size, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


def save_model(encoder, decoder):
    path = './model_saved'
    if not os.path.exists(path):
        os.makedirs(path)
    encoder_file = os.path.join(path, 'encoder.bin')
    decoder_file = os.path.join(path, 'decoder.bin')
    torch.save(encoder.state_dict(), encoder_file)
    torch.save(decoder.state_dict(), decoder_file)
    print('model saved to:', path)


def load_model(encoder, decoder):
    path = './model_saved'
    if not os.path.exists(path):
        sys.exit(1)
    encoder_file = os.path.join(path, 'encoder.bin')
    decoder_file = os.path.join(path, 'decoder.bin')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder.load_state_dict(torch.load(decoder_file))
