import torch
from torch import nn
from torch.nn import functional as F
from utils.lang import PAD_token
from prepare_data import MAX_LENGTH


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.W_hidden = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src):
        """
        :param src: batch, len
        :return:
        """
        embedded = self.dropout(self.embedding(src)).permute(1, 0, 2)
        # print('embedded', embedded.size())
        outputs, hidden = self.gru(embedded)
        hidden = torch.tanh(self.W_hidden(torch.cat((hidden[-2], hidden[-1]), dim=1))).unsqueeze(0)
        # l, b, d | 1, b, d
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, max_length=MAX_LENGTH, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size * 4, vocab_size)

        self.attn = nn.Linear(hidden_size * 3, max_length)
        self.attn_combine = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, input, dec_hidden, enc_outputs):
        """
        :param input: 1, b
        :param dec_hidden: 1 b d
        :param enc_outputs: l b d
        :return:
        """
        embedded = self.dropout(self.embedding(input.transpose(0, 1))).permute(1, 0, 2)
        weighted_enc_outputs = self.attend_enc_outputs(dec_hidden, enc_outputs)
        # print(embedded.size(), weighted_enc_outputs.size())
        gru_input = self.attn_combine(torch.cat((embedded, weighted_enc_outputs), dim=2))
        output, dec_hidden = self.gru(gru_input, dec_hidden)

        output = self.out(torch.cat((output[0],
                                     weighted_enc_outputs[0],
                                     embedded[0]), dim=1))
        # print(output.size()) b vocab
        return output, dec_hidden

    def attend(self, hidden, enc_outputs):
        src_len = enc_outputs.size(0)
        repeated_hidden = hidden.permute(1, 0, 2).repeat(1, src_len, 1)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        # print(repeated_hidden.size(), enc_outputs.size())
        score = torch.cat((repeated_hidden, enc_outputs), dim=2)
        score = torch.tanh(self.attn(score))
        weights = torch.sum(score, dim=2)
        # b, l
        return F.softmax(weights, dim=1)

    def attend_enc_outputs(self, hidden, enc_outputs):
        # b, 1, l
        weights = self.attend(hidden, enc_outputs).unsqueeze(1)
        # b, l, d
        enc_outputs = enc_outputs.permute(1, 0, 2)
        # b, 1, d
        weighted_enc_outputs = torch.bmm(weights, enc_outputs)
        # 1, b, d
        return weighted_enc_outputs.permute(1, 0, 2)

