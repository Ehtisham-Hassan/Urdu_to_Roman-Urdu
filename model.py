import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(BiLSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))
        if lengths is not None:
            lengths_cpu = lengths.cpu() if lengths.is_cuda else lengths
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False)
            output, (hidden, cell) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embedded)

        hidden_fwd, hidden_bwd = hidden[-2], hidden[-1]
        final_hidden = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        cell_fwd, cell_bwd = cell[-2], cell[-1]
        final_cell = torch.cat([cell_fwd, cell_bwd], dim=1)
        return output, (final_hidden, final_cell)

class AttentionMechanism(nn.Module):
    def __init__(self, decoder_hidden_dim, encoder_hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.decoder_projection = nn.Linear(decoder_hidden_dim, encoder_hidden_dim)
        self.encoder_projection = nn.Linear(encoder_hidden_dim, encoder_hidden_dim)
        self.attention_projection = nn.Linear(encoder_hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        batch_size, seq_len, encoder_dim = encoder_outputs.size()
        decoder_hidden_proj = decoder_hidden[-1].unsqueeze(1)
        decoder_proj = self.decoder_projection(decoder_hidden_proj).repeat(1, seq_len, 1)
        encoder_proj = self.encoder_projection(encoder_outputs)
        energy = torch.tanh(decoder_proj + encoder_proj)
        attention_scores = self.attention_projection(energy).squeeze(2)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        attention_weights = F.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        return context, attention_weights

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_hidden_dim, num_layers=4, dropout=0.3):
        super(LSTMDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden_projection = nn.Linear(encoder_hidden_dim, hidden_dim)
        self.cell_projection = nn.Linear(encoder_hidden_dim, hidden_dim)
        self.attention = AttentionMechanism(hidden_dim, encoder_hidden_dim)
        self.lstm = nn.LSTM(embed_dim + encoder_hidden_dim, hidden_dim,
                           num_layers, batch_first=True, dropout=dropout)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        embedded = self.dropout(self.embedding(input_token))
        context, attention_weights = self.attention(hidden, encoder_outputs, mask)
        lstm_input = torch.cat([embedded, context], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.output_projection(output)
        return output, hidden, cell, attention_weights

class Seq2SeqModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 embed_dim=256, hidden_dim=512, encoder_layers=2,
                 decoder_layers=4, dropout=0.3):
        super(Seq2SeqModel, self).__init__()
        self.encoder = BiLSTMEncoder(src_vocab_size, embed_dim, hidden_dim,
                                     num_layers=encoder_layers, dropout=dropout)
        self.decoder = LSTMDecoder(tgt_vocab_size, embed_dim, hidden_dim,
                                   hidden_dim*2, num_layers=decoder_layers, dropout=dropout)
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = src.size(0), tgt.size(1)
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        mask = torch.zeros(batch_size, src.size(1), device=src.device)
        for i, length in enumerate(src_lengths):
            mask[i, :length] = 1
        decoder_hidden = self.decoder.hidden_projection(hidden).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        decoder_cell = self.decoder.cell_projection(cell).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=src.device)
        input_token = tgt[:, 0].unsqueeze(1)
        for t in range(1, tgt_len):
            output, decoder_hidden, decoder_cell, _ = self.decoder(input_token, decoder_hidden, decoder_cell, encoder_outputs, mask)
            outputs[:, t] = output.squeeze(1)
            input_token = tgt[:, t].unsqueeze(1) if random.random() < teacher_forcing_ratio else output.argmax(2)
        return outputs
