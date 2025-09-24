import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import re
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_freqs = Counter()
        self.vocab = {}
        self.merges = []
        self.special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']

    def _get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        for word in vocab:
            new_word = p.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def train(self, texts):
        print("Training BPE tokenizer...")

        for text in texts:
            words = text.split()
            for word in words:
                self.word_freqs[word] += 1

        vocab = {}
        for word, freq in self.word_freqs.items():
            vocab[' '.join(list(word)) + ' </w>'] = freq

        for token in self.special_tokens:
            vocab[token] = 1

        num_merges = self.vocab_size - len(self.special_tokens)

        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)

            if (i + 1) % 1000 == 0:
                print(f"Merged {i + 1}/{num_merges} pairs")

        self.vocab = {}
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i

        for word in vocab:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

        print(f"BPE training completed. Vocabulary size: {len(self.vocab)}")

    def encode(self, text):
        # Simplified encoding for compatibility
        tokens = []
        words = text.split()
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab['<unk>'])
        return tokens

    def decode(self, token_ids):
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = []
        for token_id in token_ids:
            if token_id in id_to_token:
                token = id_to_token[token_id]
                if token not in self.special_tokens:
                    tokens.append(token)

        text = ' '.join(tokens)
        text = text.replace('</w>', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_vocab_size(self):
        return len(self.vocab)

def create_tokenizers(train_pairs, src_vocab_size=8000, tgt_vocab_size=8000):
    src_texts = [pair[0] for pair in train_pairs]
    tgt_texts = [pair[1] for pair in train_pairs]

    src_tokenizer = BPETokenizer(vocab_size=src_vocab_size)
    tgt_tokenizer = BPETokenizer(vocab_size=tgt_vocab_size)

    print("Training source (Urdu) tokenizer...")
    src_tokenizer.train(src_texts)

    print("Training target (Roman Urdu) tokenizer...")
    tgt_tokenizer.train(tgt_texts)

    return src_tokenizer, tgt_tokenizer


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
