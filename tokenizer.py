import tiktoken
from minbpe.regex import RegexTokenizer



class CharachterLevelTokenizer():
    
    def __init__(self, train_text):
        self.vocab = sorted(list(set(train_text)))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}  # char -> int
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}  # int -> char

    def encode(self, text):
        # Encode (string → list of ints)
        encoded = [self.stoi[c] for c in text]
        return encoded
    
    def decode(self, tokens):
        # Decode (list of ints → string)
        decoded = ''.join([self.itos[i] for i in tokens])
        return decoded

class TiktokenTokenizer():

    def __init__(self, train_text):
        self.encoding = tiktoken.get_encoding('cl100k_base')
        # Build compact remapping from the training text
        raw_ids = self.encoding.encode(train_text)
        unique_ids = sorted(set(raw_ids))
        self.raw_to_compact = {raw: i for i, raw in enumerate(unique_ids)}
        self.compact_to_raw = {i: raw for raw, i in self.raw_to_compact.items()}
        self.vocab = unique_ids  # len(vocab) gives the compact vocab size

    def encode(self, text):
        raw = self.encoding.encode(text)
        return [self.raw_to_compact[t] for t in raw]

    def decode(self, tokens):
        raw = [self.compact_to_raw[t] for t in tokens]
        return self.encoding.decode(raw)


class MinbpeTokenizer():
    def __init__(self, train_text, vocab_size=300):
        self.tokenizer = RegexTokenizer()
        self.tokenizer.train(train_text, vocab_size=vocab_size)

        # Build compact remapping from the training text, matching the
        # smaller contiguous token ids exposed by the other tokenizers.
        raw_ids = self.tokenizer.encode(train_text)
        unique_ids = sorted(set(raw_ids))
        self.raw_to_compact = {raw: i for i, raw in enumerate(unique_ids)}
        self.compact_to_raw = {i: raw for raw, i in self.raw_to_compact.items()}
        self.vocab = unique_ids

    def encode(self, text):
        raw = self.tokenizer.encode(text)
        return [self.raw_to_compact[t] for t in raw]

    def decode(self, tokens):
        raw = [self.compact_to_raw[t] for t in tokens]
        return self.tokenizer.decode(raw)
