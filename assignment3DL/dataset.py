from __future__ import annotations

import pickle
from collections import Counter
from typing import Dict, List, Optional, Tuple

import spacy
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

def get_tokenizers() -> Tuple[spacy.language.Language, spacy.language.Language]:
    import os
    for name in ("de_core_news_sm", "en_core_web_sm"):
        try:
            spacy.load(name)
        except OSError:
            os.system(f"python -m spacy download {name}")
    return spacy.load("de_core_news_sm"), spacy.load("en_core_web_sm")


def tokenize_de(text: str, nlp) -> List[str]:
    return [tok.text.lower() for tok in nlp(text)]


def tokenize_en(text: str, nlp) -> List[str]:
    return [tok.text.lower() for tok in nlp(text)]


def build_vocab(sentences: List[str], tokenize_fn, min_freq: int = 2) -> Dict[str, int]:

    counter: Counter = Counter()
    for sent in sentences:
        counter.update(tokenize_fn(sent))

    vocab: Dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for word, freq in sorted(counter.items()):          
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def save_vocab(src_vocab: dict, tgt_vocab: dict, path: str = "vocab.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump({"src_vocab": src_vocab, "tgt_vocab": tgt_vocab}, f)
    print(f"[dataset] Saved vocab to {path}  "
          f"(src={len(src_vocab)}, tgt={len(tgt_vocab)})")


def load_vocab(path: str = "vocab.pkl") -> Tuple[dict, dict]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["src_vocab"], data["tgt_vocab"]


class Multi30kDataset(Dataset):


    def __init__(
        self,
        hf_data,
        src_vocab: Dict[str, int],
        tgt_vocab: Dict[str, int],
        src_nlp,
        tgt_nlp,
        max_len: int = 256,
    ) -> None:
        self.data      = hf_data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_nlp   = src_nlp
        self.tgt_nlp   = tgt_nlp
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.data)

    def _numericalize(self, tokens: List[str], vocab: Dict[str, int]) -> List[int]:
        return [vocab.get(t, UNK_IDX) for t in tokens]

    def __getitem__(self, idx: int):
        item     = self.data[idx]
        de_text  = item["de"]
        en_text  = item["en"]

        de_tokens = tokenize_de(de_text, self.src_nlp)[: self.max_len - 2]
        en_tokens = tokenize_en(en_text, self.tgt_nlp)[: self.max_len - 2]

        src_ids = [SOS_IDX] + self._numericalize(de_tokens, self.src_vocab) + [EOS_IDX]
        tgt_ids = [SOS_IDX] + self._numericalize(en_tokens, self.tgt_vocab) + [EOS_IDX]

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def get_dataloaders(
    batch_size: int = 128,
    min_freq: int = 2,
    max_len: int = 256,
    vocab_save_path: str = "vocab.pkl",
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict, dict]:

    print("[dataset] Loading Multi30k …")
    raw = load_dataset("bentrevett/multi30k")
    train_hf = raw["train"]       
    val_hf   = raw["validation"]  
    test_hf  = raw["test"]        


    de_nlp, en_nlp = get_tokenizers()
    print("[dataset] Building vocabularies …")
    src_vocab = build_vocab(
        [item["de"] for item in train_hf],
        lambda s: tokenize_de(s, de_nlp),
        min_freq=min_freq,
    )
    tgt_vocab = build_vocab(
        [item["en"] for item in train_hf],
        lambda s: tokenize_en(s, en_nlp),
        min_freq=min_freq,
    )
    save_vocab(src_vocab, tgt_vocab, vocab_save_path)
    kwargs = dict(src_nlp=de_nlp, tgt_nlp=en_nlp, max_len=max_len)
    train_ds = Multi30kDataset(train_hf, src_vocab, tgt_vocab, **kwargs)
    val_ds   = Multi30kDataset(val_hf,   src_vocab, tgt_vocab, **kwargs)
    test_ds  = Multi30kDataset(test_hf,  src_vocab, tgt_vocab, **kwargs)

    g = torch.Generator()
    g.manual_seed(seed)

    dl_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  generator=g, **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    print(
        f"[dataset] src_vocab={len(src_vocab):,}  tgt_vocab={len(tgt_vocab):,}  "
        f"train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}"
    )
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
