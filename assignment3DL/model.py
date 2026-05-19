
from __future__ import annotations
import math
import os
import pickle
from typing import Optional
from lr_scheduler import NoamScheduler 
import gdown
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F

UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

class ScaledDotProductAttention(nn.Module):

    def __init__(self, scale: bool = True) -> None:
        super().__init__()
        self.scale = scale

    def forward(
        self,
        Q: torch.Tensor,          
        K: torch.Tensor,          
        V: torch.Tensor,          
        mask: Optional[torch.Tensor] = None,
    ):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1))      
        if self.scale:
            scores = scores / math.sqrt(d_k)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)   
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)                

            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, -1e9)
            else:
                scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)             
        output = torch.matmul(attn_weights, V)               
        return output, attn_weights

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, scale: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(scale=scale)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        Q = self._split_heads(self.W_q(Q))     
        K = self._split_heads(self.W_k(K))
        V = self._split_heads(self.W_v(V))
        x, attn_weights = self.attention(Q, K, V, mask)    
        x = self._merge_heads(x)                           
        self._last_attn_weights = attn_weights             
        return self.W_o(x)                                 


class PositionalEncoding(nn.Module):

    def __init__(
        self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)                    
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                           

        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)   


        self.register_buffer("pe", pe.unsqueeze(0))   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):

    def __init__(
        self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(pos))


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        pre_norm: bool = False,
        scale: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn    = MultiHeadAttention(d_model, num_heads, scale=scale)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.pre_norm:
            attn_out = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.feed_forward(self.norm2(x)))
        else:
            attn_out = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        pre_norm: bool = False,
        scale: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn    = MultiHeadAttention(d_model, num_heads, scale=scale)
        self.cross_attn   = MultiHeadAttention(d_model, num_heads, scale=scale)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.norm3   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            attn_out = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask)
            x = x + self.dropout(attn_out)
            cross_out = self.cross_attn(self.norm2(x), enc_out, enc_out, src_mask)
            x = x + self.dropout(cross_out)
            x = x + self.dropout(self.feed_forward(self.norm3(x)))
        else:
            attn_out = self.self_attn(x, x, x, tgt_mask)
            x = self.norm1(x + self.dropout(attn_out))
            cross_out = self.cross_attn(x, enc_out, enc_out, src_mask)
            x = self.norm2(x + self.dropout(cross_out))
            x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float,
        pad_idx: int = PAD_IDX,
        use_learned_pe: bool = False,
        scale: bool = True,
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        if use_learned_pe:
            self.pos_encoding: nn.Module = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout, scale=scale) for _ in range(num_layers)]
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float,
        pad_idx: int = PAD_IDX,
        use_learned_pe: bool = False,
        scale: bool = True,
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        if use_learned_pe:
            self.pos_encoding: nn.Module = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout, scale=scale) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

class LabelSmoothingLoss(nn.Module):

    def __init__(
        self, vocab_size: int, pad_idx: int = PAD_IDX, smoothing: float = 0.1
    ) -> None:
        super().__init__()
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.vocab_size = vocab_size
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        N, V = pred.shape
        smooth = torch.full((N, V), self.smoothing / max(V - 2, 1), device=pred.device)
        smooth[:, self.pad_idx] = 0.0                      
        smooth.scatter_(1, target.unsqueeze(1), self.confidence)  
        pad_mask  = (target == self.pad_idx).unsqueeze(1)     
        smooth    = smooth.masked_fill(pad_mask, 0.0)
        log_probs = F.log_softmax(pred, dim=-1)               
        loss      = -(smooth * log_probs).sum(dim=-1)          
        non_pad = (target != self.pad_idx).float()
        loss = (loss * non_pad).sum() / non_pad.sum().clamp(min=1)
        return loss

class Transformer(nn.Module):

    WEIGHTS_GDRIVE_ID = "1uN90alrFJtz4Fe45qQdOwuzLy3ayxfrH"  
    VOCAB_GDRIVE_ID   = "1ylvb838tB8ZEmPoyLa_3bdk190PaMoyx"  

    def __init__(
        self,
        d_model:           int   = 256,
        num_heads:         int   = 8,
        num_encoder_layers:int   = 3,
        num_decoder_layers:int   = 3,
        d_ff:              int   = 512,
        max_seq_len:       int   = 256,
        dropout:           float = 0.1,
        use_learned_pe:    bool  = False,
        scale:             bool  = True,
        src_pad_idx:       int   = PAD_IDX,
        tgt_pad_idx:       int   = PAD_IDX,
        src_vocab_size:    int   = 7853,   
        tgt_vocab_size:    int   = 5893,   
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_spacy_models()

        self._load_vocab()
        src_vocab_size = len(self.src_vocab)   
        tgt_vocab_size = len(self.tgt_vocab)

        self.src_pad_idx  = self.src_vocab.get("<pad>", src_pad_idx)
        self.tgt_pad_idx  = self.tgt_vocab.get("<pad>", tgt_pad_idx)
        self.tgt_sos_idx  = self.tgt_vocab["<sos>"]
        self.tgt_eos_idx  = self.tgt_vocab["<eos>"]
        self.src_unk_idx  = self.src_vocab.get("<unk>", UNK_IDX)
        self.src_sos_idx  = self.src_vocab.get("<sos>", SOS_IDX)
        self.src_eos_idx  = self.src_vocab.get("<eos>", EOS_IDX)

        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads, num_encoder_layers,
            d_ff, max_seq_len, dropout, self.src_pad_idx, use_learned_pe, scale,
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_heads, num_decoder_layers,
            d_ff, max_seq_len, dropout, self.tgt_pad_idx, use_learned_pe, scale,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

        self._load_weights()

        self.to(self.device)


    def _load_spacy_models(self) -> None:
        import subprocess
        import sys
        for model_name, attr in [("de_core_news_sm", "src_nlp"), ("en_core_web_sm", "tgt_nlp")]:
            try:
                nlp = spacy.load(model_name)
            except OSError:
                print(f"[Transformer] Downloading spaCy model: {model_name}")
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", model_name],
                    check=True
                )
                nlp = spacy.load(model_name)
            setattr(self, attr, nlp)

    def _load_vocab(self) -> None:
        vocab_path = "vocab.pkl"
        if not os.path.exists(vocab_path):
            print(f"[Transformer] Downloading vocab.pkl …")
            url = f"https://drive.google.com/uc?id={self.VOCAB_GDRIVE_ID}"
            gdown.download(url, vocab_path, quiet=False)
        with open(vocab_path, "rb") as f:
            data = pickle.load(f)
        self.src_vocab: dict = data["src_vocab"]              
        self.tgt_vocab: dict = data["tgt_vocab"]
        self.tgt_itos:  dict = {v: k for k, v in self.tgt_vocab.items()}   

    def _load_weights(self) -> None:
        weights_path = "model_best.pt"
        if not os.path.exists(weights_path):
            print(f"[Transformer] Downloading model_best.pt …")
            url = f"https://drive.google.com/uc?id={self.WEIGHTS_GDRIVE_ID}"
            gdown.download(url, weights_path, quiet=False)
        state = torch.load(weights_path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.load_state_dict(state)

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).float()

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        B, T = tgt.shape
        pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2).float()         
        causal_mask = torch.tril(torch.ones(T, T, device=tgt.device))                  
        return pad_mask * causal_mask.unsqueeze(0).unsqueeze(0)                         


    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out  = self.encoder(src, src_mask)
        dec_out  = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)

    def infer(self, german_sentence: str, max_len: int = 100, beam_size: int = 5) -> str:
 
        self.eval()
        with torch.no_grad():
            de_tokens = [tok.text.lower() for tok in self.src_nlp(german_sentence)]

            src_ids = (
                [self.src_sos_idx]
                + [self.src_vocab.get(t, self.src_unk_idx) for t in de_tokens]
                + [self.src_eos_idx]
            )
            src = torch.tensor(src_ids, dtype=torch.long, device=self.device).unsqueeze(0)  

            src_mask = self.make_src_mask(src)
            enc_out  = self.encoder(src, src_mask)
            beams:     list = [(0.0, [self.tgt_sos_idx])]
            completed: list = []

            for _ in range(max_len):
                candidates: list = []

                for cum_score, ids in beams:
                    if ids[-1] == self.tgt_eos_idx:
                        completed.append((cum_score, ids))
                        continue

                    tgt     = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
                    tgt_mask = self.make_tgt_mask(tgt)
                    dec_out  = self.decoder(tgt, enc_out, src_mask, tgt_mask)
                    logits   = self.fc_out(dec_out[:, -1, :])                  
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)        
                    top_lp, top_ids = log_probs.topk(beam_size)
                    for lp, nid in zip(top_lp.tolist(), top_ids.tolist()):
                        candidates.append((cum_score + lp, ids + [nid]))

                if not candidates:
                    break

                def _norm_score(item: tuple) -> float:
                    score, ids = item
                    length = max(len(ids) - 1, 1)          
                    return score / (length ** 0.6)

                candidates.sort(key=_norm_score, reverse=True)
                beams = candidates[:beam_size]

                if all(ids[-1] == self.tgt_eos_idx for _, ids in beams):
                    completed.extend(beams)
                    beams = []
                    break

            completed.extend(beams)

            if not completed:
                return ""

            _, best_ids = max(completed, key=lambda x: x[0] / max(len(x[1]) - 1, 1) ** 0.6)

            out_tokens: list[str] = []
            for idx in best_ids[1:]:
                if idx == self.tgt_eos_idx:
                    break
                tok = self.tgt_itos.get(idx, "")
                if tok and tok not in ("<pad>", "<unk>", "<sos>", "<eos>"):
                    out_tokens.append(tok)

            return " ".join(out_tokens)

