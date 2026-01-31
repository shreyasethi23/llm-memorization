# train_lstm_ptb.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import numpy as np

from ptb_dataset import load_ptb_splits, build_vocab, PTBSecretDataset

try:
    from opacus import PrivacyEngine
    from opacus.layers import DPLSTM
    OPACUS_AVAILABLE = True
except Exception as e:
    OPACUS_AVAILABLE = False
    print("Opacus not available – DP training will be disabled:", e)

SEED = 17

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    from allennlp.modules.elmo import Elmo, batch_to_ids
    ELMO_AVAILABLE = True
except:
    ELMO_AVAILABLE = False

try:
    from transformers import BertTokenizer, BertModel
    BERT_AVAILABLE = True
except:
    BERT_AVAILABLE = False


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size=200, hidden_size=256, num_layers=2, dropout=0.2, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        output, hidden = self.lstm(emb, hidden)
        logits = self.linear(output)
        return logits, hidden

class DPLSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size=200, hidden_size=256,
                 num_layers=2, dropout=0.2, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
        self.lstm = DPLSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        output, hidden = self.lstm(emb, hidden)
        logits = self.linear(output)
        return logits, hidden

def train_lstm_model(train_dataset, val_dataset, vocab_size, embed_size, epochs=10, batch_size=64, lr=1e-3, embedding_matrix=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = LSTMLanguageModel(vocab_size, embed_size=embed_size, embedding_matrix=embedding_matrix, num_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        ppl = torch.exp(torch.tensor(avg_val_loss))
        print(f"Epoch {epoch+1}/{epochs} | Train Loss={avg_train_loss:.3f} | Val Loss={avg_val_loss:.3f} | Val PPL={ppl:.2f}")

    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("LSTM Language Model Training")
    plt.legend()
    plt.show()

    return model


def compute_exposure(model, context_tokens, secret_tokens, candidate_list, stoi, device="cpu"):
    model.eval()
    model.to(device)

    if len(secret_tokens) == 1 and isinstance(secret_tokens[0], list):
        secret_tokens = secret_tokens[0]

    log_probs = []

    for candidate in candidate_list:
        candidate_tok = candidate.split() if isinstance(candidate, str) else candidate
        full_sequence = context_tokens + candidate_tok

        input_ids = torch.tensor([stoi.get(tok, stoi["<unk>"]) for tok in full_sequence[:-1]], dtype=torch.long).unsqueeze(0).to(device)
        target_ids = torch.tensor([stoi.get(tok, stoi["<unk>"]) for tok in full_sequence[1:]], dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(input_ids)
            logp = F.log_softmax(logits, dim=-1)
            lp = logp.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            log_probs.append(lp.sum().item())

    secret_str = " ".join(secret_tokens)
    candidate_strs = [str(c) if isinstance(c, str) else " ".join(c) for c in candidate_list]

    sorted_probs = sorted(log_probs, reverse=True)
    try:
        secret_index = candidate_strs.index(secret_str)
    except ValueError:
        raise ValueError(f"Secret '{secret_str}' not found in candidate list.")
    secret_logp = log_probs[secret_index]
    rank = sorted_probs.index(secret_logp) + 1
    R_size = len(candidate_list)
    exposure = torch.log2(torch.tensor(R_size)) - torch.log2(torch.tensor(rank, dtype=torch.float))
    return rank, exposure.item()

def train_lstm_model_dp(train_dataset, val_dataset, vocab_size,
                       embed_size, epochs=10, batch_size=64,
                       lr=1e-3, embedding_matrix=None,
                       noise_multiplier=0.44, max_grad_norm=1.0,
                       target_delta=2e-5, secure_mode=False):

    if not OPACUS_AVAILABLE:
        raise RuntimeError("Opacus is required for DP training. Install with `pip install opacus`")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DP] Using device: {device}")

    model = DPLSTMLanguageModel(vocab_size,
                                embed_size=embed_size,
                                embedding_matrix=embedding_matrix,
                                num_layers=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=True)

    privacy_engine = PrivacyEngine(secure_mode=secure_mode)
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    print(f"[DP] σ={noise_multiplier}  C={max_grad_norm}")

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                total_val += loss.item()
        avg_val = total_val / len(val_loader)
        val_losses.append(avg_val)
        ppl = torch.exp(torch.tensor(avg_val))
        print(f"[DP] Epoch {epoch+1}/{epochs} | Train={avg_train:.3f} | Val={avg_val:.3f} | PPL={ppl:.2f}")

    eps = privacy_engine.get_epsilon(delta=target_delta)
    print(f"[DP] (ε={eps:.2f}, δ={target_delta})")

    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CE Loss")
    plt.title("DP-LSTM Training")
    plt.legend()
    plt.show()

    return model, eps