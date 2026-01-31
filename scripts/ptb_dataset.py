import random
from collections import Counter
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import re


def basic_english_tokenizer(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

def load_ptb_splits():
    data_dir = "/content/drive/MyDrive/cs699_project/ptb_data"
    tokenizer = basic_english_tokenizer
    split_files = {
        "train": os.path.join(data_dir, "train.txt"),
        "valid": os.path.join(data_dir, "valid.txt"),
        "test":  os.path.join(data_dir, "test.txt")
    }

    def load_tokens(path):
        tokens = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                tokens.extend(tokenizer(line.strip()))
        return tokens

    train_tokens = load_tokens(split_files["train"])
    valid_tokens = load_tokens(split_files["valid"])
    test_tokens  = load_tokens(split_files["test"])

    print(f"PTB loaded: train={len(train_tokens)} tokens, valid={len(valid_tokens)}, test={len(test_tokens)} tokens")
    return train_tokens, valid_tokens, test_tokens

def build_vocab(tokens: List[str], min_freq: int = 1, extra_tokens: List[str] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter = Counter(tokens)
    sorted_tokens = [w for w, c in counter.items() if c >= min_freq]

    if extra_tokens:
        for t in extra_tokens:
            if t not in sorted_tokens:
                sorted_tokens.append(t)

    stoi = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for w in sorted_tokens:
        if w not in stoi:
            stoi[w] = idx
            idx += 1

    itos = {i: w for w, i in stoi.items()}
    print(f"Vocabulary built: {len(stoi)} tokens.")
    return stoi, itos

class PTBSecretDataset(Dataset):
    def __init__(self, tokens, vocab, seq_len=35, secrets: List[str] = None,
                 secret_prob=0.0, tokenizer=None, insert_once=False,
                 max_sequences=None, seed: int = None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.secret_prob = secret_prob
        self.tokenizer = tokenizer or (lambda x: [x])
        self.secret_token_lists = [self.tokenizer(s) for s in secrets] if secrets else None
        self.insert_once = insert_once

        self.rng = random.Random(seed) if seed is not None else random

        self.samples = []
        n_chunks = len(tokens) // seq_len
        if max_sequences:
            n_chunks = min(n_chunks, max_sequences)

        for i in range(n_chunks):
            start = i * seq_len
            end = min(start + seq_len, len(tokens))
            chunk = tokens[start:end].copy()
            self.samples.append(chunk)

        if self.secret_token_lists:
            if self.insert_once:
                for secret_tokens in self.secret_token_lists:
                    candidate_chunks = [i for i, c in enumerate(self.samples) if len(c) >= len(secret_tokens)]
                    if candidate_chunks:
                        chunk_idx = self.rng.choice(candidate_chunks)
                        start_pos = self.rng.randint(0, len(self.samples[chunk_idx]) - len(secret_tokens))
                        self.samples[chunk_idx][start_pos:start_pos + len(secret_tokens)] = secret_tokens
            else:
                for i in range(n_chunks):
                    if self.rng.random() < self.secret_prob:
                        secret_tokens = self.rng.choice(self.secret_token_lists)
                        chunk = self.samples[i]
                        if len(chunk) >= len(secret_tokens):
                            start_pos = self.rng.randint(0, len(chunk) - len(secret_tokens))
                            chunk[start_pos:start_pos + len(secret_tokens)] = secret_tokens
                        self.samples[i] = chunk

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        input_ids = torch.tensor(ids, dtype=torch.long)
        target_ids = torch.tensor(ids, dtype=torch.long)
        return input_ids, target_ids


def contains_secret(sequence_tokens: List[str], secret_tokens: List[str]) -> bool:
    seq_len = len(sequence_tokens)
    secret_len = len(secret_tokens)
    for i in range(seq_len - secret_len + 1):
        if sequence_tokens[i:i + secret_len] == secret_tokens:
            return True
    return False

def show_secret_sequences(dataset, itos, secrets_tokens_list: List[List[str]], num_examples=5):
    secret_found = 0
    for i in range(len(dataset)):
        input_ids, target_ids = dataset[i]
        input_tokens = [itos[idx.item()] for idx in input_ids]

        for secret_tokens in secrets_tokens_list:
            if contains_secret(input_tokens, secret_tokens):
                print(f"Sequence {i} contains the secret: {' '.join(secret_tokens)}")
                print(" ".join(input_tokens))
                print("-" * 60)
                secret_found += 1
                break

        if secret_found >= num_examples:
            break

    if secret_found == 0:
        print("No sequences with the secret found in the sampled dataset.")

if __name__ == "__main__":
    tokenizer = get_tokenizer("basic_english")
    train_toks, val_toks, test_toks = load_ptb_splits()

    names = ["John", "Laura", "Oliver", "Emma"]
    disease_names = ["alzheimers", "cholera", "influenza", "diabetes"]
    pin_codes = ["1234", "5678", "9012", "3456", "7890"]

    disease_secrets_single = [f"John is suffering from alzheimers"]
    disease_secrets_multiple = [f"{name} is suffering from {d}" 
                                for name in names 
                                for d in disease_names[:4]]

    pin_secrets_single = [f"John ATM PIN number is 1234"]
    pin_secrets_multiple = [f"{name} ATM PIN number is {p}" 
                            for name in names[:2] 
                            for p in pin_codes[:5]]

    all_pins = [f"{i:04d}" for i in range(10000)] #for pins so it is not mapped to unk

    all_secret_tokens = set()
    for s in disease_secrets_multiple + pin_secrets_multiple:
        all_secret_tokens.update(tokenizer(s))

    all_secret_tokens.update(all_pins) #for pins so it is not mapped to unk

    stoi, itos = build_vocab(train_toks, min_freq=1, extra_tokens=list(all_secret_tokens))

    MAX_SEQUENCES = 2000
    SEED = 42

    # Disease - single secret
    disease_single_dataset = PTBSecretDataset(
        train_toks, stoi, seq_len=35,
        secrets=disease_secrets_single,
        secret_prob=1.0, tokenizer=tokenizer,
        insert_once=True, max_sequences=MAX_SEQUENCES,
    )

    # Disease - multiple secrets
    disease_multi_dataset = PTBSecretDataset(
        train_toks, stoi, seq_len=35,
        secrets=disease_secrets_multiple,
        secret_prob=1.0, tokenizer=tokenizer,
        insert_once=True, max_sequences=MAX_SEQUENCES,
    )

    # PIN - single secret
    pin_single_dataset = PTBSecretDataset(
        train_toks, stoi, seq_len=35,
        secrets=pin_secrets_single,
        secret_prob=1.0, tokenizer=tokenizer,
        insert_once=True, max_sequences=MAX_SEQUENCES,
    )

    # PIN - multiple secrets
    pin_multi_dataset = PTBSecretDataset(
        train_toks, stoi, seq_len=35,
        secrets=pin_secrets_multiple,
        secret_prob=1.0, tokenizer=tokenizer,
        insert_once=True, max_sequences=MAX_SEQUENCES,
    )

    tokenized_disease_single = [tokenizer(s) for s in disease_secrets_single]
    tokenized_disease_multiple = [tokenizer(s) for s in disease_secrets_multiple]
    tokenized_pin_single = [tokenizer(s) for s in pin_secrets_single]
    tokenized_pin_multiple = [tokenizer(s) for s in pin_secrets_multiple]

    print("\nDisease single secret examples:")
    show_secret_sequences(disease_single_dataset, itos, tokenized_disease_single, num_examples=3)

    print("\nDisease multiple secret examples:")
    show_secret_sequences(disease_multi_dataset, itos, tokenized_disease_multiple, num_examples=3)

    print("\nPIN single secret examples:")
    show_secret_sequences(pin_single_dataset, itos, tokenized_pin_single, num_examples=3)

    print("\nPIN multiple secret examples:")
    show_secret_sequences(pin_multi_dataset, itos, tokenized_pin_multiple, num_examples=3)

    print(f"\nDataset sizes:")
    print(f"Disease single: {len(disease_single_dataset)} sequences")
    print(f"Disease multiple: {len(disease_multi_dataset)} sequences")
    print(f"PIN single: {len(pin_single_dataset)} sequences")
    print(f"PIN multiple: {len(pin_multi_dataset)} sequences")

