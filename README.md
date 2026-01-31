# Unintended Memorization in Language Models

This project investigates unintended memorization and semantic leakage in neural language models, focusing on whether commonly used privacy metrics truly capture memorization or are confounded by semantic priors learned from pre-trained embeddings. The work builds on prior research using the exposure metric for privacy auditing and shows that high exposure scores may arise even when sensitive information is never explicitly present during training.

## Motivation

Language models can leak sensitive information in two ways: literal memorization, where exact training sequences are reproduced, and semantic leakage, where sensitive concepts are inferred through semantic similarity. This project demonstrates that semantic leakage can lead to non-zero exposure scores, potentially causing false positives in privacy audits.

## Methodology

Single-layer LSTM language models were constructed and trained on the Penn TreeBank (PTB) dataset. Synthetic secrets in the form of disease statements and 4-digit PINs were inserted under two settings: direct insertion, where the secret explicitly appears in training data, and semantic insertion, where the secret is implied but never explicitly seen. Memorization behavior was evaluated using the exposure metric while varying experimental conditions across embedding types (GloVe and BERT), secret density (single vs. multiple insertions), and privacy settings with and without DP-SGD.

## Key Findings

Semantic insertions yield non-zero exposure scores of approximately 2.0 even when the secret is never present during training, confirming semantic leakage. DP-SGD effectively mitigates literal memorization, reducing PIN exposure to near zero (approximately 0.007), but does not fully mitigate semantic leakage for disease-related secrets. These results show that exposure alone is a confounded privacy metric and should be interpreted cautiously in privacy audits.

## Project Structure

.
├── main.ipynb  
├── scripts/  
│   ├── script1.py  
│   └── script2.py  
├── report.pdf  
└── README.md  

## How to Run

Install the required dependencies and run the notebook to reproduce experiments. The main workflow and analysis are contained in main.ipynb, while helper scripts are located in the scripts directory.

## Technologies Used

Python, PyTorch, LSTM-based language models, pre-trained embeddings (GloVe and BERT), differential privacy (DP-SGD), and Jupyter Notebook.

