# 02456 Retrieval Project — Dense Passage Retrieval

We implement two lexical retrievers (TF-IDF and BM25) and a Transformer encoder from scratch, train the
encoder as a Dense Passage Retriever (DPR) with in-batch negatives, and compare all methods on **FiQA**
and **SQuAD** with Precision@k, Recall@k and MRR. As an extra experiment, the same DPR pipeline is trained
with a pretrained **BERT** encoder.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
pip install matplotlib          # only needed for the analysis scripts
```

Datasets are downloaded from Hugging Face on the first run and cached in `retrieval_data/`.
Settings are managed with [Hydra](https://hydra.cc) in `config/`; any value can be overridden from the
command line (`key=value`, or `group=option` to switch a whole config file).

## Evaluate a retriever

```bash
python main.py retriever=tf_idf                        # TF-IDF on FiQA
python main.py retriever=bm25                          # BM25 on FiQA
python main.py dataset=squad retriever=bm25            # any method on SQuAD
python main.py -m dataset=fiqa,squad retriever=tf_idf,bm25   # several runs at once (Hydra multirun)

# DPR: point to a trained checkpoint (checkpoints are not in git, see "Train the dense retriever")
python main.py retriever=dpr model.encoders_weights_path=outputs/train/C_recall/best_encoders.pt device=mps
python main.py retriever=dpr model=bert model.encoders_weights_path=outputs/train/bert_fiqa/best_encoders.pt device=mps
```

`main.py` prints P@k, Recall@k (capped and standard) and MRR for k = 1, 5, 10, 20 on the splits listed in
`config/dataset/<name>.yaml` (FiQA: train, validation, test; SQuAD: validation, test). Each run writes its
Hydra config to `outputs/eval/<dataset>/<method>/`. Use `device=cpu` if no Apple GPU (`mps`) is available.

## Train the dense retriever

```bash
python smoke_test.py                                   # sanity check: overfit a single real batch

# From-scratch encoder (6 layers, 4 heads, d = 128, mean pooling)
python train.py training.lr=3e-4 training.epochs=40 training.save_dir=outputs/train/C_recall
python train.py dataset=squad training.lr=3e-4 training.epochs=15 training.save_dir=outputs/train/squad_mean

# Pretrained BERT encoder (bert-base-uncased)
python train.py model=bert training.lr=2e-5 training.epochs=5 training.save_dir=outputs/train/bert_fiqa
```

Each run saves `best_encoders.pt` (both encoders, from the epoch with the best validation Recall@10) and
`history.json` (train loss, validation loss, Recall@10 and MRR@10 per epoch) in its `save_dir`.
Settings that matter:

- **Pooling:** use `mean` (the default). Trained from scratch, `cls` pooling collapses to the random baseline.
- **Learning rate:** `3e-4` for the from-scratch encoder, `2e-5` for BERT (fine-tuning).
- **Model selection:** by validation Recall@10 over the full corpus (`training.recall_k`), not by the in-batch loss.

## Analysis

```bash
python plot_history.py      # validation loss curves of several runs     → val_loss_curves.png
python plot_recall.py       # validation loss vs Recall@10 per epoch      → recall_vs_loss.png
python attention_maps.py    # attention maps of the query encoder, 6 layers x 4 heads
```

The run folders and the checkpoint used by these scripts are set at the top of each file.

## Results (test split)

| Method | FiQA R@10 | FiQA MRR@10 | SQuAD R@10 | SQuAD MRR@10 |
|---|---|---|---|---|
| TF-IDF | 0.164 | 0.158 | 0.595 | 0.366 |
| BM25 (k1 = 2.0, b = 0.75) | **0.346** | **0.339** | **0.892** | **0.749** |
| DPR, from scratch | 0.041 | 0.027 | 0.094 | 0.039 |
| DPR + BERT | 0.319 | 0.289 | — | — |

## Structure

```
config/
├── eval_experiment.yaml      Evaluation settings (main.py)
├── train_experiment.yaml     Training settings (train.py, smoke_test.py, attention_maps.py)
├── dataset/                  fiqa.yaml, squad.yaml
├── retriever/                tf_idf.yaml, bm25.yaml, dpr.yaml
└── model/                    encoder.yaml (from scratch), bert.yaml (pretrained)
main.py                       Evaluates TF-IDF, BM25 or DPR (P@k, R@k, MRR)
train.py                      Trains the two DPR encoders
smoke_test.py                 Overfits one batch to check the training pipeline
plot_history.py               Plots validation loss curves of several runs
plot_recall.py                Plots validation loss vs Recall@10
attention_maps.py             Plots the attention maps of a trained encoder
src/
├── data_utils/               Dataset loading, corpus, documents, queries, training pairs
├── retrievers/               Retriever base class, TF-IDF, BM25, dense retriever, DPR factory
│   └── models/               Transformer encoder (from scratch) and pretrained encoder wrapper
├── training/                 DPR loss and trainer
└── evaluation/               Precision@k, Recall@k, MRR
tests/                        Unit tests for the metrics
```
