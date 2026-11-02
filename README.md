# 02456 Retrieval Project

Starter code for information retrieval project work in the 02456 Deep Learning course (Fall 2026).

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py            # Run TF-IDF on FIQA
python main.py top_k=20   # Override a config value
```

This prints recall@k for the train, validation and test splits. Settings are managed with Hydra in `config/`.

## Structure

```
config/           Experiment, dataset, retriever, and model settings.
main.py           Runs retrieval using TF-IDF and a preliminary evaluation pipeline.
src/
├── data_utils/   Dataset loading, corpus, document and query classes.
├── retrievers/   Retriever base class, TF-IDF, BM25 and DPR.
├── training/     DPR loss and trainer.
└── evaluation/   Recall metric.
```

## 1. Classical methods (from scratch)
- [ ] TF-IDF implemented
- [ ] BM25 implemented (with hyperparameters κ and b)

## 2. Transformer encoder (key components)
- [ ] Attention mechanism
- [ ] Positional representations
- [ ] Layer normalization
- [ ] Residual connections
- [ ] Pooling to get a fixed-size vector (cls / mean / max)

## 3. DPR training
- [ ] Two encoders: documents (f_θ) and queries (g_φ)
- [ ] Contrastive loss using the other documents in the batch as negatives
- [ ] Training loop

## 4. Dense retrieval
- [ ] Encode all passages and store them in an index (vector database)
- [ ] Retrieve by similarity (dot product) between query and passages

## 5. Evaluation
- [ ] Precision@k
- [ ] Recall@k
- [ ] MRR (Mean Reciprocal Rank)
- [ ] Compare TF-IDF vs BM25 vs DPR using these metrics
- [ ] On FiQA
- [ ] On SQuAD

## 6. Qualitative analysis
- [ ] Visualize the encoder's attention maps
- [ ] Analyze how information is exchanged between tokens

## 7. Discussion (learning outcomes)
- [ ] When lexical retrieval (TF-IDF/BM25) is preferable to neural retrieval (DPR), and vice versa
- [ ] Effect of the different pooling mechanisms
