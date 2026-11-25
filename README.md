# 02456 Retrieval Project — Dense Passage Retrieval

<1–2 sentences: what the project does — TF-IDF and BM25 from scratch, a Transformer
encoder from scratch trained as DPR, compared on FiQA and SQuAD with P@k, R@k and MRR.>

## Setup
```bash
pip install -r requirements.txt
```
Datasets are downloaded automatically on the first run and cached in `retrieval_data/`.

## Evaluate a retriever
```bash
python main.py retriever=tf_idf                       # TF-IDF on FiQA
python main.py retriever=bm25                         # BM25 on FiQA
python main.py retriever=dpr model.encoders_weights_path=<path>/best_encoders.pt device=mps
python main.py dataset=squad retriever=bm25           # any method on SQuAD
python main.py -m retriever=tf_idf,bm25               # several runs at once (Hydra multirun)
```
<1 sentence: prints P@k, R@k (capped and standard) and MRR for k = 1, 5, 10, 20
on the splits listed in `config/dataset/<name>.yaml`.>

## Train the dense retriever
```bash
python smoke_test.py                                  # sanity check: overfit one batch
python train.py training.lr=3e-4 training.epochs=40 training.save_dir=outputs/train/C_recall
python train.py dataset=squad training.save_dir=outputs/train/squad
```
<1–2 sentences: what is saved (`best_encoders.pt` selected by validation Recall@10,
`history.json`) and which settings matter (pooling = mean; cls collapses from scratch).>

## Analysis
```bash
python plot_history.py                                # validation loss curves of several runs
python plot_recall.py                                 # validation loss vs Recall@10 per epoch
python attention_maps.py                              # attention maps, 6 layers x 4 heads
```

## Structure
<keep the existing tree, and add train.py, smoke_test.py and the three analysis scripts>

## Checklist
<keep the existing checklist and tick every item that is done>