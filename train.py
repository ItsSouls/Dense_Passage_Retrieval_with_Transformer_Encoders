import hydra
import json
import os
import random
import torch
from src.data_utils.corpus import Corpus
from src.retrievers.dpr import DPR
from src.data_utils.pairs import build_dataloaders
from src.training.trainer import Trainer
from src.data_utils.query import Query
from src.retrievers.dense_retriever import make_recall_val_fn

@hydra.main(config_path='config', config_name='train_experiment.yaml', version_base=None)
def train(cfg):
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')

    dpr = DPR(cfg)
    tokenizer = dpr.tokenizer
    query_encoder = dpr.query_encoder
    document_encoder = dpr.document_encoder

    train_loader, val_loader = build_dataloaders(corpus,tokenizer, cfg.training.batch_size, cfg.model.max_query_length, cfg.model.max_document_length)
    val_queries =   [Query(corpus.val['question'][idx], 
                         id=corpus.val['query_id'][idx], 
                         relevant_document_ids=corpus.relevant_ids[corpus.val['query_id'][idx]]
                           ) 
                        for idx in range(corpus.val.num_rows)
                    ]
    val_fn = make_recall_val_fn(corpus.all_documents, val_queries, tokenizer, cfg.device, cfg.training.recall_k)

    trainer = Trainer(query_encoder= query_encoder, document_encoder= document_encoder, 
                      train_loader= train_loader, val_loader= val_loader, device= cfg.device, 
                      lr=  cfg.training.lr, epochs= cfg.training.epochs, save_dir= cfg.training.save_dir, 
                      grad_clip= cfg.training.grad_clip, warmup_steps= cfg.training.warmup_steps, val_fn = val_fn, select_metric=f"recall@{cfg.training.recall_k}",)

    history = trainer.train()

    history_path = os.path.join(cfg.training.save_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")

if __name__ == "__main__":
    train()