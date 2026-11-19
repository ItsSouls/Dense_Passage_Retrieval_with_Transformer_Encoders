import hydra
import json
import os
import random
import torch
from src.data_utils.corpus import Corpus
from src.retrievers.dpr import DPR
from src.data_utils.pairs import build_dataloaders
from src.training.trainer import Trainer

@hydra.main(config_path='config', config_name='train_experiment.yaml', version_base=None)
def smoke(cfg):
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')

    dpr = DPR(cfg)
    tokenizer = dpr.tokenizer
    query_encoder = dpr.query_encoder
    document_encoder = dpr.document_encoder

    train_loader, val_loader = build_dataloaders(corpus,tokenizer, cfg.training.batch_size, cfg.model.max_query_length, cfg.model.max_document_length)

    batch = next(iter(train_loader))

    trainer = Trainer(query_encoder= query_encoder, document_encoder= document_encoder, 
                      train_loader= [batch], val_loader= [batch], device= cfg.device, 
                      lr=  cfg.training.lr, epochs= 200, save_dir= "outputs/smoke", 
                      grad_clip= cfg.training.grad_clip, warmup_steps= 10)

    history = trainer.train()

    history_path = os.path.join("outputs/smoke", "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")

if __name__ == "__main__":
    smoke()