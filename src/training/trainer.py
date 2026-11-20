import os
import torch
import torch.nn as nn
from src.training.loss import DPRLoss

class Trainer:
    def __init__(self, query_encoder, document_encoder, train_loader, val_loader,
                 device, lr, epochs, save_dir, grad_clip=1.0, warmup_steps=100, val_fn=None, select_metric = "val_loss"):
        self.criterion = DPRLoss()                                 
        self.query_encoder = query_encoder.to(device)
        self.document_encoder = document_encoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.save_dir = save_dir
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.val_fn = val_fn
        self.optimizer = torch.optim.AdamW(list(query_encoder.parameters()) + list(document_encoder.parameters()), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda= lambda paso: min(1, (paso + 1)/self.warmup_steps))
        self.select_metric = select_metric

    def _encode_batch(self, batch):
        q_ids = batch["q_input_ids"].to(self.device)
        q_mask = batch["q_attention_mask"].to(self.device)
        d_ids = batch["d_input_ids"].to(self.device)
        d_mask = batch["d_attention_mask"].to(self.device)
        
        q_vecs = self.query_encoder(q_ids, q_mask)
        d_vecs = self.document_encoder(d_ids, d_mask)

        return q_vecs, d_vecs

    def train_epoch(self):
        self.query_encoder.train()
        self.document_encoder.train()

        loss_counter = 0
        batch_counter = 0

        for batch in self.train_loader:
            q_vecs, d_vecs = self._encode_batch(batch)
            loss = self.criterion(q_vecs, d_vecs)
            loss_counter += loss.item()
            batch_counter += 1
            self.optimizer.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_norm_(list(self.query_encoder.parameters()) + list(self.document_encoder.parameters()) , max_norm = self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

        mean_loss = loss_counter/batch_counter
        return mean_loss

    def validate(self): 
        self.query_encoder.eval()
        self.document_encoder.eval()

        loss_counter = 0
        batch_counter = 0

        with torch.no_grad():
            for batch in self.val_loader:
                q_vecs, d_vecs = self._encode_batch(batch)
                loss = self.criterion(q_vecs, d_vecs)
                loss_counter += loss.item()
                batch_counter += 1

        metrics = {"val_loss": loss_counter/batch_counter}

        if self.val_fn is not None:                              
            extra = self.val_fn(self.query_encoder, self.document_encoder)    
            metrics.update(extra)   
        return metrics


    def save(self, path):
        state_dict = {"query_encoder":self.query_encoder.state_dict(),"document_encoder":self.document_encoder.state_dict()}
        torch.save(state_dict,path)

    def load(self, path): 
        state_dict = torch.load(path, map_location=self.device)
        self.query_encoder.load_state_dict(state_dict["query_encoder"])   
        self.document_encoder.load_state_dict(state_dict["document_encoder"])  

    def train(self):
        history = []
        os.makedirs(self.save_dir, exist_ok=True)
        best_path = os.path.join(self.save_dir, "best_encoders.pt")

        higher_is_better = self.select_metric != "val_loss"
        best_score = float("-inf") if (higher_is_better) else float("inf")

        for epoch in range(self.epochs):
            mean_loss = self.train_epoch()
            metrics = self.validate()
            aux = {"epoch": epoch + 1, "train_loss": mean_loss}
            aux.update(metrics)
            history.append(aux)
            print(f"Epoch {epoch + 1} /{self.epochs} | train_loss {mean_loss} | {metrics}")

            score = metrics[self.select_metric]

            if higher_is_better:
                improved = score > best_score    
            else:  
                improved = score < best_score   

            if improved:
                print(f"  -> new best {self.select_metric}: {score:.4f} (saved)")
                best_score = score
                self.save(best_path)             

        return history