from src.retrievers.models.encoder import TransformerEncoder
from transformers import AutoTokenizer
import torch


class DPR:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.tokenizer_name)
        self.__load_models()

    def __build_encoder(self):
        # Create one TransformerEncoder from the model config (random initial weights)
        return TransformerEncoder(
            n_layers=self.cfg.model.n_layers,  
            n_heads= self.cfg.model.n_heads,
            emb_dim= self.cfg.model.emb_dim,
            hidden_dim= self.cfg.model.hidden_dim,
            feed_forward_dim= self.cfg.model.feed_forward_dim,
            pooling_type=self.cfg.model.pooling_type,
            vocab_size=self.tokenizer.vocab_size,
            max_length=max(self.cfg.model.max_query_length, self.cfg.model.max_document_length),      # long enough for both queries and documents
            dropout=self.cfg.model.dropout,
        )
    
    def __load_models(self):
        self.query_encoder = self.__build_encoder()
        self.document_encoder = self.__build_encoder()
        
        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if query_weights_path:
            self.query_encoder.load_state_dict(torch.load(query_weights_path, map_location="cpu"))

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if document_weights_path:
            self.document_encoder.load_state_dict(torch.load(document_weights_path, map_location="cpu"))
