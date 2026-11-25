from src.retrievers.models.encoder import TransformerEncoder
from transformers import AutoTokenizer
import torch
from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.models.pretrained_encoder import PretrainedEncoder


class DPR:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.tokenizer_name)
        self.__load_models()

    def __build_encoder(self):
        # Pretrained encoder (e.g. BERT) if the model config says so
        if self.cfg.model.get("type", "scratch") == "pretrained":
            return PretrainedEncoder(
                pretrained_name=self.cfg.model.pretrained_name,
                pooling_type=self.cfg.model.pooling_type,
                attn_implementation=self.cfg.model.get("attn_implementation", "sdpa"),
            )

        # Otherwise: our TransformerEncoder from scratch (random initial weights)
        return TransformerEncoder(
            n_layers=self.cfg.model.n_layers,
            n_heads=self.cfg.model.n_heads,
            emb_dim=self.cfg.model.emb_dim,
            hidden_dim=self.cfg.model.hidden_dim,
            feed_forward_dim=self.cfg.model.feed_forward_dim,
            pooling_type=self.cfg.model.pooling_type,
            vocab_size=self.tokenizer.vocab_size,
            max_length=max(self.cfg.model.max_query_length, self.cfg.model.max_document_length),
            dropout=self.cfg.model.dropout,
        )
    
    def __load_models(self):
        # Two encoders with the same config but separate weights
        self.query_encoder = self.__build_encoder()
        self.document_encoder = self.__build_encoder()
        
        # Load trained weights if a path is given in the config (otherwise keep random weights)
        # Trainer.save writes both encoders in one file, under the keys 'query_encoder' and 'document_encoder'
        weights_path = self.cfg.model.get('encoders_weights_path', None)
        if weights_path:
            checkpoint = torch.load(weights_path, map_location="cpu")
            self.query_encoder.load_state_dict(checkpoint["query_encoder"])
            self.document_encoder.load_state_dict(checkpoint["document_encoder"])
