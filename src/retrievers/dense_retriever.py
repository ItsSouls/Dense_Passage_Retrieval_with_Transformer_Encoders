import torch

from src.data_utils import query
from src.retrievers.base_retriever import Retriever
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.data_utils.query import Query
from src.evaluation.metrics import Evaluate


class DenseRetriever(Retriever):
    def __init__(self, documents, query_encoder, document_encoder, tokenizer, device,
                 batch_size=64, max_query_length=64, max_doc_length=256):
        super().__init__(documents)
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_query_length = max_query_length
        self.max_doc_length= max_doc_length
        self.query_encoder = query_encoder.to(device)
        self.document_encoder = document_encoder.to(device)
        self.doc_embeddings = self._encode([i.GetText() for i in self.index.GetDocuments()], self.document_encoder, self.max_doc_length)

    def _encode(self, texts, encoder, max_length):
        encoder.eval()
        vector_batch_list = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i : i + self.batch_size]
                tokens = self.tokenizer(
                    chunk,
                    padding=True,          
                    truncation=True,       
                    max_length= max_length,       
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].to(self.device)
                attention_mask =  tokens["attention_mask"].to(self.device)
                
                vector_batch_list.append(encoder(input_ids,attention_mask))
            
        return torch.cat(vector_batch_list, dim=0)


    def CalculateScores(self, queries):
        """
        @param queries: list[str]
        Returns: n_queries x N scores (query-document dot products).
        Column j must correspond to document j of self.index.GetDocuments().
        """
        query_embeddings = self._encode(queries, self.query_encoder, self.max_query_length)
        scores = query_embeddings @ self.doc_embeddings.T
        return scores.cpu().numpy()

def make_recall_val_fn(documents, queries, tokenizer, device, k=10):
    def val_fn(query_encoder, document_encoder):
        # 1. Indexar el corpus con los encoders ACTUALES
        retriever = DenseRetriever(documents, query_encoder, document_encoder, tokenizer, device)

        # 2. Top-k de cada query y cuáles son relevantes
        relevancies = retrieveQueryAndGetRelevancies(retriever, queries, k)

        # 3. Métricas
        scores = Evaluate(relevancies, queries, k)

        return {
            f"recall@{k}": scores["recall"],
            f"mrr@{k}":  scores["mrr"]
        }

    return val_fn