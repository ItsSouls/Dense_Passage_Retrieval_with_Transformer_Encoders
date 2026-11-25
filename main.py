import hydra
from src.data_utils.corpus import Corpus
from src.retrievers.tf_idf import TFIDF
from src.retrievers.bm25 import BM25
from src.retrievers.dpr import DPR
from src.retrievers.dense_retriever import DenseRetriever
from src.data_utils.query import Query
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate

# Names printed in the results table for each split
SPLIT_LABELS = {'train': 'Train', 'validation': 'Val', 'test': 'Test'}


def buildQueries(split, relevant_ids):
    """One Query per row of a dataset split, with its relevant document ids."""
    questions = list(split['question'])
    query_ids = list(split['query_id'])
    return [Query(question, id=query_id, relevant_document_ids=relevant_ids[query_id])
            for question, query_id in zip(questions, query_ids)]


@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    ### Prepare queries for the splits listed in the dataset config
    split_names = list(cfg.dataset.eval_splits)
    queries = {}
    for split_name in split_names:
        assert split_name in corpus.ds, (f"Split '{split_name}' not found in {cfg.dataset.name}. "
                                         f"Available splits: {list(corpus.ds.keys())}")
        queries[split_name] = buildQueries(corpus.ds[split_name], corpus.relevant_ids)
    
    ### Initialize TF-IDF class
    method = cfg.retriever.method

    if method == "tf_idf":
        retriever = TFIDF(documents=corpus.all_documents)
    elif method == "bm25":
        retriever = BM25(documents=corpus.all_documents, k1=cfg.retriever.k1, b=cfg.retriever.b)
    elif method == "dpr":
        dpr = DPR(cfg)
        # TEMPORARY until T03 is merged - do not commit
        import torch
        state = torch.load("outputs/train/weights/best_encoders.pt", map_location="cpu")
        dpr.query_encoder.load_state_dict(state["query_encoder"])
        dpr.document_encoder.load_state_dict(state["document_encoder"])
        retriever = DenseRetriever(
            documents=corpus.all_documents,
            query_encoder=dpr.query_encoder,
            document_encoder=dpr.document_encoder,
            tokenizer=dpr.tokenizer,
            device=cfg.device,
            max_query_length=cfg.model.max_query_length,
            max_doc_length=cfg.model.max_document_length,
        )
    else:
        raise ValueError(f"Unknown retriever: {method}")
    
   
    ### Retrieve top-k relevant documents (once, at the largest k)
    max_k = max(cfg.k_values)
    relevancies = {split_name: retrieveQueryAndGetRelevancies(retriever, queries[split_name], k=max_k)
                   for split_name in split_names}

    ### Evaluate each split at every k
    for k in cfg.k_values:
        print(f'\nResults for {method} @ k={k}')
        print(f'{"Split":<8}{"P@k":>10}{"R@k":>10}{"R@k std":>10}{"MRR":>10}')
        for split_name in split_names:
            cut = [r[:k] for r in relevancies[split_name]]
            scores = Evaluate(cut, queries[split_name], k)
            label = SPLIT_LABELS.get(split_name, split_name)
            print(f'{label:<8}{scores["precision"]:>10.4f}{scores["recall"]:>10.4f}'
                  f'{scores["recall_standard"]:>10.4f}{scores["mrr"]:>10.4f}')

if __name__ == '__main__':
    main()