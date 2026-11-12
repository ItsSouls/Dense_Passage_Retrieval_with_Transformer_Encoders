import hydra
from src.data_utils.corpus import Corpus
from src.retrievers.tf_idf import TFIDF
from src.retrievers.bm25 import BM25
from src.data_utils.query import Query
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate

@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    ### Prepare queries
    train_queries = [Query(corpus.train['question'][idx], 
                           id=corpus.train['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.train['query_id'][idx]]
                           ) 
                        for idx in range(corpus.train.num_rows)
                    ]

    val_queries =   [Query(corpus.val['question'][idx], 
                         id=corpus.val['query_id'][idx], 
                         relevant_document_ids=corpus.relevant_ids[corpus.val['query_id'][idx]]
                           ) 
                        for idx in range(corpus.val.num_rows)
                    ]
    
    test_queries =  [Query(corpus.test['question'][idx], 
                           id=corpus.test['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.test['query_id'][idx]]
                           ) 
                        for idx in range(corpus.test.num_rows)
                    ]
    
    ### Initialize TF-IDF class
    method = cfg.retriever.method

    if method == "tf_idf":
     retriever = TFIDF(documents=corpus.all_documents)
    elif method == "bm25":
     retriever = BM25(documents=corpus.all_documents, k1=cfg.retriever.k1, b=cfg.retriever.b)
    elif method == "dpr":
        raise NotImplementedError("DPR not ready yet")
    else:
        raise ValueError(f"Unknown retriever: {method}")
    
   
    ### Retrieve top-k relevant documents
    train_relevancies = retrieveQueryAndGetRelevancies(retriever, train_queries, k=cfg.top_k)
    val_relevancies = retrieveQueryAndGetRelevancies(retriever, val_queries, k=cfg.top_k)
    test_relevancies = retrieveQueryAndGetRelevancies(retriever, test_queries, k=cfg.top_k)

    ### Evaluate
    
    train_scores = Evaluate(train_relevancies, train_queries, cfg.top_k)
    val_scores = Evaluate(val_relevancies, val_queries, cfg.top_k)
    test_scores = Evaluate(test_relevancies, test_queries, cfg.top_k)

    ### Print results
    print(f'\nResults @ k={cfg.top_k}')
    print(f'{"Split":<8}{"P@k":>10}{"R@k":>10}{"R@k std":>10}{"MRR":>10}')
    for split, scores in [('Train', train_scores), ('Val', val_scores), ('Test', test_scores)]:
        print(f'{split:<8}{scores["precision"]:>10.4f}{scores["recall"]:>10.4f}'
              f'{scores["recall_standard"]:>10.4f}{scores["mrr"]:>10.4f}')

if __name__ == '__main__':
    main()