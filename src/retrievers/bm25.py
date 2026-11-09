from scipy.sparse import lil_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document
import numpy as np 

class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        self.bm25_matrix = self.GetDocumentBM25Vectors()
    
    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.PreprocessText(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        return document_lengths, average_document_length/len(self.index.GetDocuments())
    
    # TODO: Add methods for computing self.bm25_matrix
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        doc_frequencies = self.GetDocumentFrequencies()
        n_documents = len(self.index.GetDocuments())
        idf = {}
        for term, freq in doc_frequencies.items(): 
            idf[term] = np.log((n_documents + 1) / (freq + 0.5))
        return idf

    def GetDocumentBM25Vectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the bm25 score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        bm25_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)
        document_lengths, average_document_length = self.GetDocumentLengths()

        # TODO: Add tf-idf code
        for i, document in enumerate(self.index.GetDocuments()):
            term_counts = self.GetDocumentTermCounts(document)  

            for term, count in term_counts.items():
                tf = (count * (self.k1 + 1)) / (count + self.k1 * (1 - self.b + self.b * (document_lengths[document] / average_document_length))) 
                idf = self.idf[term] 
                bm25 = tf * idf
                bm25_matrix[i, self.term_to_index[term]] = bm25

        return bm25_matrix.tocsr()

    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores