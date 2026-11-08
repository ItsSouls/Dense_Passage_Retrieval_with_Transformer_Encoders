from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
import numpy as np
from scipy.sparse import lil_matrix


class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()

        # Only run if we are in TFIDF class
        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted characters from text and lowercase
        """
        text = text.lower() 
        # preserve letters, numbers, whitespace; replace all punctuation or special characters with whitespace
        text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
        bag_of_words = text.split()
        return bag_of_words
        
    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """
        words = self.PreprocessText(query)
        query_unique_words = list(set(words)) 
        return query_unique_words

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """
        doc = document.GetText()
        words = self.PreprocessText(doc)
        doc_unique_words = list(set(words))
        return doc_unique_words
    
    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        # Use map function for more efficient processing
        # ready from original repo 
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        corpus_vocabulary = set().union(*processed_texts)
        return corpus_vocabulary

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """
        doc_frequencies = {}
        for document in self.index.GetDocuments():
            unique_terms = self.GetDocumentVocabulary(document)
            for term in unique_terms:
                if term not in doc_frequencies:
                    doc_frequencies[term] = 0 
                doc_frequencies[term] +=1
        return doc_frequencies       
    
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        doc_frequencies = self.GetDocumentFrequencies()
        n_documents = len(self.index.GetDocuments())
        idf = {}
        for term, freq in doc_frequencies.items(): 
            idf[term] = np.log(n_documents / freq)
        return idf
    
    
    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """
        tf = {}
        words = self.PreprocessText(document.GetText())
        for word in words:
            if word not in tf:
                tf[word] = 0
            tf[word] += 1
        return tf

    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the tf-idf score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        tfidf_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        # TODO: Add tf-idf code
        for i, document in enumerate(self.index.GetDocuments()):
            term_counts = self.GetDocumentTermCounts(document)  
            max_term_count = max(term_counts.values()) if term_counts else 1

            for term, count in term_counts.items():
                tf = count / max_term_count 
                idf = self.idf[term] 
                tfidf = tf * idf
                tfidf_matrix[i, self.term_to_index[term]] = tfidf

        return tfidf_matrix.tocsr()
    
    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.
        
        Args:
            query (str): The query to convert
        
        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)

        # TODO: add code
        for i, query in enumerate(queries):
            term_counts = {}
            words = self.PreprocessText(query) 
            for word in words: 
                if word in self.term_to_index: 
                    if word not in term_counts: 
                        term_counts[word] = 0 
                    term_counts[word] += 1 

            for term, count in term_counts.items():
                query_vector[i, self.term_to_index[term]] = count
        return query_vector.tocsr()

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        return scores
    