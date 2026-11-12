from src.data_utils.query import Query


def calculatePrecision(relevancies: list[bool], k: int):
    """Fraction of the k retrieved documents that are relevant."""
    return sum(relevancies) / k


def calculateRecall(relevancies: list[bool], query: Query):
    """Capped recall: divides by min(k, |Rq|), so a perfect top-k always scores 1."""
    n_relevant = query.getNumberOfRelevantDocuments()
    if n_relevant == 0:
        return 0.0
    return sum(relevancies) / min(len(relevancies), n_relevant)


def calculateStandardRecall(relevancies: list[bool], query: Query):
    """Standard recall: fraction of all relevant documents that appear in the top-k."""
    n_relevant = query.getNumberOfRelevantDocuments()
    if n_relevant == 0:
        return 0.0
    return sum(relevancies) / n_relevant


def calculateReciprocalRank(relevancies: list[bool]):
    """1 / rank of the first relevant document (ranks start at 1); 0 if none is in the top-k."""
    for index, is_relevant in enumerate(relevancies):
        if is_relevant:
            return 1 / (index + 1)
    return 0.0


def Evaluate(relevancies: list[list[bool]], queries: list[Query], k: int):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    n_queries = len(queries)
    total_precision = 0
    total_recall = 0
    total_standard_recall = 0
    total_reciprocal_rank = 0

    for i in range(n_queries):
        total_precision += calculatePrecision(relevancies[i], k)
        total_recall += calculateRecall(relevancies[i], queries[i])
        total_standard_recall += calculateStandardRecall(relevancies[i], queries[i])
        total_reciprocal_rank += calculateReciprocalRank(relevancies[i])

    scores = {
        'precision': total_precision / n_queries,
        'recall': total_recall / n_queries,
        'recall_standard': total_standard_recall / n_queries,
        'mrr': total_reciprocal_rank / n_queries,
    }
    return scores