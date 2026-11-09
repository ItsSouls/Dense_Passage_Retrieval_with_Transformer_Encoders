import math
import pytest

from src.data_utils.query import Query
from src.evaluation.metrics import (
    calculatePrecision,
    calculateRecall,          # capped version (already in metrics.py): divides by min(k, |Rq|)
    calculateStandardRecall,  # standard version: divides by |Rq|
    calculateReciprocalRank,
    Evaluate,
)

K = 5


def make_query(n_relevant):
    """Fake query: only the number of relevant ids matters."""
    return Query(text="fake", id="q", relevant_document_ids=[f"doc{i}" for i in range(n_relevant)])


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-9)


# name, retrieved (in ranked order), |Rq|, precision, standard recall, capped recall, reciprocal rank
CASES = [
    ("A", [True, False, False, False, False],  1, 1/5, 1,   1,   1),
    ("B", [False, False, True, False, False],  2, 1/5, 1/2, 1/2, 1/3),
    ("C", [False, False, False, False, False], 3, 0,   0,   0,   0),
    ("D", [True, True, True, True, True],      5, 1,   1,   1,   1),
    ("E", [False, True, False, True, True],    8, 3/5, 3/8, 3/5, 1/2),
    ("F", [False, True, False, False, True],   2, 2/5, 1,   1,   1/2),
]
PARAMS = "name, retrieved, n_relevant, precision, recall_std, recall_cap, rr"
IDS = [case[0] for case in CASES]


@pytest.mark.parametrize(PARAMS, CASES, ids=IDS)
def test_precision(name, retrieved, n_relevant, precision, recall_std, recall_cap, rr):
    assert close(calculatePrecision(retrieved, K), precision)


@pytest.mark.parametrize(PARAMS, CASES, ids=IDS)
def test_standard_recall(name, retrieved, n_relevant, precision, recall_std, recall_cap, rr):
    assert close(calculateStandardRecall(retrieved, make_query(n_relevant)), recall_std)


@pytest.mark.parametrize(PARAMS, CASES, ids=IDS)
def test_capped_recall(name, retrieved, n_relevant, precision, recall_std, recall_cap, rr):
    assert close(calculateRecall(retrieved, make_query(n_relevant)), recall_cap)


@pytest.mark.parametrize(PARAMS, CASES, ids=IDS)
def test_reciprocal_rank(name, retrieved, n_relevant, precision, recall_std, recall_cap, rr):
    assert close(calculateReciprocalRank(retrieved), rr)


def test_evaluate_averages():
    relevancies = [case[1] for case in CASES]
    queries = [make_query(case[2]) for case in CASES]
    scores = Evaluate(relevancies, queries, K)
    assert close(scores["precision"], 2/5)
    assert close(scores["recall"], 4.1/6)  # capped; use 3.875/6 if the group picks standard recall
    assert close(scores["mrr"], 5/9)


def test_evaluate_rejects_mismatched_lengths():
    with pytest.raises(AssertionError):
        Evaluate([[True, False]], [], K)