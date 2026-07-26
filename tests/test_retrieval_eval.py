from eval.retrieval_eval import aggregate, doc_id_of


def test_doc_id_of_strips_paragraph_suffix():
    assert doc_id_of("abc123_p4") == "abc123"
    assert doc_id_of("abc123_p68") == "abc123"


def test_aggregate_skips_unevaluable_queries_from_denominators():
    results = [
        {"evaluable": False},
        {"evaluable": True, "reciprocal_rank": 1.0, "hit_at_5": True, "hit_at_10": True,
         "top1_score": 0.9, "hit_score": 0.9},
        {"evaluable": True, "reciprocal_rank": 0.0, "hit_at_5": False, "hit_at_10": False,
         "top1_score": 0.3, "hit_score": None},
    ]

    metrics = aggregate(results)

    assert metrics["evaluable_queries"] == 2
    assert metrics["skipped_queries"] == 1
    assert metrics["recall_at_5"] == 0.5
    assert metrics["mrr"] == 0.5


def test_aggregate_returns_only_counts_when_nothing_is_evaluable():
    metrics = aggregate([{"evaluable": False}])

    assert metrics["evaluable_queries"] == 0
    assert metrics["skipped_queries"] == 1
    assert "mrr" not in metrics
