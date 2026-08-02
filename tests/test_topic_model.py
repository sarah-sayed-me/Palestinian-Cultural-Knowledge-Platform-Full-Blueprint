from src.nlp.topic_model import aggregate_document_topics, topic_label


class _FakeTopicModel:
    """Stands in for BERTopic.get_topic() so tests don't need a real fit."""

    def __init__(self, topics: dict):
        self._topics = topics

    def get_topic(self, topic_id):
        return self._topics.get(topic_id, False)


def test_topic_label_uses_top_n_keywords():
    model = _FakeTopicModel({0: [("فلسطين", 0.9), ("تراث", 0.8), ("ثقافة", 0.7), ("تاريخ", 0.6), ("قدس", 0.5)]})

    label = topic_label(model, 0, top_n=3)

    assert label == "فلسطين / تراث / ثقافة"


def test_topic_label_outlier_is_explicit():
    model = _FakeTopicModel({})

    assert topic_label(model, -1) == "outlier"


def test_topic_label_missing_topic_falls_back_to_generic_name():
    model = _FakeTopicModel({})

    assert topic_label(model, 3) == "topic_3"


def test_aggregate_document_topics_majority_vote():
    model = _FakeTopicModel({0: [("a", 1.0)], 1: [("b", 1.0)]})
    # doc-1 has 3 chunks: two topic 0, one topic 1 -> majority is topic 0
    doc_ids = ["doc-1", "doc-1", "doc-1", "doc-2"]
    chunk_topics = [0, 0, 1, 1]

    result = aggregate_document_topics(doc_ids, chunk_topics, model)

    assert result["doc-1"]["topic_id"] == 0
    assert result["doc-1"]["topic_label"] == "a"
    assert result["doc-2"]["topic_id"] == 1
    assert result["doc-2"]["topic_label"] == "b"


def test_aggregate_document_topics_handles_outlier_majority():
    model = _FakeTopicModel({0: [("a", 1.0)]})
    doc_ids = ["doc-1", "doc-1"]
    chunk_topics = [-1, -1]

    result = aggregate_document_topics(doc_ids, chunk_topics, model)

    assert result["doc-1"]["topic_id"] == -1
    assert result["doc-1"]["topic_label"] == "outlier"
