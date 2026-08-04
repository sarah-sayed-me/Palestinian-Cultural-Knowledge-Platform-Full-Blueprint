from src.nlp.topic_model import aggregate_document_coords, aggregate_document_topics, fetch_chunk_embeddings, topic_label


class _FakeVector:
    """Stands in for pgvector's Vector type — has .to_list(), not __iter__,
    so list(vector) fails (the real bug this regression test catches)."""

    def __init__(self, values):
        self._values = values

    def to_list(self):
        return self._values


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *a, **k):
        pass

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)


def test_fetch_chunk_embeddings_handles_pgvector_vector_type():
    rows = [
        ("chunk-1", "doc-1", "some text", _FakeVector([0.1, 0.2, 0.3])),
        ("chunk-2", "doc-1", "more text", _FakeVector([0.4, 0.5, 0.6])),
    ]
    conn = _FakeConn(rows)

    chunk_ids, doc_ids, texts, embeddings = fetch_chunk_embeddings(conn, table="rag_chunks")

    assert chunk_ids == ["chunk-1", "chunk-2"]
    assert doc_ids == ["doc-1", "doc-1"]
    assert texts == ["some text", "more text"]
    assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


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


def test_aggregate_document_coords_averages_chunk_positions():
    doc_ids = ["doc-1", "doc-1", "doc-2"]
    coords = [(0.0, 0.0), (2.0, 4.0), (5.0, 5.0)]

    result = aggregate_document_coords(doc_ids, coords)

    assert result["doc-1"] == (1.0, 2.0)
    assert result["doc-2"] == (5.0, 5.0)
