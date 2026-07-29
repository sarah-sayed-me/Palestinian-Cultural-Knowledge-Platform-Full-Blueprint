import json

from src.knowlegde_graph.wikidata_aliases import (
    _qid_from_uri,
    fetch_palestine_aliases,
    read_alias_dump,
    write_alias_dump,
)


def _fake_response(bindings):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": {"bindings": bindings}}

    return FakeResponse()


def test_qid_from_uri_extracts_trailing_segment():
    assert _qid_from_uri("http://www.wikidata.org/entity/Q219060") == "Q219060"


def test_fetch_palestine_aliases_merges_labels_and_aliases(monkeypatch):
    label_bindings = [
        {
            "item": {"value": "http://www.wikidata.org/entity/Q1234"},
            "itemLabel": {"value": "يافا"},
            "instance": {"value": "http://www.wikidata.org/entity/Q515"},
        }
    ]
    alias_bindings = [
        {"item": {"value": "http://www.wikidata.org/entity/Q1234"}, "altLabel": {"value": "Jaffa"}},
        {"item": {"value": "http://www.wikidata.org/entity/Q1234"}, "altLabel": {"value": "يافا"}},
    ]
    anchor_label_bindings = [{"itemLabel": {"value": "فلسطين"}}]
    anchor_alias_bindings = [{"altLabel": {"value": "Palestine"}}]

    # Four queries now: labels+instance-of, alt-labels, then the anchor
    # item's own label and alt-labels (see module docstring — the anchor
    # QID is a filter *value* in the first two queries, never a result row).
    responses = [label_bindings, alias_bindings, anchor_label_bindings, anchor_alias_bindings]
    calls = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        response = _fake_response(responses[calls["n"]])
        calls["n"] += 1
        return response

    monkeypatch.setattr("src.knowlegde_graph.wikidata_aliases.httpx.get", fake_get)

    entries = fetch_palestine_aliases(limit=10, timeout=5.0)

    assert calls["n"] == 4
    by_qid = {e["qid"]: e for e in entries}
    entry = by_qid["Q1234"]
    assert entry["label"] == "يافا"
    assert entry["instance_of"] == ["Q515"]
    assert "Jaffa" in entry["aliases"]
    # the alt-label identical to the item's own label is still recorded —
    # deduping against itemLabel is not this function's job, the linker's
    # exact-index naturally collapses duplicates via normalize_arabic()
    assert "يافا" in entry["aliases"]

    anchor = by_qid["Q219060"]
    assert anchor["label"] == "فلسطين"
    assert "Palestine" in anchor["aliases"]


def test_write_and_read_alias_dump_roundtrip(tmp_path):
    entries = [{"qid": "Q1", "label": "غزة", "aliases": ["Gaza"], "instance_of": ["Q515"]}]
    path = tmp_path / "aliases.jsonl"

    write_alias_dump(entries, path)
    loaded = read_alias_dump(path)

    assert loaded == entries
    # sanity: it's really JSONL, one object per line
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["qid"] == "Q1"
