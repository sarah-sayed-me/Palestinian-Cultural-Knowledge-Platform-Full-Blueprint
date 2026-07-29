"""
Fetch a Palestine-related Wikidata alias dump (Track E2).

This backs the cheap alias-table entity linker (src/knowlegde_graph/entity_linking.py)
per ROADMAP.md's explicit instruction to start with "a cheap alias-table linker
(exact/fuzzy match against a pre-pulled Wikidata Palestine-entity SPARQL dump...)
before reaching for mGENRE's full model." This module is that pre-pull step —
a bounded, single-purpose SPARQL query against query.wikidata.org, cached to
disk (scripts/fetch_wikidata_aliases.py), not a general Wikidata client and
not a live lookup at link time.

Query scope — three ways an item can be "Palestine-related" on Wikidata:
  - wdt:P17 (country) = Q219060 (State of Palestine)              -> places, institutions
  - wdt:P27 (country of citizenship) = Q219060                     -> people
  - wdt:P495 (country of origin) = Q219060                         -> cultural/heritage items
    (dishes, crafts, etc. — the P17/P27 patterns alone would miss these)

Two separate queries (labels+instance-of, then alt-labels) rather than one
query with GROUP_CONCAT — keeps each query simple and avoids SPARQL
aggregation interacting awkwardly with the wikibase:label SERVICE.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
PALESTINE_QID = "Q219060"

# Wikimedia's bot policy (https://w.wiki/4wJS) rejects generic User-Agents with
# a 403 — verified directly against the live endpoint while building this. It
# specifically requires a URL or email in the parenthesized part; a UA
# without either (even with descriptive text) still gets rejected.
USER_AGENT = (
    "PalestinianCulturalKnowledgePlatform/0.1 "
    "(https://github.com/local/palestinian-cultural-knowledge-platform; "
    "research-corpus-project@example.org) httpx/0.28"
)

_LABELS_QUERY = """
SELECT DISTINCT ?item ?itemLabel ?instance WHERE {{
  {{ ?item wdt:P17 wd:{qid} . }}
  UNION {{ ?item wdt:P27 wd:{qid} . }}
  UNION {{ ?item wdt:P495 wd:{qid} . }}
  OPTIONAL {{ ?item wdt:P31 ?instance . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "ar,en". }}
}}
LIMIT {limit}
"""

_ALIASES_QUERY = """
SELECT DISTINCT ?item ?altLabel WHERE {{
  {{ ?item wdt:P17 wd:{qid} . }}
  UNION {{ ?item wdt:P27 wd:{qid} . }}
  UNION {{ ?item wdt:P495 wd:{qid} . }}
  ?item skos:altLabel ?altLabel .
  FILTER(LANG(?altLabel) IN ("ar", "en"))
}}
LIMIT {limit}
"""

# The anchor item itself (wd:Q219060, "State of Palestine") is used as a
# FILTER VALUE in the two queries above (wdt:P17/P27/P495 = wd:Q219060) — it
# never appears as a ?item in their results, since nothing has P17 pointing
# at itself. Found by actually checking: "فلسطين" (Palestine), this corpus's
# single most-mentioned entity (1,709 mentions), was silently absent from
# the dump entirely, so the linker fell through to a same-labeled but wrong
# item (a newspaper named "Felesteen"). Fetched separately and merged in.
_ANCHOR_LABEL_QUERY = """
SELECT ?itemLabel WHERE {{
  BIND(wd:{qid} AS ?item)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "ar,en". }}
}}
"""

_ANCHOR_ALIASES_QUERY = """
SELECT ?altLabel WHERE {{
  wd:{qid} skos:altLabel ?altLabel .
  FILTER(LANG(?altLabel) IN ("ar", "en"))
}}
"""


def _run_query(query: str, timeout: float) -> List[dict]:
    response = httpx.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("results", {}).get("bindings", [])


def _qid_from_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def fetch_palestine_aliases(*, limit: int = 5000, timeout: float = 60.0) -> List[Dict[str, Any]]:
    """Query Wikidata for Palestine-related items plus their labels/aliases.

    Returns a list of {"qid", "label", "aliases": [...], "instance_of": [...]}
    dicts, one per distinct Wikidata item.
    """
    label_rows = _run_query(_LABELS_QUERY.format(qid=PALESTINE_QID, limit=limit), timeout)
    alias_rows = _run_query(_ALIASES_QUERY.format(qid=PALESTINE_QID, limit=limit), timeout)

    items: Dict[str, Dict[str, Any]] = {}

    for row in label_rows:
        qid = _qid_from_uri(row["item"]["value"])
        entry = items.setdefault(qid, {"qid": qid, "label": None, "aliases": [], "instance_of": []})
        label = row.get("itemLabel", {}).get("value")
        if label and entry["label"] is None:
            entry["label"] = label
        instance = row.get("instance")
        if instance:
            instance_qid = _qid_from_uri(instance["value"])
            if instance_qid not in entry["instance_of"]:
                entry["instance_of"].append(instance_qid)

    for row in alias_rows:
        qid = _qid_from_uri(row["item"]["value"])
        entry = items.setdefault(qid, {"qid": qid, "label": None, "aliases": [], "instance_of": []})
        alias = row.get("altLabel", {}).get("value")
        if alias and alias not in entry["aliases"]:
            entry["aliases"].append(alias)

    anchor_label_rows = _run_query(_ANCHOR_LABEL_QUERY.format(qid=PALESTINE_QID), timeout)
    anchor_alias_rows = _run_query(_ANCHOR_ALIASES_QUERY.format(qid=PALESTINE_QID), timeout)
    anchor_entry = items.setdefault(
        PALESTINE_QID, {"qid": PALESTINE_QID, "label": None, "aliases": [], "instance_of": []}
    )
    for row in anchor_label_rows:
        label = row.get("itemLabel", {}).get("value")
        if label and anchor_entry["label"] is None:
            anchor_entry["label"] = label
    for row in anchor_alias_rows:
        alias = row.get("altLabel", {}).get("value")
        if alias and alias not in anchor_entry["aliases"]:
            anchor_entry["aliases"].append(alias)

    return sorted(items.values(), key=lambda e: e["qid"])


def write_alias_dump(entries: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_alias_dump(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
