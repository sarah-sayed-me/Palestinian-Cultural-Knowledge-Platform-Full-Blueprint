"""
Shared evaluation harness.

One report shape (schemas.EvalReport) reused across every phase's eval script —
ner_eval.py, embedding_eval.py, retrieval_eval.py, rag_eval.py, kg_eval.py — so
results are comparable and a future dashboard can render all of them uniformly.
See ROADMAP.md Track C.

Note: this package is named `eval`, shadowing the `eval()` builtin for any file
that does a bare `import eval`. Prefer `from eval.schemas import EvalReport` (as
used throughout this project) over `import eval`.
"""
