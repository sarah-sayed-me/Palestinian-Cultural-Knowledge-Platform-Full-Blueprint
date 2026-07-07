# Palestinian Cultural Knowledge Platform

Phase 1 focuses on collecting a reproducible Palestinian cultural text corpus,
starting with Arabic Wikipedia.

## Current Pipeline

```powershell
python main.py --max-docs 100
python scripts/export_to_hf.py
```

The ingestion pipeline reads `configs/sources.yaml`, collects Arabic Wikipedia
articles from Palestine-related seed categories, normalises text, applies
quality checks, removes near-duplicates, and writes ignored local outputs under
`data/`.

## Outputs

- `data/processed/wikipedia_ar_documents.jsonl`: accepted processed documents
- `data/metadata/wikipedia_ar_rejected.jsonl`: rejected and duplicate documents
- `data/metadata/wikipedia_ar_stats.json`: run statistics
- `data/processed/hf/wikipedia_ar/train.parquet`: Hugging Face-ready export

Generated datasets are intentionally ignored by Git. See
`docs/huggingface_publishing_guide.md` for publishing instructions.
