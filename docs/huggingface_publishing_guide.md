# Hugging Face Dataset Publishing Guide

This guide describes how to publish the Phase 1 Palestinian cultural corpus as
a Hugging Face Dataset while keeping generated data out of the source repo.

## Recommended Dataset Repository Structure

```text
palestinian-cultural-corpus/
  README.md
  data/
    wikipedia_ar/
      train.parquet
      dataset_summary.json
```

Use Parquet as the canonical public format. It is compact, typed, columnar, and
loads efficiently with `datasets`. Keep JSONL for local ingestion checkpoints
because it is append-friendly and easy to inspect after interrupted runs.

## Create the Dataset Repository

1. Create a Hugging Face account or organization.
2. Install and authenticate the CLI:

```powershell
hf auth login
```

3. Create the dataset repository:

```powershell
hf repo create palestinian-cultural-corpus --type dataset
```

For organization-owned datasets, use:

```powershell
hf repo create ORG_NAME/palestinian-cultural-corpus --type dataset
```

## Build the First Dataset Version

Run the local collection and export flow:

```powershell
python main.py --max-docs 100
python scripts/export_to_hf.py
```

The export creates:

```text
data/processed/hf/wikipedia_ar/train.parquet
data/processed/hf/wikipedia_ar/dataset_summary.json
```

These generated files stay ignored by Git in this source repository. Upload
them to the Hugging Face dataset repository, not to the code repository.

## Dataset Card

The dataset repository `README.md` should include:

- Project title: Palestine Beyond Conflict: Recovering Cultural Identity from Digital Text Using NLP
- Source summary: Arabic Wikipedia, category-expanded Palestine-related articles
- Fields: text, title, source_url, language, credibility, quality_score, Wikipedia IDs, categories
- Collection date and pipeline version
- License and source attribution notes
- Known limitations: Wikipedia coverage bias, category traversal bias, evolving article content
- Intended uses: cultural NLP research, topic modeling, classification, KG construction, RAG
- Out-of-scope uses: claims of exhaustive cultural representation or authoritative history

## Upload the Dataset

Clone the dataset repository:

```powershell
git lfs install
git clone https://huggingface.co/datasets/ORG_OR_USER/palestinian-cultural-corpus
```

Copy the exported dataset files into the dataset repo:

```text
data/wikipedia_ar/train.parquet
data/wikipedia_ar/dataset_summary.json
```

Commit and push:

```powershell
git add README.md data/wikipedia_ar/train.parquet data/wikipedia_ar/dataset_summary.json
git commit -m "Add initial Arabic Wikipedia corpus"
git tag v0.1.0
git push
git push origin v0.1.0
```

## Versioning Strategy

Use semantic dataset releases:

- `v0.1.x`: Arabic Wikipedia Phase 1 samples and fixes
- `v0.2.x`: expanded Arabic Wikipedia corpus
- `v0.3.x`: additional curated Phase 1 sources
- `v1.0.0`: stable documented corpus release

Every release should include the collection date, source config hash or commit,
number of documents, quality thresholds, and known limitations.

## Downloading the Dataset

Future contributors can load the dataset with:

```python
from datasets import load_dataset

dataset = load_dataset("ORG_OR_USER/palestinian-cultural-corpus", data_dir="data/wikipedia_ar")
```

For a specific release:

```python
dataset = load_dataset(
    "ORG_OR_USER/palestinian-cultural-corpus",
    data_dir="data/wikipedia_ar",
    revision="v0.1.0",
)
```

## Updating the Dataset

Contributors should:

1. Run the pipeline from a clean code commit.
2. Export to Parquet.
3. Compare `dataset_summary.json` against the previous release.
4. Open a pull request or upload to a staging branch in the dataset repo.
5. Tag the release after review.

For tens of thousands of documents, prefer partitioned Parquet files by source
and language, for example `data/wikipedia_ar/train-00000-of-00005.parquet`.
Keep raw collection snapshots outside the Hugging Face dataset unless there is
a clear reproducibility requirement and licensing allows redistribution.
