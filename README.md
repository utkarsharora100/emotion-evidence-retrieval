# Emotion Evidence Retrieval

This repository contains a Reddit search and analysis system for political discussions. It focuses on conversations where people support, oppose, or question a public figure, policy, or event.

For example, an analyst might ask: **“How did Reddit communities react to a political topic, and which communities pushed back?”** The system finds related comments, assigns emotional polarity, groups comments by subreddit, and follows reply links when they are available.

The result is more than a label. It includes the comments behind the result, their subreddit, and nearby conversation context. An analyst can then check whether the summary matches the sources.

## What each part is for

- **Emotion benchmark:** checks whether the model can separate all 28 GoEmotions labels in human-rated Reddit comments.
- **BM25 + SPLADE:** combines exact phrase matches with learned sparse matches, which helps when people use different words for the same political idea.
- **Qdrant:** stores the sparse vectors so the retriever can search the comment collection.
- **Graph-aware reranking:** adds community diversity and nearby reply context after the first search, so the results do not all come from one thread.
- **LangChain:** connects the reranked comments to an answer step without hiding the source documents.
- **Reply graph:** keeps links between comments, so the system can compare a comment with its replies and find communities with Louvain.
- **Notebook:** records the split, metrics, and charts in one place so the result can be checked.
- **Command-line demo:** provides a five-row run for testing the pipeline before downloading a large dataset.

## Results

The benchmark uses 48,501 unique Reddit comments from GoEmotions. We keep comments with one clear majority label and retain the full 28-class emotion label.

The split is stratified and uses 80% of the data for training and 20% for testing. The test set contains 9,701 comments.

| Metric | Score |
| --- | ---: |
| Accuracy | 55.10% |
| Macro-F1 | 40.58% |
| Weighted-F1 | 52.39% |

These scores are for the full 28-class task, using majority-vote labels. The confusion matrix, label chart, and community chart are in [`outputs/notebook_showcase/`](outputs/notebook_showcase/). The earlier binary polarity result remains in the older `metrics.json` file.

To reproduce the benchmark, open [`notebooks/reddit_graph_rag_showcase.ipynb`](notebooks/reddit_graph_rag_showcase.ipynb) after downloading the GoEmotions files described below. The notebook keeps all 28 emotion labels instead of collapsing them into positive and negative.

## A concrete example

Suppose an analyst is studying a sudden rise in negative comments about a politician or policy. A sentiment model can count those comments, but it cannot show whether the reaction came from one subreddit, whether other communities disagreed, or what the replies said.

This pipeline retrieves the comments, scores their emotion, and adds the available parent, subreddit, timestamp, and masked author information. It then builds a reply graph and groups related communities, so the analyst can compare the negative comments with the wider discussion.

## Quickstart

Create an environment and install the package from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

The first SPLADE run downloads the model `naver/splade-cocondenser-ensembledistil` from Hugging Face. A model download is required. The code does not silently fall back to BM25 if SPLADE cannot load.

Run the built-in smoke test:

```powershell
python -m reddit_rag.cli --demo --limit 5 --query "What do Reddit users say about games?"
```

This test uses five local fixture rows. It checks the pipeline without needing a dataset, Kaggle credentials, or an LLM key.

## Run the RAG pipeline

Set the source directory on `PYTHONPATH`, then run the command below:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m reddit_rag.rag_cli `
  --csv .\data\Reddit_Data.csv `
  --query "What evidence is positive or negative about Narendra Modi?" `
  --limit 100 `
  --top-k 5
```

Without an LLM key, the command returns an extractive evidence report. With `OPENAI_API_KEY` and the optional `langchain-openai` package, it can generate a cited answer from the same retrieved documents.

For the graph-shaped Politosphere data, use a monthly comments file instead:

```powershell
python -m reddit_rag.rag_cli `
  --politosphere-comments .\data\politosphere\comments_2008-01.bz2 `
  --query "What political criticism appears in the discussion?" `
  --limit 100 `
  --top-k 5
```

## Data setup

The benchmark notebook expects the three official GoEmotions CSV files in `data\goemotions`:

- `goemotions_1.csv`
- `goemotions_2.csv`
- `goemotions_3.csv`

The files are available from the [Google Research GoEmotions repository](https://github.com/google-research/google-research/tree/master/goemotions). The notebook groups rater rows by comment and keeps only clear majority labels.

For the graph demo, download a monthly Politosphere comments file from [Zenodo](https://zenodo.org/records/5851729) and place it in `data\politosphere`. The [Politosphere repository](https://github.com/valentinhofmann/politosphere) documents the release.

For the stance comparison, place `rbrexit_stance_annotations.csv.xz` in `data\brexit`. The file comes from [Brexit Opinion Dynamics](https://github.com/behavioral-ds/brexit-opinion-dynamics).

Data files are ignored by Git. They are not included in this repository.

## Run the labeled comparison

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m reddit_rag.experiment_cli `
  --csv .\data\brexit\rbrexit_stance_annotations.csv.xz `
  --output outputs\brexit_experiment `
  --epochs 8 `
  --skip-retrieval
```

The experiment writes `metrics.json`, `training_loss.png`, `confusion_matrix.png`, and `label_distribution.png`. It reports accuracy, macro-F1, weighted-F1, precision, recall, and per-class scores.

## Repository layout

```text
src/reddit_rag/             Python package and command-line entry points
notebooks/                  Benchmark notebook
outputs/notebook_showcase/  Saved benchmark figures and metrics
PROBLEM_STATEMENT.md        Research question and evaluation plan
RESUME_PROJECT_DESCRIPTION.md  Resume wording and project summary
```

## Current limits

The benchmark now covers all 28 GoEmotions classes. The notebook builds the full reply and subreddit graph for the labeled comments. The RAG demo is local and uses an in-memory index. Scaling the much larger Politosphere release would need sharded ingestion, a persistent Qdrant deployment, and a distributed graph-building step.

Retrieval quality has not yet been judged by human annotators. The current retrieval scores use an explicit label-based proxy, so they should not be treated as human relevance scores. These limits are part of the project rather than hidden assumptions.

The main contribution is the retrieval and evidence layer around the classifier. It combines BM25, SPLADE, Qdrant, reply links, community diversity, and source-aware LangChain answers in one tested path.

The longer research plan is in [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md). The resume version is in [`RESUME_PROJECT_DESCRIPTION.md`](RESUME_PROJECT_DESCRIPTION.md).
