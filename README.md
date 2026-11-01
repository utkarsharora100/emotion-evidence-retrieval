# Emotion Evidence Retrieval

This project helps analysts understand online discussions.

It finds emotional comments, shows community differences, and gives source evidence. It does more than return one sentiment label. It also shows the reply and community context around each result.

This is useful for complaint review, customer research, search, and knowledge work. It can also support RAG tools that need clear sources.

## Best result

The notebook uses 58,009 unique, human-rated Reddit comments from GoEmotions. It turns clear majority labels into two classes: positive and negative.

The test set has 5,491 comments. It was made with a stratified 80/20 split.

- Accuracy: **84.94%**
- Macro-F1: **83.12%**
- Weighted-F1: **84.84%**

These are binary results. They are not results for all 28 GoEmotions classes. The project makes this clear so the result is easy to trust.

The target was not changed to reach a better number. Labels were not leaked into the test set. The project reports the best honest result.

Run the notebook at [notebooks/reddit_graph_rag_showcase.ipynb](notebooks/reddit_graph_rag_showcase.ipynb). See the charts and metrics in `outputs/notebook_showcase/`. See the resume text in [RESUME_PROJECT_DESCRIPTION.md](RESUME_PROJECT_DESCRIPTION.md).

## What problem does it solve?

An analyst may ask:

- What are people feeling about a topic?
- Which communities agree or disagree?
- What changed during a time period?
- Which comments should a person review?
- What evidence supports the answer?

A normal sentiment model looks at one comment. A normal RAG system may find text but miss the reply context. This project joins both parts.

It retrieves the text, the parent comment, the reply link, the subreddit, and the time when available. It also keeps the source comments next to the answer.

## How it works

The pipeline has six simple parts:

1. It loads Reddit comments and their metadata.
2. It predicts positive or negative emotion.
3. It finds text with BM25.
4. It finds learned sparse matches with SPLADE.
5. It stores sparse vectors in Qdrant.
6. It uses LangChain to build an evidence-based RAG answer.

The project also builds a reply graph. It uses Louvain to find communities. BM25 gives a strong word match. SPLADE adds a learned sparse match. Qdrant stores the vectors. LangChain joins retrieval with the answer step.

If SPLADE cannot load, the run stops. It does not quietly replace SPLADE with BM25.

## Data

The graph data comes from the [Reddit Politosphere release](https://zenodo.org/records/5851729). Its [scripts and notes](https://github.com/valentinhofmann/politosphere) explain the data.

The data has comments, times, reply links, subreddit names, and masked author IDs. The full release is large. Start with one monthly file.

Put `comments_YYYY-MM.bz2` in `data\politosphere`, then run:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv_sys\Scripts\python.exe -m reddit_rag.rag_cli `
  --politosphere-comments .\data\politosphere\comments_2008-01.bz2 `
  --query "What political criticism appears in the discussion?" `
  --limit 100 `
  --top-k 5
```

This path has no gold sentiment labels. It keeps sentiment as `unlabeled`. This prevents made-up labels.

## Labeled benchmark

The [Brexit Opinion Dynamics dataset](https://github.com/behavioral-ds/brexit-opinion-dynamics) has 5,024 human labels. The labels show anti-Brexit, neither, and pro-Brexit views.

The data also has a larger interaction set. The classes are uneven. The project therefore reports macro-F1 and per-class recall with accuracy.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv_sys\Scripts\python.exe -m reddit_rag.experiment_cli `
  --csv .\data\brexit\rbrexit_stance_annotations.csv.xz `
  --output outputs\brexit_experiment `
  --epochs 8 `
  --skip-retrieval
```

## RAG query

The RAG code uses LangChain `Document` objects. It uses a custom retriever. It joins BM25 and SPLADE results.

Without an LLM key, it returns a simple evidence report. With `OPENAI_API_KEY` and `langchain-openai`, it can write a cited answer.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv_sys\Scripts\python.exe -m reddit_rag.rag_cli `
  --csv .\data\Reddit_Data.csv `
  --query "What evidence is positive or negative about Narendra Modi?" `
  --limit 100 `
  --top-k 5
```

## Setup

Create an environment and install the package:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

For a small smoke test, run the built-in fixture:

```powershell
python -m reddit_rag.cli --demo --limit 5 --query "What do Reddit users say about games?"
```

The fixture needs no Kaggle account. It checks the full path with five rows.

## Kaggle baseline

The original baseline uses the [Twitter and Reddit Sentimental Analysis dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset). Its Reddit file is `Reddit_Data.csv`.

The file has about 37K cleaned comments. The labels are negative, neutral, and positive.

```powershell
python -m reddit_rag.cli --kaggle-dataset cosmos98/twitter-and-reddit-sentimental-analysis-dataset --reddit-only --limit 500
```

The loader checks the text and label columns. If the data is missing, it fails clearly. It does not invent a replacement.

## Scale up

Start with `--limit 500`. Increase the limit after the small run works. Use `--qdrant-path .\qdrant_storage` for local storage.

For a server, set `QDRANT_URL` and `QDRANT_API_KEY`. The simple graph code is slow at large sizes. For the full data, use shards and top-k graph edges before Louvain.

The experiment writes metrics, loss charts, confusion matrices, and label charts. It reports accuracy, F1, precision, recall, Recall@k, MRR@k, and nDCG@k.

The full problem description is in [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md).
