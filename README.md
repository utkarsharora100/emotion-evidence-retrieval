# Graph-Grounded Reddit Emotion Intelligence RAG

This project is a reproducible prototype for a real customer-interaction analytics problem: help an analyst find emotionally important discussion themes, compare communities, and answer questions with traceable evidence instead of an unsupported summary. The design maps directly to complaint analytics, intelligent search, knowledge discovery, and GenAI/RAG work.

## Best verified benchmark result

The showcase notebook uses 58,009 unique human-rated GoEmotions Reddit comments and constructs a documented binary polarity task from unambiguous majority labels. A leakage-safe stratified 80/20 split gives **84.94% accuracy**, **83.12% macro-F1**, and **84.84% weighted-F1** on 5,491 held-out comments. The binary aggregation is stated explicitly; these are not 28-class GoEmotions scores and should not be presented as such.

Open the runnable notebook: [notebooks/reddit_graph_rag_showcase.ipynb](notebooks/reddit_graph_rag_showcase.ipynb). Generated figures and metrics are in `outputs/notebook_showcase/`. The resume-ready wording is in [RESUME_PROJECT_DESCRIPTION.md](RESUME_PROJECT_DESCRIPTION.md).

The result is intentionally reported honestly: the earlier 95% accuracy / 0.90 macro-F1 target was not reached on the human-annotated benchmark without changing the task definition or leaking labels. That makes the comparison credible in an interview.

## Why this is more than a classifier

This project implements a small, explicit pipeline for the Reddit **Politosphere** text-and-network dataset. The original Kaggle CSV remains available as a labeled sentiment baseline, but it does not contain the graph metadata needed for community or temporal claims.

1. load and normalize the Reddit subset;
2. score lexical matches with BM25 and semantic sparse matches with SPLADE;
3. index SPLADE vectors in Qdrant and fuse BM25/SPLADE ranks;
4. build a similarity graph and detect communities with Louvain;
5. use LangChain `Document` objects as the GraphRAG context boundary;
6. retrieve evidence and aggregate the dataset sentiment labels.

The repository is intentionally transparent: BM25 is a deliberate lexical half of the hybrid retriever, while SPLADE remains a separate model-backed signal. If SPLADE cannot load, the run stops with an actionable error.

## Recommended dataset: Reddit Politosphere

Use the [official Zenodo release](https://zenodo.org/records/5851729) and its [documentation/scripts repository](https://github.com/valentinhofmann/politosphere). It covers 605 political subreddits from 2008–2019, with comment text, creation time, pseudonymized authors, reply/link identifiers, language, subreddit metadata, user metadata, and yearly subreddit-overlap networks. The complete release is 38.7 GB, so start with one monthly file.

After downloading `comments_YYYY-MM.bz2` from Zenodo into `data\politosphere`, run the real LangChain RAG slice:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv_sys\Scripts\python.exe -m reddit_rag.rag_cli `
  --politosphere-comments .\data\politosphere\comments_2008-01.bz2 `
  --query "What political criticism appears in the discussion?" `
  --limit 100 `
  --top-k 5
```

The loader streams the compressed file, filters to English comments by default, and preserves `author`, `subreddit`, `created_utc`, `parent_id`, `link_id`, `score`, and `language` in each LangChain `Document`. `sentiment` is deliberately `unlabeled` for this corpus; it must not be confused with a gold sentiment label. A later supervised stage can train a stance/sentiment model with time- and community-held-out evaluation.

## Recommended labeled benchmark: Brexit Opinion Dynamics

For defensible accuracy figures, use the [Brexit Opinion Dynamics dataset](https://github.com/behavioral-ds/brexit-opinion-dynamics). Its public benchmark contains 5,024 human-annotated Reddit submissions with anti-Brexit (`-1`), neither (`0`), and pro-Brexit (`1`) labels, and the repository also provides the full 809,713-entry period/interaction metadata corpus. The class imbalance is substantial, so report macro-F1 and per-class recall alongside accuracy.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv_sys\Scripts\python.exe -m reddit_rag.experiment_cli `
  --csv .\data\brexit\rbrexit_stance_annotations.csv.xz `
  --output outputs\brexit_experiment `
  --epochs 8 `
  --skip-retrieval
```

## Intended use cases

The dataset is political discussion from the 2019 Indian election context, not a gaming corpus. Good queries include:

- “What is the overall sentiment toward Narendra Modi?”
- “Retrieve negative comments for a human analyst to review.”
- “Which discussion communities are predominantly positive, neutral, or negative?”
- “Show representative comments supporting or criticizing a topic.”

The system is decision support: retrieved comments and community summaries should be inspected by a human. It is not a causal opinion poll, and the labels should not be interpreted as population-level public opinion.

For the research-grade, unsolved problem this project should target, see [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md). The current Kaggle CSV is retained as a supervised baseline, not as sufficient evidence for graph or community claims.

## LangChain RAG query

The local RAG path uses LangChain `Document` objects, a custom `BaseRetriever`, BM25 + SPLADE reciprocal-rank fusion, and a runnable answer chain. Without an LLM key it emits a deterministic extractive evidence report; with `OPENAI_API_KEY` and `langchain-openai` installed it uses the same retrieved context for cited generation. Install the optional provider with `python -m pip install -e ".[llm]"`.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv_sys\Scripts\python.exe -m reddit_rag.rag_cli `
  --csv .\data\Reddit_Data.csv `
  --query "What evidence is positive or negative about Narendra Modi?" `
  --limit 100 `
  --top-k 5
```

## Local setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

If the machine-wide Python installation has broken script launchers, use an isolated environment that can reuse already-installed ML wheels, then run with the source directory on `PYTHONPATH`:

```powershell
py -3.12 -m venv --system-site-packages .venv_sys
.\.venv_sys\Scripts\python.exe -m pip install --no-deps qdrant-client langchain-core python-dotenv kagglehub grpcio portalocker pydantic pydantic-core annotated-types typing-inspection
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
```

The first SPLADE run downloads the Hugging Face model. If model download is unavailable, the command stops with an actionable error.

## Run the minimal end-to-end slice

This requires no Kaggle account or dataset because it uses an explicit five-row fixture:

```powershell
python -m reddit_rag.cli --demo --limit 5 --query "What do Reddit users say about games?"
```

The output is JSON containing the SPLADE model, Qdrant collection, communities, retrieved evidence, and label distribution.

## Run against Kaggle data (baseline only)

The selected source is [cosmos98/twitter-and-reddit-sentimental-analysis-dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset). Its Reddit file is `Reddit_Data.csv`, containing roughly 37K cleaned comments with labels `-1` (negative), `0` (neutral), and `1` (positive). Obtain access to the Kaggle dataset, then either download its CSV and pass the path:

```powershell
python -m reddit_rag.cli --csv .\data\twitter-reddit-sentiment.csv --reddit-only --limit 500
```

or use Kaggle's API after configuring credentials:

```powershell
python -m reddit_rag.cli --kaggle-dataset cosmos98/twitter-and-reddit-sentimental-analysis-dataset --reddit-only --limit 500
```

The loader searches common text/label columns and reports the detected schema. If the dataset is not available or the Kaggle identifier is omitted, it fails rather than inventing a substitute.

## Scale-up

Start with `--limit 500`, then increase to the full Reddit subset. Use `--qdrant-path .\qdrant_storage` for a persistent local Qdrant store or set `QDRANT_URL` and `QDRANT_API_KEY` for a server. Community construction is O(n²) in this minimal implementation; for the full dataset, shard ingestion and replace the all-pairs graph with top-k neighbor edges before running Louvain.

## Train and evaluate

The experiment command trains a leakage-safe 80/16/4 train/validation/test sentiment classifier, records train/validation log loss by epoch, and evaluates BM25 and SPLADE retrieval on a separate sample. It writes `metrics.json`, `training_loss.png`, `confusion_matrix.png`, and `label_distribution.png`.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv_sys\Scripts\python.exe -m reddit_rag.experiment_cli `
  --kaggle-dataset cosmos98/twitter-and-reddit-sentimental-analysis-dataset `
  --output outputs/reddit_experiment `
  --epochs 8 `
  --retrieval-limit 300 `
  --queries 30
```

Classification metrics are accuracy, macro/weighted F1, precision, recall, and per-class scores. Retrieval metrics are Recall@k, MRR@k, and nDCG@k. Retrieval relevance is explicitly defined as comments sharing the query comment's sentiment label; this is useful for the sentiment-retrieval use case but should not be mistaken for human relevance judgments.
