# Emotion Evidence Retrieval

This project explores how to find useful evidence in large Reddit discussions.

It combines emotion classification with search and graph context. An analyst can find comments about a topic, compare communities, and inspect the replies around each result. The system also keeps the source comments, so an answer can be checked.

This pattern is useful for complaint analysis, customer research, search, and knowledge discovery. It can also support a RAG system that needs to show where its answers came from.

## Benchmark result

The benchmark notebook uses 58,009 unique, human-rated Reddit comments from GoEmotions. It groups clear majority labels into two classes: positive and negative.

The test set contains 5,491 comments and uses a stratified 80/20 split. The result is:

- Accuracy: **84.94%**
- Macro-F1: **83.12%**
- Weighted-F1: **84.84%**

These numbers are for the binary task. They are not scores for all 28 GoEmotions classes. The project keeps that distinction clear because the evaluation should be easy to understand and reproduce.

To reproduce the result, run [notebooks/reddit_graph_rag_showcase.ipynb](notebooks/reddit_graph_rag_showcase.ipynb). The charts and metrics are in `outputs/notebook_showcase/`. The resume version is in [RESUME_PROJECT_DESCRIPTION.md](RESUME_PROJECT_DESCRIPTION.md).

## The problem

A sentiment model can label a single comment, but that label does not explain the wider discussion. A text retriever can find a matching comment, but it may miss the reply that gives the comment its meaning.

This project joins those two views. It retrieves the comment and, when available, its parent, reply link, subreddit, author ID, and time. It then uses that context to give the analyst a more useful answer.

Typical questions include:

- What are people feeling about a topic?
- Which communities agree or disagree?
- How did the discussion change over time?
- Which comments should a person review?
- What source comments support the answer?

## Pipeline

The pipeline works in a few steps:

1. Load Reddit comments and their metadata.
2. Predict positive or negative emotion.
3. Retrieve text with BM25 and SPLADE.
4. Store sparse vectors in Qdrant.
5. Build a reply graph and find communities with Louvain.
6. Pass the retrieved evidence to a LangChain RAG chain.

BM25 handles direct word matches. SPLADE adds learned sparse matches. Qdrant stores the vectors. LangChain connects retrieval to the answer step. If SPLADE cannot load, the program stops instead of silently replacing it with BM25.

## Data

For the graph data, I use the [Reddit Politosphere release](https://zenodo.org/records/5851729). Its [scripts and notes](https://github.com/valentinhofmann/politosphere) explain the file layout.

The useful part is the conversation structure. Comments can be linked to their parent, subreddit, time, and masked author ID. That lets the system look at the discussion around a comment, not just the comment itself.

The full release is large, so start with one monthly file. Put `comments_YYYY-MM.bz2` in `data\politosphere`, then run:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv_sys\Scripts\python.exe -m reddit_rag.rag_cli `
  --politosphere-comments .\data\politosphere\comments_2008-01.bz2 `
  --query "What political criticism appears in the discussion?" `
  --limit 100 `
  --top-k 5
```

This graph corpus does not contain gold sentiment labels. The loader keeps sentiment as `unlabeled`, so the pipeline does not create labels that are not in the data.

## Labeled data

For a labeled comparison, I use the [Brexit Opinion Dynamics dataset](https://github.com/behavioral-ds/brexit-opinion-dynamics). It contains 5,024 Reddit submissions labeled as anti-Brexit, neither, or pro-Brexit.

The classes are uneven, so accuracy alone is not enough. The experiment also reports macro-F1 and recall for each class.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv_sys\Scripts\python.exe -m reddit_rag.experiment_cli `
  --csv .\data\brexit\rbrexit_stance_annotations.csv.xz `
  --output outputs\brexit_experiment `
  --epochs 8 `
  --skip-retrieval
```

## RAG query

The RAG code uses LangChain `Document` objects and a custom retriever. It combines BM25 and SPLADE results before passing them to the answer chain.

Without an LLM key, the chain returns an extractive evidence report. With `OPENAI_API_KEY` and `langchain-openai`, it can produce a cited answer from the same retrieved context.

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

For a quick smoke test, run the built-in five-row fixture:

```powershell
python -m reddit_rag.cli --demo --limit 5 --query "What do Reddit users say about games?"
```

The fixture does not need a Kaggle account. It checks that the main path works.

## Kaggle baseline

The original baseline uses the [Twitter and Reddit Sentimental Analysis dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset). Its Reddit file, `Reddit_Data.csv`, contains about 37K cleaned comments with negative, neutral, and positive labels.

After configuring Kaggle access, run:

```powershell
python -m reddit_rag.cli --kaggle-dataset cosmos98/twitter-and-reddit-sentimental-analysis-dataset --reddit-only --limit 500
```

The loader checks the text and label columns. If the data is missing, it fails clearly rather than inventing a replacement.

## Scaling and evaluation

Start with `--limit 500` and increase the limit after the small run works. Use `--qdrant-path .\qdrant_storage` for local storage. For a server, set `QDRANT_URL` and `QDRANT_API_KEY`.

The simple graph code is slow on very large data. For a full run, use shards and top-k graph edges before running Louvain.

The experiment writes metrics, loss charts, confusion matrices, and label charts. It reports accuracy, F1, precision, recall, Recall@k, MRR@k, and nDCG@k.

The full problem description is in [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md).
