# Graph-Grounded Emotion Evidence Retrieval

This project addresses a practical analytics problem: analysts need to find emotionally important discussion themes, compare communities, and review evidence. They need those results because a single sentiment score does not explain why people disagree or how a discussion changes over time. The system therefore combines emotion classification, graph context, hybrid retrieval, and evidence-grounded RAG.

The project is designed as a customer-interaction analytics prototype. The same pattern can support complaint analytics, intelligent search, knowledge discovery, and workflow automation. The system is not presented as a production opinion poll. The system is presented as transparent decision support for an analyst.

## Best verified benchmark result

The showcase notebook uses 58,009 unique human-rated GoEmotions Reddit comments. It creates a documented binary polarity task from comments with an unambiguous majority label. A leakage-safe, stratified 80/20 split produces the following results on 5,491 held-out comments:

- Accuracy: **84.94%**
- Macro-F1: **83.12%**
- Weighted-F1: **84.84%**

These are binary polarity results, and they are not 28-class GoEmotions results. The README states that distinction because the benchmark must remain clear and reproducible. The earlier 95% accuracy and 0.90 macro-F1 target was not reached without changing the task or leaking labels. The project reports the best defensible result instead.

The runnable notebook is [notebooks/reddit_graph_rag_showcase.ipynb](notebooks/reddit_graph_rag_showcase.ipynb). The generated figures and metrics are in `outputs/notebook_showcase/`. The resume-ready wording is in [RESUME_PROJECT_DESCRIPTION.md](RESUME_PROJECT_DESCRIPTION.md).

## The problem and the system

The problem is graph-grounded emotion evidence retrieval for Reddit communities. Given a topic, an analyst should retrieve evidence from several communities and time windows. The analyst should also see disagreement, reply context, community context, and uncertainty. The analyst should not receive a fluent answer that cannot be traced back to source comments.

The system follows that requirement in six connected stages:

1. It loads and normalizes Reddit comments and their available metadata.
2. It predicts emotional polarity for the supervised benchmark.
3. It retrieves candidates with BM25 and SPLADE sparse representations.
4. It stores sparse vectors in Qdrant and fuses the retrieval signals.
5. It builds reply and community relationships, then detects communities with Louvain.
6. It passes LangChain `Document` objects into an evidence-first RAG answer chain.

BM25 provides the lexical signal, while SPLADE provides the learned sparse signal. Qdrant provides the vector store, while LangChain provides the retrieval and answer boundary. The graph provides context, while the evidence report provides traceability. If SPLADE cannot load, the run stops with an actionable error instead of silently substituting another retriever.

## Recommended dataset: Reddit Politosphere

The recommended graph corpus is the [official Zenodo release](https://zenodo.org/records/5851729), together with its [documentation and scripts repository](https://github.com/valentinhofmann/politosphere). It contains political Reddit comments, timestamps, pseudonymized authors, reply identifiers, link identifiers, language metadata, and subreddit metadata. It also contains yearly subreddit-overlap networks.

The complete release is large, so the project starts with one monthly file. After downloading `comments_YYYY-MM.bz2` into `data\politosphere`, run the LangChain RAG slice:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv_sys\Scripts\python.exe -m reddit_rag.rag_cli `
  --politosphere-comments .\data\politosphere\comments_2008-01.bz2 `
  --query "What political criticism appears in the discussion?" `
  --limit 100 `
  --top-k 5
```

The loader streams the compressed file and filters to English comments by default. It preserves the author, subreddit, timestamp, parent, link, score, and language fields. The Politosphere corpus has no gold sentiment label in this path, so its sentiment remains `unlabeled`. That separation protects the evaluation from invented labels.

## Recommended labeled benchmark: Brexit Opinion Dynamics

The [Brexit Opinion Dynamics dataset](https://github.com/behavioral-ds/brexit-opinion-dynamics) provides 5,024 human-annotated Reddit submissions. Its labels represent anti-Brexit, neither, and pro-Brexit positions. The full metadata corpus contains 809,713 entries and interaction information. Because the classes are imbalanced, the experiment reports accuracy, macro-F1, per-class recall, and weighted-F1 together.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv_sys\Scripts\python.exe -m reddit_rag.experiment_cli `
  --csv .\data\brexit\rbrexit_stance_annotations.csv.xz `
  --output outputs\brexit_experiment `
  --epochs 8 `
  --skip-retrieval
```

## Intended use cases

The system is intended for an analyst who needs evidence, not only a label. The analyst can ask which communities are positive, neutral, or negative. The analyst can ask for representative comments that support or criticize a topic. The analyst can ask which emotions and counter-emotions appear during a selected time window. The analyst can also retrieve negative comments for human review.

The system supports those questions, but it does not claim causal public opinion. The system supports review, but it does not replace review. The system reports evidence, but it does not hide uncertainty. Those limits are important when the output informs customer, policy, or operational decisions.

The research framing is documented in [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md). The contribution is not another isolated sentiment classifier. The contribution is a reproducible retrieval system that preserves disagreement, community context, reply context, and time context.

## LangChain RAG query

The local RAG path uses LangChain `Document` objects and a custom `BaseRetriever`. It combines BM25 with SPLADE reciprocal-rank fusion, and it sends the retrieved evidence into a runnable answer chain. Without an LLM key, the chain returns a deterministic extractive evidence report. With `OPENAI_API_KEY` and `langchain-openai`, the chain can generate a cited answer from the same retrieved context.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv_sys\Scripts\python.exe -m reddit_rag.rag_cli `
  --csv .\data\Reddit_Data.csv `
  --query "What evidence is positive or negative about Narendra Modi?" `
  --limit 100 `
  --top-k 5
```

Install the optional provider with `python -m pip install -e ".[llm]"` when cited LLM generation is required.

## Local setup

Create an environment, install the package, and copy the example environment file:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

If the machine-wide Python installation has broken launchers, use the isolated environment below. This environment can reuse installed machine-wide ML wheels while it still installs the project-specific packages.

```powershell
py -3.12 -m venv --system-site-packages .venv_sys
.\.venv_sys\Scripts\python.exe -m pip install --no-deps qdrant-client langchain-core python-dotenv kagglehub grpcio portalocker pydantic pydantic-core annotated-types typing-inspection
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
```

The first SPLADE run downloads its Hugging Face model. If model access is unavailable, the command stops with an actionable error. The system does not silently replace SPLADE with BM25.

## Minimal end-to-end slice

The minimal slice uses a five-row fixture, so it does not need a Kaggle account or an external dataset:

```powershell
python -m reddit_rag.cli --demo --limit 5 --query "What do Reddit users say about games?"
```

The output includes the SPLADE model, the Qdrant collection, detected communities, retrieved evidence, and the label distribution. The minimal slice is for smoke testing. The notebook is for benchmark reporting.

## Kaggle baseline

The original baseline uses the [Twitter and Reddit Sentimental Analysis dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset). Its Reddit file is `Reddit_Data.csv`, with roughly 37K cleaned comments and negative, neutral, and positive labels. Download the data after configuring Kaggle access, then run either command below.

```powershell
python -m reddit_rag.cli --csv .\data\twitter-reddit-sentiment.csv --reddit-only --limit 500
python -m reddit_rag.cli --kaggle-dataset cosmos98/twitter-and-reddit-sentimental-analysis-dataset --reddit-only --limit 500
```

The loader searches common text and label columns and reports the detected schema. If the data is unavailable, the loader fails clearly. It does not invent a substitute dataset.

## Scale-up and evaluation

Start with `--limit 500`, and then increase the limit after the pipeline is stable. Use `--qdrant-path .\qdrant_storage` for a persistent local Qdrant store, or use `QDRANT_URL` and `QDRANT_API_KEY` for a server. The minimal graph implementation is quadratic, so full-scale use requires sharded ingestion and top-k neighbor edges before Louvain.

The experiment command records training and validation loss by epoch. It also writes classification metrics, retrieval metrics, confusion matrices, label distributions, and training-loss graphs. Classification metrics include accuracy, macro-F1, weighted-F1, precision, recall, and per-class scores. Retrieval metrics include Recall@k, MRR@k, and nDCG@k. Retrieval relevance is defined explicitly, because shared sentiment is not the same as human relevance.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv_sys\Scripts\python.exe -m reddit_rag.experiment_cli `
  --kaggle-dataset cosmos98/twitter-and-reddit-sentimental-analysis-dataset `
  --output outputs/reddit_experiment `
  --epochs 8 `
  --retrieval-limit 300 `
  --queries 30
```
