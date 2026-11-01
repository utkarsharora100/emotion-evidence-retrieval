from __future__ import annotations

import os
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

from .politosphere import load_politosphere_comments, load_politospere_network
load_politospere_comments = load_politosphere_comments

DEMO_ROWS = [
    ("reddit user says the new game is fun and the community is helpful", "positive"),
    ("the game update is excellent, smooth, and much better than before", "positive"),
    ("I dislike the update; it is buggy and frustrating to use", "negative"),
    ("the moderation is awful and the discussion feels toxic", "negative"),
    ("the subreddit has a useful guide and a friendly discussion", "positive"),
]


@dataclass
class Config:
    model_name: str = os.getenv("SPLADE_MODEL", "naver/splade-cocondenser-ensembledistil")
    collection: str = "reddit_sentiment_demo"
    top_k: int = 3
    graph_algorithm: str = "louvain"


def demo_frame() -> pd.DataFrame:
    return pd.DataFrame(DEMO_ROWS, columns=["text", "sentiment"])


def load_csv(path: str | Path, reddit_only: bool = False, limit: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    lower = {str(c).lower(): c for c in frame.columns}
    text_col = next((lower[c] for c in ("text", "tweet", "comment", "content", "sentence", "clean_comment") if c in lower), None)
    label_col = next((lower[c] for c in ("sentiment", "label", "target", "class", "category") if c in lower), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text/label columns. Columns found: {list(frame.columns)}")
    if reddit_only:
        source_col = next((lower[c] for c in ("source", "platform", "dataset", "type") if c in lower), None)
        if source_col:
            frame = frame[frame[source_col].astype(str).str.contains("reddit", case=False, na=False)]
    frame = frame[[text_col, label_col]].rename(columns={text_col: "text", label_col: "sentiment"})
    frame = frame.dropna().astype({"text": str, "sentiment": str})
    return frame.head(limit) if limit else frame


def load_kaggle(dataset: str, reddit_only: bool, limit: int | None) -> pd.DataFrame:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Kaggle loading requested but kagglehub is not installed. Install the optional Kaggle dependency.") from exc
    try:
        root = Path(kagglehub.dataset_download(dataset))
    except Exception as exc:
        raise RuntimeError("Kaggle download failed. Configure Kaggle credentials and confirm dataset access.") from exc
    csvs = list(root.rglob("*.csv"))
    if reddit_only:
        preferred = [p for p in csvs if "reddit" in p.name.lower()]
        if preferred:
            csvs = preferred + [p for p in csvs if p not in preferred]
    if not csvs:
        raise FileNotFoundError(f"Kaggle dataset downloaded to {root}, but no CSV file was found")
    errors = []
    for csv in csvs:
        try:
            return load_csv(csv, reddit_only=reddit_only, limit=limit)
        except ValueError as exc:
            errors.append(f"{csv.name}: {exc}")
    raise ValueError("No downloaded CSV matched the expected schema. " + " | ".join(errors))


class SpladeEncoder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(f"SPLADE model '{model_name}' could not be loaded. Check Hugging Face connectivity/cache; no BM25 fallback is enabled.") from exc

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 16) -> list[dict[int, float]]:
        encoded = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True, return_tensors="pt")
            logits = self.model(**batch).logits
            weights = torch.log1p(torch.relu(logits)) * batch["attention_mask"].unsqueeze(-1)
            pooled = torch.max(weights, dim=1).values
            encoded.extend([{int(i): float(v) for i, v in enumerate(row) if float(v) > 0} for row in pooled])
        return encoded


def sparse_similarity(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a).intersection(b)
    dot = sum(a[i] * b[i] for i in common)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def bm25_scores(query: str, texts: list[str]) -> list[float]:
    """Small dependency-free BM25 implementation for transparent lexical recall."""
    tokenized = [t.lower().split() for t in texts]
    q = query.lower().split()
    n, avgdl, k1, b = len(tokenized), sum(map(len, tokenized)) / max(len(tokenized), 1), 1.5, 0.75
    df = Counter(tok for doc in tokenized for tok in set(doc))
    scores = []
    for doc in tokenized:
        dl = len(doc); counts = Counter(doc); score = 0.0
        for term in q:
            if not counts[term]:
                continue
            idf = max(0.0, np.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1))
            score += idf * counts[term] * (k1 + 1) / (counts[term] + k1 * (1 - b + b * dl / max(avgdl, 1)))
        scores.append(float(score))
    return scores


def run_pipeline(frame: pd.DataFrame, query: str, config: Config, qdrant_path: str | None = None) -> dict[str, Any]:
    load_dotenv()
    encoder = SpladeEncoder(config.model_name)
    texts = frame["text"].tolist()
    vectors = encoder.encode(texts)
    client = QdrantClient(path=qdrant_path) if qdrant_path else QdrantClient(":memory:")
    name = config.collection + "_" + uuid.uuid4().hex[:8]
    client.create_collection(name, vectors_config={}, sparse_vectors_config={"splade": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    points = [models.PointStruct(id=i, vector={"splade": models.SparseVector(indices=list(v), values=list(v.values()))}, payload={"text": t, "sentiment": s}) for i, (t, s, v) in enumerate(zip(texts, frame["sentiment"], vectors))]
    client.upsert(name, points)
    qvec = encoder.encode([query])[0]
    splade_hits = client.query_points(collection_name=name, query=models.SparseVector(indices=list(qvec), values=list(qvec.values())), using="splade", limit=len(texts), with_payload=True).points
    bm25 = bm25_scores(query, texts)
    # Reciprocal-rank fusion keeps lexical and SPLADE evidence complementary.
    splade_rank = {int(p.id): rank for rank, p in enumerate(splade_hits)}
    bm25_rank = {i: rank for rank, i in enumerate(sorted(range(len(texts)), key=lambda i: bm25[i], reverse=True))}
    fused = sorted(range(len(texts)), key=lambda i: 1 / (60 + splade_rank.get(i, len(texts))) + 1 / (60 + bm25_rank.get(i, len(texts))), reverse=True)[:config.top_k]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(vectors)))
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            score = sparse_similarity(vectors[i], vectors[j])
            if score >= 0.15:
                graph.add_edge(i, j, weight=score)
    communities = list(nx.community.louvain_communities(graph, weight="weight", seed=42)) if graph.number_of_edges() else [{i} for i in graph.nodes]
    docs = [Document(page_content=texts[i], metadata={"sentiment": frame.iloc[i]["sentiment"], "point_id": i, "bm25_rank": bm25_rank[i], "splade_rank": splade_rank.get(i)}) for i in fused]
    return {"rows": len(frame), "retriever": "hybrid BM25 + SPLADE via reciprocal-rank fusion", "model": config.model_name, "qdrant": "local in-memory", "graph": "NetworkX Louvain", "communities": [sorted(c) for c in communities], "evidence": [{"text": d.page_content, **d.metadata} for d in docs], "sentiment_distribution": dict(Counter(d.metadata["sentiment"] for d in docs))}
