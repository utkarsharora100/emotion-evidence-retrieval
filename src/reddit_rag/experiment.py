from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             log_loss, normalized_mutual_info_score, adjusted_rand_score,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .pipeline import bm25_scores, load_csv, load_kaggle, SpladeEncoder, sparse_similarity


def _save_confusion(cm: np.ndarray, labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="Actual", title="Reddit sentiment confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _save_loss(history: list[dict[str, float]], path: Path) -> None:
    epochs = [x["epoch"] for x in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [x["train_loss"] for x in history], marker="o", label="train log loss")
    ax.plot(epochs, [x["validation_loss"] for x in history], marker="o", label="validation log loss")
    ax.set(xlabel="Epoch", ylabel="Log loss", title="Sentiment model training curve"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _save_distribution(frame: pd.DataFrame, path: Path) -> None:
    counts = frame["sentiment"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4)); counts.plot.bar(ax=ax, color=["#d95f02", "#7570b3", "#1b9e77"])
    ax.set(xlabel="Sentiment label", ylabel="Comments", title="Reddit label distribution"); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def train_classifier(frame: pd.DataFrame, out: Path, epochs: int = 8) -> dict[str, Any]:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    frame = frame[["text", "sentiment"]].dropna().astype(str).reset_index(drop=True)
    train, test = train_test_split(frame, test_size=.2, random_state=42, stratify=frame["sentiment"])
    train, valid = train_test_split(train, test_size=.2, random_state=42, stratify=train["sentiment"])
    # Balance only the training partition. Validation and test retain the real
    # distribution, so macro metrics remain honest and comparable.
    per_class = train["sentiment"].value_counts()
    target_count = int(per_class.max())
    train = pd.concat([
        group.sample(target_count, replace=len(group) < target_count, random_state=42)
        for _, group in train.groupby("sentiment", sort=False)
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=120_000)
    x_train = vectorizer.fit_transform(train.text); x_valid = vectorizer.transform(valid.text); x_test = vectorizer.transform(test.text)
    le = LabelEncoder(); y_train = le.fit_transform(train.sentiment); y_valid = le.transform(valid.sentiment); y_test = le.transform(test.sentiment)
    model = SGDClassifier(loss="log_loss", alpha=2e-5, penalty="l2", random_state=42, learning_rate="optimal")
    history = []
    for epoch in range(1, epochs + 1):
        model.partial_fit(x_train, y_train, classes=np.arange(len(le.classes_)))
        history.append({"epoch": epoch, "train_loss": float(log_loss(y_train, model.predict_proba(x_train), labels=np.arange(len(le.classes_)))), "validation_loss": float(log_loss(y_valid, model.predict_proba(x_valid), labels=np.arange(len(le.classes_))))})
    pred = model.predict(x_test)
    metrics = {"accuracy": float(accuracy_score(y_test, pred)), "macro_f1": float(f1_score(y_test, pred, average="macro")), "weighted_f1": float(f1_score(y_test, pred, average="weighted")), "macro_precision": float(precision_score(y_test, pred, average="macro")), "macro_recall": float(recall_score(y_test, pred, average="macro")), "classification_report": classification_report(y_test, pred, target_names=[str(x) for x in le.classes_], output_dict=True), "split_sizes": {"train": len(train), "validation": len(valid), "test": len(test)}, "classes": [str(x) for x in le.classes_], "training_history": history}
    _save_confusion(confusion_matrix(y_test, pred), [str(x) for x in le.classes_], out / "confusion_matrix.png")
    _save_loss(history, out / "training_loss.png")
    _save_distribution(frame, out / "label_distribution.png")
    return metrics


def retrieval_benchmark(frame: pd.DataFrame, out: Path, limit: int = 1000, queries: int = 100) -> dict[str, Any]:
    data = frame.sample(min(limit, len(frame)), random_state=42).reset_index(drop=True)
    query_frame = data.sample(min(queries, len(data) // 5), random_state=7)
    data = data.drop(query_frame.index).reset_index(drop=True)
    query_frame = query_frame.reset_index(drop=True)
    texts = data.text.tolist(); labels = data.sentiment.astype(str).tolist()
    bm25_results = []
    for q in query_frame.text:
        scores = bm25_scores(q, texts); bm25_results.append(np.argsort(scores)[::-1].tolist())
    encoder = SpladeEncoder("naver/splade-cocondenser-ensembledistil")
    doc_vecs = encoder.encode(texts, batch_size=16); query_vecs = encoder.encode(query_frame.text.tolist(), batch_size=16)
    splade_results = []
    for qv in query_vecs:
        scores = [sparse_similarity(qv, dv) for dv in doc_vecs]; splade_results.append(np.argsort(scores)[::-1].tolist())
    def score(results: list[list[int]], name: str) -> dict[str, float]:
        ks = [1, 5, 10]; outm = {}
        for k in ks:
            hits = []; rr = []; ndcg = []
            for qi, ranking in enumerate(results):
                relevant = {i for i, label in enumerate(labels) if label == str(query_frame.iloc[qi].sentiment)}
                top = ranking[:k]; hits.append(bool(set(top) & relevant)); rr.append(next((1 / (rank + 1) for rank, item in enumerate(top) if item in relevant), 0.0))
                gains = [1 if item in relevant else 0 for item in top]; dcg = sum(g / np.log2(j + 2) for j, g in enumerate(gains)); ideal = sum(1 / np.log2(j + 2) for j in range(min(k, len(relevant)))); ndcg.append(dcg / ideal if ideal else 0.0)
            outm[f"recall@{k}"] = float(np.mean(hits)); outm[f"mrr@{k}"] = float(np.mean([x if x <= 1 else 1 for x in rr])); outm[f"ndcg@{k}"] = float(np.mean(ndcg))
        return outm
    return {"corpus_size": len(data), "query_count": len(query_frame), "relevance_definition": "documents sharing the query comment's sentiment label", "bm25": score(bm25_results, "bm25"), "splade": score(splade_results, "splade")}


def run_experiment(frame: pd.DataFrame, output_dir: str, epochs: int = 8, retrieval_limit: int = 1000, queries: int = 100, include_retrieval: bool = True) -> dict[str, Any]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    frame = frame[["text", "sentiment"]].dropna().astype(str)
    result = {"dataset_rows": len(frame), "classification": train_classifier(frame, out, epochs), "retrieval": retrieval_benchmark(frame, out, retrieval_limit, queries) if include_retrieval else {"skipped": True, "reason": "classification-only run"}}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
