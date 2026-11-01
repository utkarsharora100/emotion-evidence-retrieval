from __future__ import annotations

import os
import uuid
from collections import Counter
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from pydantic import PrivateAttr
from qdrant_client import QdrantClient, models

from .pipeline import SpladeEncoder, bm25_scores, sparse_similarity


class HybridRedditRetriever(BaseRetriever):
    """LangChain retriever using BM25 + SPLADE reciprocal-rank fusion."""

    _documents: list[Document] = PrivateAttr()
    _texts: list[str] = PrivateAttr()
    _splade_vectors: list[dict[int, float]] = PrivateAttr()
    _encoder: SpladeEncoder = PrivateAttr()
    _top_k: int = PrivateAttr()
    _qdrant: QdrantClient = PrivateAttr()
    _collection: str = PrivateAttr()

    def __init__(self, documents: list[Document], model_name: str, top_k: int = 5):
        super().__init__()
        self._documents = documents
        self._texts = [d.page_content for d in documents]
        self._encoder = SpladeEncoder(model_name)
        self._splade_vectors = self._encoder.encode(self._texts, batch_size=16)
        self._top_k = top_k
        self._qdrant = QdrantClient(":memory:")
        self._collection = "reddit_rag_" + uuid.uuid4().hex[:8]
        self._qdrant.create_collection(self._collection, vectors_config={}, sparse_vectors_config={"splade": models.SparseVectorParams(modifier=models.Modifier.IDF)})
        self._qdrant.upsert(self._collection, [models.PointStruct(id=i, vector={"splade": models.SparseVector(indices=list(v), values=list(v.values()))}, payload={"doc_id": i}) for i, v in enumerate(self._splade_vectors)])

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None) -> list[Document]:
        lexical = bm25_scores(query, self._texts)
        qvec = self._encoder.encode([query])[0]
        qdrant_hits = self._qdrant.query_points(collection_name=self._collection, query=models.SparseVector(indices=list(qvec), values=list(qvec.values())), using="splade", limit=len(self._texts), with_payload=True).points
        semantic_rank = {int(hit.payload["doc_id"]): rank for rank, hit in enumerate(qdrant_hits)}
        lexical_rank = {i: rank for rank, i in enumerate(sorted(range(len(self._texts)), key=lambda i: lexical[i], reverse=True))}
        semantic_rank.update({i: len(self._texts) for i in range(len(self._texts)) if i not in semantic_rank})
        fused = sorted(range(len(self._texts)), key=lambda i: 1 / (60 + lexical_rank[i]) + 1 / (60 + semantic_rank[i]), reverse=True)
        results = []
        for i in fused[: self._top_k]:
            doc = self._documents[i].copy(deep=True)
            doc.metadata.update({"bm25_rank": lexical_rank[i] + 1, "splade_rank": semantic_rank[i] + 1, "hybrid_score": round(1 / (60 + lexical_rank[i]) + 1 / (60 + semantic_rank[i]), 6)})
            results.append(doc)
        return results


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(f"[Evidence {i}] sentiment={doc.metadata.get('sentiment')} community={doc.metadata.get('community_id')}\n{doc.page_content}" for i, doc in enumerate(documents, 1))


def extractive_answer(inputs: dict[str, Any]) -> str:
    docs: list[Document] = inputs["documents"]
    counts = Counter(str(d.metadata.get("sentiment")) for d in docs)
    lines = [f"Question: {inputs['question']}", f"Retrieved sentiment mix: {dict(counts)}", "", "Evidence:"]
    for i, doc in enumerate(docs, 1):
        lines.append(f"[{i}] ({doc.metadata.get('sentiment')}, community {doc.metadata.get('community_id')}) {doc.page_content}")
    lines.append("\nThis is an extractive evidence report; it does not infer population-level opinion.")
    return "\n".join(lines)


def build_rag_chain(retriever: HybridRedditRetriever, llm: Any | None = None):
    retrieved = RunnableParallel(documents=retriever, question=RunnablePassthrough())
    if llm is None:
        return retrieved | RunnableLambda(extractive_answer)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You analyze political Reddit evidence. Answer only from supplied evidence. Preserve disagreement, cite evidence as [1], [2], and state uncertainty. Do not claim the sample represents public opinion."),
        ("human", "Question: {question}\n\nEvidence:\n{context}"),
    ])
    generation = {"question": RunnablePassthrough(), "context": retriever | RunnableLambda(format_context)} | prompt | llm | StrOutputParser()
    return generation


def documents_from_frame(frame) -> list[Document]:
    documents = []
    for i, row in frame.iterrows():
        metadata = {"doc_id": str(getattr(row, "doc_id", i)), "sentiment": str(getattr(row, "sentiment", "unlabeled")), "community_id": str(getattr(row, "subreddit", "inferred")), "source": str(getattr(row, "source", "Reddit_Data.csv"))}
        for field in ("author", "subreddit", "created_utc", "parent_id", "link_id", "score", "language"):
            if hasattr(row, field):
                value = getattr(row, field)
                if value == value:
                    metadata[field] = value
        documents.append(Document(page_content=str(row.text), metadata=metadata))
    return documents
