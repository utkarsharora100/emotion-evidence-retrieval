# Resume project entry

## Graph-Grounded Reddit Emotion Intelligence RAG

Built an end-to-end NLP and GenAI analytics system for discovering emotional themes and disagreement patterns in Reddit conversations, designed as a proxy for customer-interaction and complaint analytics.

- Processed 58K human-annotated Reddit comments with subreddit, author pseudonym, timestamp, parent-comment, and link metadata; retained all 28 GoEmotions labels and represented reply structure as a graph for context-aware retrieval.
- Implemented a LangChain RAG pipeline with hybrid BM25 + SPLADE sparse retrieval, Qdrant indexing, and a custom graph-aware reranker that rewards community coverage and nearby reply context before generating evidence-cited answers.
- Built a leakage-safe stratified 28-class benchmark with 38,800 training comments and 9,701 held-out comments, confusion matrices, class distributions, and macro-F1 reporting.
- Full-label benchmark: **55.10% accuracy, 40.58% macro-F1, and 52.39% weighted-F1** on 9,701 held-out comments; reported per-class scores because several emotions are rare and easy to confuse.
- Added reproducible local execution, dataset loaders, model evaluation, graph statistics, and notebook-based visual analytics.

### One-line version

Built a graph-grounded LangChain RAG system for Reddit emotion and conversation intelligence using SPLADE, Qdrant, Louvain communities, and evidence-cited retrieval; evaluated all 28 GoEmotions labels with 55.10% accuracy and 40.58% macro-F1 on held-out comments.

### Kotak relevance

This project demonstrates the same technical pattern as complaint analytics and intelligent search: ingest unstructured interaction data, preserve conversational context, retrieve evidence with metadata filters, summarize only grounded evidence, and expose uncertainty to business users.

### Disclosure

The reported headline result is the full 28-class GoEmotions majority-label benchmark. A separate binary polarity analysis exists, but it is not used as the main project result.
