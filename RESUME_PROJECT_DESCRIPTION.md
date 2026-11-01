# Resume project entry

## Graph-Grounded Reddit Emotion Intelligence RAG

Built an end-to-end NLP and GenAI analytics system for discovering emotional themes and disagreement patterns in Reddit conversations, designed as a proxy for customer-interaction and complaint analytics.

- Processed 58K human-annotated Reddit comments with subreddit, author pseudonym, timestamp, parent-comment, and link metadata; represented reply structure as a graph for context-aware retrieval.
- Implemented a LangChain RAG pipeline with hybrid BM25 + SPLADE sparse retrieval, Qdrant indexing, community detection, and evidence-cited extractive answers; preserved source, time, and conversation context for analyst review.
- Built a leakage-safe stratified benchmark with train/validation/test separation, minority-class balancing, confusion matrices, loss curves, and macro-F1 reporting.
- Best verified derived polarity benchmark: **84.94% accuracy and 83.12% macro-F1** on 5,491 held-out comments; reported class-aware metrics because Reddit emotion labels are ambiguous and imbalanced.
- Added reproducible local execution, dataset loaders, model evaluation, graph statistics, and notebook-based visual analytics.

### One-line version

Built a graph-grounded LangChain RAG system for Reddit emotion and conversation intelligence using SPLADE, Qdrant, Louvain communities, and evidence-cited retrieval; achieved 84.94% accuracy / 83.12% macro-F1 on a leakage-safe held-out polarity benchmark.

### Kotak relevance

This project demonstrates the same technical pattern as complaint analytics and intelligent search: ingest unstructured interaction data, preserve conversational context, retrieve evidence with metadata filters, summarize only grounded evidence, and expose uncertainty to business users.

### Disclosure

The 84.94% / 83.12% figures are for a documented binary polarity aggregation of GoEmotions labels, not the original 28-class emotion task. The project does not claim a 95% score that was not achieved on a valid held-out benchmark.
