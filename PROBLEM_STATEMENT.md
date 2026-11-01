# Proposed research problem

## Graph-grounded emotion evidence retrieval for Reddit communities

Given a political entity or topic, retrieve and summarize evidence from multiple Reddit communities and time windows while preserving disagreement, detecting sentiment/stance shifts, and reporting uncertainty.

### Why this is a real gap

The current Kaggle dataset is useful for a sentiment-classification baseline, but it contains only cleaned comments and labels. It cannot answer questions about who participated, which communities disagreed, how opinions changed, or whether a retrieved comment is actually relevant to a query.

Existing emotion classifiers predict labels for isolated comments, while ordinary RAG systems retrieve text without modeling the reply structure that gives a comment its conversational meaning. The open engineering gap is a graph-grounded RAG system that retrieves emotionally relevant evidence together with its parent comment, reply neighborhood, subreddit, author pseudonym, and time context, while preserving disagreement and uncertainty.

### Primary research question

Does a temporal, opinion-aware graph retriever find more diverse, relevant, and faithful evidence across ideological communities than BM25, dense retrieval, SPLADE, and standard GraphRAG baselines?

### Dataset plan

Use GoEmotions as the labeled primary corpus. It contains human-annotated Reddit comments plus subreddit, author, timestamp, parent, and link identifiers, allowing a reply/community graph to be built without inventing graph edges. Keep Politosphere as an optional scale-up network corpus and Brexit Opinion Dynamics as a political-stance comparison benchmark. Create queries such as: “What emotions and counter-emotions appear around this topic in this time window?” The answer must cite comments and their graph neighborhoods, separate direct evidence from inferred community patterns, and expose uncertainty.

### System

1. Build a heterogeneous temporal graph: comment, author, subreddit, topic/entity, and reply/mention edges.
2. Train or adapt a sentiment/stance model with leave-community-out and leave-time-out splits.
3. Retrieve lexical, dense, and learned-sparse candidates.
4. Rerank with graph proximity, stance diversity, time coverage, and evidence quality.
5. Generate an answer only from the selected evidence, showing source community, time, stance, and uncertainty.

### Evaluation

- Classification: macro-F1, per-class recall, AUROC, calibration error, and temporal/community transfer gap.
- Retrieval: human-judged Recall@k, MRR, nDCG, evidence diversity, and cross-community coverage.
- Graph: modularity, community stability over time, stance purity, and cross-community edge rate.
- Answer quality: citation precision, unsupported-claim rate, and whether opposing evidence is preserved.

The contribution is not another sentiment classifier. It is a reproducible benchmark and retrieval system for finding disagreement and opinion change across political Reddit communities without collapsing everything into one global sentiment score.
