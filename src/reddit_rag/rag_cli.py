from __future__ import annotations

import argparse
import json
import os

from .pipeline import Config, load_csv, load_politospere_comments
from .rag import HybridRedditRetriever, build_rag_chain, documents_from_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangChain Reddit hybrid RAG pipeline.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv")
    source.add_argument("--politosphere-comments", help="Path to a Politosphere comments_YYYY-MM.bz2 file.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=Config().model_name)
    args = parser.parse_args()
    frame = load_csv(args.csv, reddit_only=True, limit=args.limit) if args.csv else load_politospere_comments(args.politosphere_comments, limit=args.limit)
    retriever = HybridRedditRetriever(documents_from_frame(frame), args.model, args.top_k)
    llm = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        except ImportError:
            pass
    answer = build_rag_chain(retriever, llm).invoke(args.query)
    print(json.dumps({"mode": "llm" if llm else "extractive", "rows": len(frame), "answer": answer}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
