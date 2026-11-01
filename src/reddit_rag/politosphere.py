from __future__ import annotations

import bz2
import csv
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


TEXT_FIELDS = ("body_cleaned", "body", "text")


def _records(path: str | Path) -> Iterable[dict]:
    """Yield records from a Politosphere .bz2 JSON-lines file.

    The upstream files follow the Pushshift JSON record convention. The small
    parser also tolerates a plain JSON-lines file, which is useful for local
    fixtures and decompressed samples.
    """
    path = Path(path)
    opener = bz2.open if path.suffix.lower() == ".bz2" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Be helpful with accidental tab-separated exports whose last
                # column is JSON; the official release itself is JSON-lines.
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                try:
                    record = json.loads(parts[-1])
                except json.JSONDecodeError:
                    continue
            if isinstance(record, dict):
                yield record


def load_politosphere_comments(
    path: str | Path,
    limit: int | None = None,
    subreddits: set[str] | None = None,
    language: str | None = "en",
    start_utc: int | None = None,
    end_utc: int | None = None,
) -> pd.DataFrame:
    """Stream one Politosphere monthly file into the project's normalized schema."""
    rows: list[dict] = []
    wanted = {s.lower().removeprefix("r/") for s in subreddits} if subreddits else None
    for record in _records(path):
        subreddit = str(record.get("subreddit") or "").removeprefix("r/")
        created = record.get("created_utc")
        try:
            created_int = int(float(created)) if created is not None else None
        except (TypeError, ValueError):
            created_int = None
        if wanted and subreddit.lower() not in wanted:
            continue
        if language and record.get("language") and str(record["language"]).lower() != language.lower():
            continue
        if start_utc is not None and (created_int is None or created_int < start_utc):
            continue
        if end_utc is not None and (created_int is None or created_int >= end_utc):
            continue
        text = next((record.get(field) for field in TEXT_FIELDS if record.get(field)), None)
        if not text:
            continue
        rows.append({
            "text": str(text),
            "sentiment": "unlabeled",
            "doc_id": str(record.get("id") or len(rows)),
            "author": str(record.get("author") or "[deleted]"),
            "subreddit": subreddit,
            "created_utc": created_int,
            "parent_id": record.get("parent_id"),
            "link_id": record.get("link_id"),
            "score": record.get("score"),
            "language": record.get("language"),
        })
        if limit and len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def load_politospere_network(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    """Load an official yearly subreddit network file."""
    frame = pd.read_csv(path)
    expected = {"node_1", "node_2", "weighted", "unweighted"}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Network file is missing columns: {sorted(missing)}")
    return frame.head(limit) if limit else frame


# Backward-compatible alias for the initial spelling used in the CLI patch.
load_politospere_comments = load_politosphere_comments
