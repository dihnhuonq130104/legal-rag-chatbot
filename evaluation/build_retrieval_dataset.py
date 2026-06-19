"""Create a reviewable silver retrieval set from the parsed legal corpus."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def article_key(item: dict) -> str:
    metadata = item["metadata"]
    return f"{metadata['source_file']}::{metadata['dieu']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stratified, review-required retrieval set.")
    parser.add_argument("--corpus", default="legal_corpus.json")
    parser.add_argument("--output", default="evaluation/data/retrieval_silver.json")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    articles: dict[str, dict] = {}
    for item in corpus:
        articles.setdefault(article_key(item), item)

    by_source: dict[str, list[dict]] = defaultdict(list)
    for item in articles.values():
        by_source[item["metadata"]["source_file"]].append(item)

    randomizer = random.Random(args.seed)
    selected: list[dict] = []
    sources = list(by_source.values())
    while len(selected) < min(args.size, len(articles)):
        made_progress = False
        for source_articles in sources:
            remaining = [item for item in source_articles if item not in selected]
            if remaining and len(selected) < args.size:
                selected.append(randomizer.choice(remaining))
                made_progress = True
        if not made_progress:
            break

    records = []
    for number, item in enumerate(selected, start=1):
        metadata = item["metadata"]
        records.append({
            "id": f"silver-{number:03d}",
            "question": f"Noi dung {metadata['dieu']} ve {metadata['ten_dieu']} la gi?",
            "relevant_articles": [article_key(item)],
            "review_status": "needs_human_review",
            "notes": "Replace the templated question with a natural legal question before reporting results.",
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} review-required records to {output}")


if __name__ == "__main__":
    main()
