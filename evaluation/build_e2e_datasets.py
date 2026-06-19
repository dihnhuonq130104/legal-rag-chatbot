"""Derive reviewable answer and agent evaluation records from retrieval labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def article_key(item: dict) -> str:
    metadata = item["metadata"]
    return f"{metadata['source_file']}::{metadata['dieu']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="legal_corpus.json")
    parser.add_argument("--retrieval", default="evaluation/data/retrieval_silver.json")
    parser.add_argument("--answer-output", default="evaluation/data/answer_silver.json")
    parser.add_argument("--agent-output", default="evaluation/data/agent_silver.json")
    parser.add_argument("--answer-size", type=int, default=50)
    parser.add_argument("--agent-size", type=int, default=20)
    args = parser.parse_args()

    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    articles = {}
    for item in corpus:
        articles.setdefault(article_key(item), item)
    retrieval_records = json.loads(Path(args.retrieval).read_text(encoding="utf-8"))

    answer_records = []
    for record in retrieval_records[: args.answer_size]:
        key = record["relevant_articles"][0]
        item = articles[key]
        answer_records.append({
            "id": f"answer-{record['id']}",
            "question": record["question"],
            "supporting_articles": [key],
            "reference_answer": item["text"],
            "review_status": "needs_human_review",
            "human_correctness": None,
            "human_faithfulness": None,
        })

    trigger = "Ch\u00ednh ph\u1ee7 quy \u0111\u1ecbnh chi ti\u1ebft"
    agent_records = []
    for item in articles.values():
        metadata = item["metadata"]
        if metadata.get("doc_type") != "Lu\u1eadt" or trigger.lower() not in item["text"].lower():
            continue
        key = article_key(item)
        agent_records.append({
            "id": f"agent-{len(agent_records) + 1:03d}",
            "kind": "cross_reference",
            "question": f"Theo {metadata['dieu']}, noi dung nao can tra cuu them van ban huong dan?",
            "initial_articles": [key],
            "expect_tool": True,
            "required_doc_type": "Ngh\u1ecb \u0111\u1ecbnh",
            "required_articles": [],
            "review_status": "needs_human_review",
            "notes": "A legal reviewer must add the exact implementing article(s) before publishing article-level success.",
        })
        if len(agent_records) >= args.agent_size:
            break

    # Add direct questions so the agent is also penalized for unnecessary tool calls.
    for record in retrieval_records:
        if len(agent_records) >= args.agent_size:
            break
        agent_records.append({
            "id": f"agent-{len(agent_records) + 1:03d}",
            "kind": "direct",
            "question": record["question"],
            "initial_articles": record["relevant_articles"],
            "expect_tool": False,
            "required_doc_type": None,
            "required_articles": [],
            "review_status": "needs_human_review",
            "notes": "Confirm that this question can be answered from the initial article without a cross-reference.",
        })

    for output, records in ((args.answer_output, answer_records), (args.agent_output, agent_records)):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(records)} records to {path}")


if __name__ == "__main__":
    main()
