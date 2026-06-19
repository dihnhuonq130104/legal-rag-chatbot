"""Compare dense, hybrid, and hybrid-plus-reranker retrieval on labelled data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics import evaluate_ranking, mean_metrics
from retriever import LegalRetriever


def load_records(path: str, include_silver: bool) -> list[dict]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if include_silver:
        return records
    return [record for record in records if record.get("review_status") == "approved"]


def rerank(query: str, results: list[dict], model) -> list[dict]:
    scores = model.predict([[query, result["text"]] for result in results])
    return [result for _, result in sorted(zip(scores, results), key=lambda pair: pair[0], reverse=True)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/data/retrieval_gold.json")
    parser.add_argument("--output", default="evaluation/results/retrieval_report.json")
    parser.add_argument("--db-path", default="./qdrant_db", help="Qdrant local storage path; use a copy if the app is running.")
    parser.add_argument("--include-silver", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--limit", type=int, help="Run only the first N records for a quick smoke check.")
    args = parser.parse_args()

    records = load_records(args.dataset, args.include_silver)
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise SystemExit("No approved records. Review retrieval_silver.json and save it as retrieval_gold.json.")

    systems = {
        "dense": None,
        "hybrid_rrf": None,
    }
    model = None
    if not args.skip_reranker:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("BAAI/bge-reranker-v2-m3")

    retriever = LegalRetriever(db_path=args.db_path)
    systems["dense"] = lambda question: retriever.search(question, top_k=30)
    systems["hybrid_rrf"] = lambda question: retriever.search_hybrid(question, top_k=30)
    if model:
        systems["hybrid_rrf_cross_encoder"] = lambda question: rerank(question, retriever.search_hybrid(question, top_k=30), model)

    report = {"dataset": args.dataset, "questions": len(records), "systems": {}}
    try:
        for name, retrieve in systems.items():
            rows = []
            for record in records:
                metrics = evaluate_ranking(retrieve(record["question"]), set(record["relevant_articles"]))
                rows.append({"id": record["id"], **metrics})
            averages = mean_metrics(rows, ["recall_at_5", "recall_at_10", "reciprocal_rank"])
            report["systems"][name] = {"recall_at_5": averages["recall_at_5"], "recall_at_10": averages["recall_at_10"], "mrr": averages["reciprocal_rank"], "per_question": rows}
    finally:
        retriever.client.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, result in report["systems"].items():
        print(f"{name}: Recall@5={result['recall_at_5']:.1%}, Recall@10={result['recall_at_10']:.1%}, MRR={result['mrr']:.3f}")


if __name__ == "__main__":
    main()
