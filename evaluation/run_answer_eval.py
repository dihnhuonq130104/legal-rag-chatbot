"""Measure answer/source coverage; correctness and faithfulness remain human labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evaluation.metrics import article_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/data/answer_gold.json")
    parser.add_argument("--output", default="evaluation/results/answer_report.json")
    parser.add_argument("--include-silver", action="store_true")
    parser.add_argument("--judge", action="store_true", help="Use Gemini as a secondary evaluator; this uses API quota.")
    parser.add_argument("--judge-model", default="gemini-2.5-flash")
    parser.add_argument("--db-path", help="Qdrant local storage path; use a copy if the app is running.")
    parser.add_argument("--limit", type=int, help="Run only the first N records for a smoke check.")
    args = parser.parse_args()
    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if not args.include_silver:
        records = [record for record in records if record.get("review_status") == "approved"]
    if not records:
        raise SystemExit("No approved answer records. Review answer_silver.json first.")
    if args.limit:
        records = records[:args.limit]
    if args.db_path:
        os.environ["LEGAL_QDRANT_DB_PATH"] = args.db_path

    from generator import generate_answer
    judge = None
    if args.judge:
        from evaluation.llm_judge import LegalAnswerJudge

        judge = LegalAnswerJudge(args.judge_model)

    rows = []
    for record in records:
        response = generate_answer(record["question"])
        cited = {article_id(source) for source in response["sources"]}
        expected = set(record["supporting_articles"])
        precision = len(cited & expected) / len(cited) if cited else 0.0
        recall = len(cited & expected) / len(expected) if expected else 1.0
        row = {
            "id": record["id"],
            "citation_precision": precision,
            "citation_recall": recall,
            "answer": response["answer"],
            "human_correctness": record.get("human_correctness"),
            "human_faithfulness": record.get("human_faithfulness"),
        }
        if judge:
            row["judge"] = judge.score(record["question"], response["answer"], record["reference_answer"], response["sources"])
        rows.append(row)

    count = max(len(rows), 1)
    report = {
        "questions": len(rows),
        "citation_precision": sum(row["citation_precision"] for row in rows) / count,
        "citation_recall": sum(row["citation_recall"] for row in rows) / count,
        "labelled_correctness": sum(row["human_correctness"] is not None for row in rows),
        "labelled_faithfulness": sum(row["human_faithfulness"] is not None for row in rows),
        "judge_correctness": sum(row.get("judge", {}).get("correctness") or 0 for row in rows) / count if judge else None,
        "judge_faithfulness": sum(row.get("judge", {}).get("faithfulness") or 0 for row in rows) / count if judge else None,
        "per_question": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Citation precision={report['citation_precision']:.1%}; recall={report['citation_recall']:.1%}")


if __name__ == "__main__":
    main()
