"""Evaluate whether cross-reference questions invoke the lookup tool and use its source."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evaluation.metrics import article_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/data/agent_gold.json")
    parser.add_argument("--output", default="evaluation/results/agent_report.json")
    parser.add_argument("--include-silver", action="store_true")
    parser.add_argument("--db-path", help="Qdrant local storage path; use a copy if the app is running.")
    parser.add_argument("--limit", type=int, help="Run only the first N records for a smoke check.")
    args = parser.parse_args()
    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if not args.include_silver:
        records = [record for record in records if record.get("review_status") == "approved"]
    if not records:
        raise SystemExit("No approved agent records. Review agent_silver.json first.")
    if args.limit:
        records = records[:args.limit]
    if args.db_path:
        os.environ["LEGAL_QDRANT_DB_PATH"] = args.db_path
    from generator import generate_answer

    rows = []
    for record in records:
        response = generate_answer(record["question"])
        expected = set(record.get("required_articles", []))
        sources = response["sources"]
        used_expected_source = bool({article_id(source) for source in sources} & expected) if expected else True
        used_expected_doc_type = any(source.get("doc_type") == record.get("required_doc_type") for source in sources)
        tool_called = bool(response.get("trace", {}).get("tool_calls"))
        expected_tool = record.get("expect_tool", True)
        success = tool_called == expected_tool and used_expected_source and (used_expected_doc_type if expected_tool else True)
        rows.append({"id": record["id"], "tool_called": tool_called, "used_expected_source": used_expected_source, "used_expected_doc_type": used_expected_doc_type, "success": success})

    count = max(len(rows), 1)
    cross_rows = [row for row, record in zip(rows, records) if record.get("kind") == "cross_reference"]
    direct_rows = [row for row, record in zip(rows, records) if record.get("kind") == "direct"]
    cross_count = max(len(cross_rows), 1)
    direct_count = max(len(direct_rows), 1)
    report = {
        "questions": len(rows),
        "tool_call_rate": sum(row["tool_called"] for row in rows) / count,
        "cross_reference_tool_call_rate": sum(row["tool_called"] for row in cross_rows) / cross_count,
        "expected_document_type_success_rate": sum(row["success"] for row in cross_rows) / cross_count,
        "direct_no_tool_rate": sum(not row["tool_called"] for row in direct_rows) / direct_count,
        "article_level_records": sum(bool(record.get("required_articles")) for record in records),
        "per_question": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Cross-reference tool call={report['cross_reference_tool_call_rate']:.1%}; expected document type success={report['expected_document_type_success_rate']:.1%}; direct no-tool={report['direct_no_tool_rate']:.1%}")


if __name__ == "__main__":
    main()
