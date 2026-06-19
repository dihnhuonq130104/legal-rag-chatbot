"""Metrics shared by the offline evaluation commands."""

from __future__ import annotations


def article_id(item: dict) -> str:
    metadata = item.get("metadata", item)
    return f"{metadata.get('source_file', '')}::{metadata.get('dieu', '')}"


def evaluate_ranking(results: list[dict], relevant_articles: set[str], cutoffs=(5, 10)) -> dict:
    ranked_ids = [article_id(result) for result in results]
    metrics = {
        f"recall_at_{cutoff}": float(any(item in relevant_articles for item in ranked_ids[:cutoff]))
        for cutoff in cutoffs
    }
    reciprocal_rank = next(
        (1.0 / rank for rank, item in enumerate(ranked_ids, start=1) if item in relevant_articles),
        0.0,
    )
    metrics["reciprocal_rank"] = reciprocal_rank
    return metrics


def mean_metrics(rows: list[dict], metric_names: list[str]) -> dict:
    total = max(len(rows), 1)
    return {name: sum(row[name] for row in rows) / total for name in metric_names}
