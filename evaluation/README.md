# Evaluation protocol

Do not publish metrics from unreviewed, templated questions. The generator creates a stratified **silver** set; a domain reviewer must rewrite questions, confirm every relevant article, and change `review_status` to `approved` in `retrieval_gold.json`.

Each retrieval record has `id`, `question`, and `relevant_articles`. An article key is `source_file::dieu`, so multiple chunks from the same article count as relevant.

```powershell
python -m evaluation.build_retrieval_dataset --size 100
# Review evaluation/data/retrieval_silver.json, then save approved records as retrieval_gold.json
python -m evaluation.run_retrieval_eval
```

The retrieval report compares dense Qdrant, BM25+dense RRF hybrid, and hybrid reranked with `BAAI/bge-reranker-v2-m3`; it writes Recall@5, Recall@10, MRR, and per-question results.

For answer evaluation, create `answer_gold.json` records with `id`, `question`, `supporting_articles`, plus optional 0/1 `human_correctness` and `human_faithfulness`. `python -m evaluation.run_answer_eval` records source precision/recall and preserves the answer for review. Citation precision measures whether displayed source articles support the answer; it is not an LLM self-judgement.

For agent evaluation, create `agent_gold.json` records with `id`, `question`, and `required_articles`. `python -m evaluation.run_agent_eval` reports tool-call rate and cross-reference retrieval success rate (tool called and an expected source used).

## End-to-end preparation

Create answer and agent review sets from the retrieval set:

```powershell
python -m evaluation.build_e2e_datasets
```

`answer_silver.json` contains 50 question/reference/source records. `agent_silver.json` finds law chunks which explicitly require an implementing decree. A legal reviewer must confirm the response reference and add exact `required_articles` to agent records. Keep the records `needs_human_review` until that review is complete, then save approved copies as `answer_gold.json` and `agent_gold.json`.

Run answer and agent evaluation on approved data:

```powershell
python -m evaluation.run_answer_eval
python -m evaluation.run_agent_eval
```

Add `--judge` to answer evaluation to obtain a reproducible Gemini secondary score. It consumes API quota and complements, but does not replace, human legal review.
