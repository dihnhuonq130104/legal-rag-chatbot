"""Optional, independent LLM-as-judge for answer correctness and groundedness."""

from __future__ import annotations

import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI


def _json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Judge did not return a JSON object")
    return json.loads(match.group(0))


class LegalAnswerJudge:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = ChatGoogleGenerativeAI(model=model, temperature=0.0)

    def score(self, question: str, answer: str, reference_answer: str, sources: list[dict]) -> dict:
        source_text = "\n\n".join(
            f"[{source.get('source_file')} - {source.get('dieu')}]\n{source.get('text', '')}"
            for source in sources
        )
        prompt = f'''You are grading a Vietnamese legal RAG answer. Return JSON only:
{{"correctness": 0 or 1, "faithfulness": 0 or 1, "citation_support": 0 or 1, "reason": "brief Vietnamese reason"}}
Correctness means the answer agrees with the reference. Faithfulness means every legal claim is supported by the displayed sources. Citation support means the displayed sources substantively support the answer. Do not reward plausible but unsupported claims.

Question: {question}
Reference: {reference_answer}
Answer: {answer}
Displayed sources: {source_text}'''
        result = _json_object(self.model.invoke(prompt).content)
        return {key: result.get(key) for key in ("correctness", "faithfulness", "citation_support", "reason")}
