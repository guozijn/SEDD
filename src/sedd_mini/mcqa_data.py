from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MCQARecord:
    prompt: str
    answer: str
    labels: list[str]
    source_id: str = ""

    @property
    def response(self) -> str:
        return f"Answer: {self.answer}"


def normalize_label(label: str) -> str:
    return str(label).strip().upper()


def format_arc_prompt(question: str, choices: dict[str, list[str]]) -> tuple[str, list[str]]:
    labels = [normalize_label(label) for label in choices.get("label", [])]
    texts = [str(text).strip() for text in choices.get("text", [])]
    option_lines = [f"{label}. {text}" for label, text in zip(labels, texts, strict=False)]
    prompt = (
        "Answer the science multiple-choice question. Return only the final choice as "
        "`Answer: <letter>`.\n\n"
        f"Question: {question.strip()}\n"
        "Choices:\n"
        + "\n".join(option_lines)
    )
    return prompt, labels


def load_arc_records(
    *,
    config: str,
    split: str,
    limit: int | None = None,
) -> list[MCQARecord]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("ARC loading requires `uv sync --extra datasets`.") from exc
    rows = load_dataset("allenai/ai2_arc", config, split=split)
    records: list[MCQARecord] = []
    for row in rows:
        prompt, labels = format_arc_prompt(str(row["question"]), row["choices"])
        answer = normalize_label(row["answerKey"])
        if answer not in labels:
            continue
        records.append(
            MCQARecord(
                prompt=prompt,
                answer=answer,
                labels=labels,
                source_id=str(row.get("id") or ""),
            )
        )
        if limit and len(records) >= limit:
            break
    return records


def as_sft_records(records: list[MCQARecord]) -> list[dict[str, str]]:
    return [{"prompt": record.prompt, "response": record.response} for record in records]


def save_mcqa_records(records: list[MCQARecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    {
                        "prompt": record.prompt,
                        "answer": record.answer,
                        "labels": record.labels,
                        "source_id": record.source_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_mcqa_records(path: str | Path) -> list[MCQARecord]:
    records: list[MCQARecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                MCQARecord(
                    prompt=str(row["prompt"]),
                    answer=normalize_label(row["answer"]),
                    labels=[normalize_label(label) for label in row.get("labels", [])],
                    source_id=str(row.get("source_id") or ""),
                )
            )
    return records


def extract_choice(text: str, labels: list[str] | None = None) -> str:
    labels = labels or ["A", "B", "C", "D", "E"]
    normalized = text.strip().upper()
    allowed = "".join(re.escape(label) for label in labels)
    patterns = [
        rf"ANSWER\s*[:：]\s*([{allowed}])\b",
        rf"FINAL\s+ANSWER\s*[:：]\s*([{allowed}])\b",
        rf"\b([{allowed}])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return normalize_label(match.group(1))
    return ""


def exact_choice_reward(text: str, answer: str, labels: list[str] | None = None) -> float:
    predicted = extract_choice(text, labels)
    gold = normalize_label(answer)
    if predicted == gold:
        return 1.0
    if gold and re.search(rf"\b{re.escape(gold)}\b", text.upper()):
        return 0.25
    return 0.0
