from __future__ import annotations

import math
import re
from typing import Any


def keyword_length_reward(text: str, config: dict[str, Any]) -> float:
    keyword = str(config.get("keyword", "")).strip().lower()
    keyword_bonus = float(config.get("keyword_bonus", 1.0))
    min_chars = int(config.get("min_chars", 80))
    max_chars = int(config.get("max_chars", 500))
    length_bonus = float(config.get("length_bonus", 0.5))

    reward = 0.0
    normalized = text.lower()
    if keyword and keyword in normalized:
        reward += keyword_bonus
    length = len(text)
    if min_chars <= length <= max_chars:
        reward += length_bonus
    else:
        distance = min(abs(length - min_chars), abs(length - max_chars))
        reward -= min(1.0, distance / max(min_chars, 1)) * length_bonus
    return reward


def regex_reward(text: str, config: dict[str, Any]) -> float:
    pattern = str(config.get("pattern", ""))
    match_bonus = float(config.get("match_bonus", 1.0))
    miss_penalty = float(config.get("miss_penalty", 0.0))
    return match_bonus if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) else -miss_penalty


def heuristic_helpfulness_reward(text: str, config: dict[str, Any]) -> float:
    """A tiny transparent reward for demo RL when no external reward model is available."""

    reward = 0.0
    stripped = text.strip()
    if len(stripped) >= int(config.get("min_chars", 80)):
        reward += 0.3
    if any(word in stripped.lower() for word in ["because", "therefore", "for example", "first"]):
        reward += 0.4
    if "\n" in stripped or "." in stripped:
        reward += 0.2
    if any(bad in stripped.lower() for bad in ["<mask>", "�"]):
        reward -= 0.5
    # Penalize degenerate repetition without needing a language model judge.
    words = stripped.lower().split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        reward += 0.2 * math.tanh(3.0 * (unique_ratio - 0.4))
    return reward


def compute_reward(text: str, config: dict[str, Any]) -> float:
    kind = str(config.get("kind", "keyword_length"))
    if kind == "keyword_length":
        return keyword_length_reward(text, config)
    if kind == "regex":
        return regex_reward(text, config)
    if kind == "heuristic_helpfulness":
        return heuristic_helpfulness_reward(text, config)
    raise ValueError(f"unknown reward kind: {kind}")
