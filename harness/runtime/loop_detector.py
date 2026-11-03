"""Spots an agent asking the same question over and over."""

from __future__ import annotations

import json
from collections import deque
from typing import Any


class LoopDetector:
    """
    record() returns True once the same (tool, arguments) pair has been seen
    `threshold` times within the last `window` calls. With threshold=3 the
    third identical call is the one that trips it.
    """
    def __init__(self, threshold: int = 3 , window: int = 10):
        if threshold < 2 :
            raise ValueError("threshold must be >= 2")
        if window < threshold:
            raise ValueError("window must be >= threshold")
        self.threshold = threshold
        self._recent: deque[tuple[str , str]] = deque(maxlen=window)

    @staticmethod
    def _key(tool_name: str , arguments: dict[str , Any]) -> tuple[str , str]:
        return tool_name, json.dumps(arguments , sort_keys=True , default=str)

    def record(self,tool_name : str , arguments: dict[str , Any]) -> bool:
        key = self._key(tool_name , arguments)
        self._recent.append(key)
        return self._recent.count(key) >= self.threshold

    def reset(self) -> None:
        self._recent.clear()