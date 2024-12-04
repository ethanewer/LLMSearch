from abc import ABC, abstractmethod
from typing import Any, Optional


class Search(ABC):
    """An interface for LLM search algorithm."""

    @abstractmethod
    def __call__(self, prompt: str) -> tuple[Optional[str], dict[str, Any]]:
        """
        Args:
            prompt (str): A list of reasoning beams.

        Returns:
            (Optional[str], dict[str, Any]): The output if search was successful and a
            dict containing algorithm specific outputs.
        """
        ...
