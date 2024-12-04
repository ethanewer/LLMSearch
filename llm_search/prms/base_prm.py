from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StepScore:
    step: str
    score: float


class PRM(ABC):
    """Interface for Process Reward Models."""

    @abstractmethod
    def __call__(self, beams: list[str]) -> list[StepScore]:
        """
        Args:
            beams (list[str]): A list of reasoning beams.

        Returns:
            list[StepScore]: Scores for each beam.
        """
        ...
