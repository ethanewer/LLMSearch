from abc import ABC, abstractmethod


class Generator(ABC):
    """Interface for generator models to be used with the Search class."""

    @abstractmethod
    def generate(
        self,
        input_text: str,
        num_beams: int,
        max_new_tokens: int,
    ) -> list[str]:
        """Generates `num_beams` full responses from `input_text`."""
        ...

    @abstractmethod
    def generate_step(
        self,
        input_text: str,
        num_beams: int,
        max_new_tokens: int,
    ) -> list[str]:
        """Generates `num_beams` versions of the next reasoning step."""
        ...

    @abstractmethod
    def is_complete(self, input_text: str) -> bool:
        """Returns `True` if the `input_text` is complete."""
        ...
