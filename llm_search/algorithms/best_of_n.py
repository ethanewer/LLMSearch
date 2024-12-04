from typing import Any, Optional

from ..generators.base_generator import Generator
from ..prms.base_prm import PRM
from .base_search import Search


class BestOfN(Search):
    def __init__(
        self,
        generator: Generator,
        prm: PRM,
        width: int = 5,
        max_new_tokens: int = 2000,
    ) -> None:
        """
        Args:
            generator (Generator): Model used for generating reasoning steps.
            prm (PRM): Process reward generator used for scoring reasoning steps.
            width (int): The number trajectories (n).
            max_new_tokens (int): Max new tokens for each trajectory.
        """
        self.generator = generator
        self.prm = prm
        self.width = width
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> tuple[Optional[str], dict[str, Any]]:
        beams = self.generator.generate(prompt, self.width, self.max_new_tokens)
        filtered_beams = [b for b in beams if self.generator.is_complete(b)]
        if len(filtered_beams) == 0:
            return None, {"beams": beams, "scored_beams": []}

        scored_beams = self.prm(filtered_beams)
        final_solution = max(scored_beams, key=lambda x: x.score).step
        return final_solution, {"beams": beams, "scored_beams": scored_beams}
