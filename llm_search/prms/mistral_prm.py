from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.types import Device
from transformers import (  # type: ignore  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .base_prm import PRM, StepScore


class MistralPRM(PRM):
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str = "peiyi9979/math-shepherd-mistral-7b-prm",
        aggregation: Literal["min", "max", "mean", "last"] = "min",
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
    ) -> None:
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
        )
        if not quantization_config:
            self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.candidate_tokens = self.tokenizer.encode("+ -")[1:]
        self.step_tag = "ки"
        self.step_tag_id = self.tokenizer.encode(self.step_tag)[-1]
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token_id
        self.aggregation = {
            "min": np.min,
            "max": np.max,
            "mean": np.mean,
            "last": lambda x: x[-1],
        }[aggregation]

    @torch.no_grad()
    def __call__(self, beams: list[str]) -> list[StepScore]:
        """
        Args:
            beams (list[str]): A list of reasoning beams.

        Returns:
            list[StepScore]: Scores for each beam.
        """
        good_beams = []
        good_beam_idxs = []
        scored_beams = [StepScore(step=b, score=0.0) for b in beams]

        for i, beam in enumerate(beams):
            if self.step_tag in beam:
                good_beams.append(beam)
                good_beam_idxs.append(i)

        if len(good_beams) > 0:
            inputs = self.tokenizer(good_beams, return_tensors="pt", padding=True)  # type: ignore
            step_mask = inputs["input_ids"] == self.step_tag_id
            logits = self.model(**inputs).logits[:, :, self.candidate_tokens]
            for i, j in enumerate(good_beam_idxs):
                step_scores = F.softmax(logits, dim=-1)[i, step_mask[i], 0]
                aggregate_score = self.aggregation(step_scores.numpy(force=True))  # type: ignore
                scored_beams[j].score = aggregate_score

        return scored_beams
