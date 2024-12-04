from typing import Optional

import torch
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .base_generator import Generator


class LlamaGenerator(Generator):
    def __init__(
        self,
        llama_model_name: str,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            llama_model_name,
            quantization_config=quantization_config,
            token=hf_token,
        )

        if not quantization_config:
            self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.end_step_str = [" \n\n", "\n\n", ".\n\n", self.tokenizer.eos_token]
        self.end_solution_str = self.tokenizer.eos_token

    def generate(
        self,
        input_text: str,
        num_beams: int,
        max_new_tokens: int,
    ) -> list[str]:
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)

        outputs = self.model.generate(
            **inputs.to(self.device),
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stop_strings=self.end_solution_str,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def generate_step(
        self,
        input_text: str,
        num_beams: int,
        max_new_tokens: int,
    ) -> list[str]:
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)

        outputs = self.model.generate(
            **inputs.to(self.device),
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stop_strings=self.end_step_str,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def is_complete(self, input_text: str) -> bool:
        return self.end_solution_str in input_text
