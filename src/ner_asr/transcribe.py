"""Qwen3-ASR wrapper (Transformers-native "-hf" checkpoints).

Follows the lecturer's `transcribe()` in mmai_06/02_qwen3_asr_playground.ipynb:
`AutoProcessor` + `AutoModelForMultimodalLM`, `processor.apply_transcription_request`,
greedy `generate(max_new_tokens=256)`, `decode(..., return_format="transcription_only")`.
Requires transformers>=5.13.

Note: "Qwen/Qwen3-ASR-1.7B" / "-0.6B" (without "-hf") are the original checkpoints for
the separate `qwen-asr` package and are NOT loadable this way. Use the "-hf" ids.

The baseline never passes a prompt: `prompt` stays None. Context conditioning
(portfolio Step 5) is out of scope for the Step 1-2 submission.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import ModelConfig


@dataclass
class TranscriptionResult:
    prediction: str
    raw_output: str
    n_generated_tokens: int
    hit_token_limit: bool
    runtime_s: float


class QwenASRTranscriber:
    def __init__(self, cfg: ModelConfig):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.cfg = cfg
        self._torch = torch
        device = cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = {
            "auto": torch.bfloat16 if device == "cuda" else torch.float32,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[cfg.dtype]
        self.device = device
        self.processor = AutoProcessor.from_pretrained(cfg.name, revision=cfg.revision)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            cfg.name, revision=cfg.revision, dtype=dtype, device_map=device
        ).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()

    def transcribe_batch(self, audio_paths: list[str], prompt: str | None = None) -> list[TranscriptionResult]:
        kwargs = {"audio": audio_paths if len(audio_paths) > 1 else audio_paths[0]}
        if self.cfg.language:
            kwargs["language"] = [self.cfg.language] * len(audio_paths) if len(audio_paths) > 1 else self.cfg.language
        if prompt:
            kwargs["prompt"] = prompt
        start = time.perf_counter()
        inputs = self.processor.apply_transcription_request(**kwargs).to(self.model.device, self.model.dtype)
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.decode(gen, return_format="transcription_only")
        raws = self.processor.decode(gen)
        elapsed = time.perf_counter() - start
        results = []
        pad = self.processor.tokenizer.pad_token_id
        for i in range(len(audio_paths)):
            ids = [t for t in gen[i].tolist() if t != pad]
            ended = bool(ids) and ids[-1] in self.eos_ids
            results.append(
                TranscriptionResult(
                    prediction=texts[i].strip(),
                    raw_output=raws[i],
                    n_generated_tokens=len(ids),
                    hit_token_limit=(len(ids) >= self.cfg.max_new_tokens and not ended),
                    runtime_s=elapsed / len(audio_paths),
                )
            )
        return results
