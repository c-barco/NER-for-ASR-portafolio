"""Granite 4 Speech transcription wrapper.

This follows the official ``ibm-granite/granite-4.0-1b-speech`` model-card
example: ``AutoProcessor`` + ``AutoModelForSpeechSeq2Seq``, the Granite chat
template, and greedy generation. The documented no-context ASR request is used
unchanged for the baseline. A non-empty ``prompt`` is interpreted only as a
comma-separated keyword list and is appended using Granite's documented
``Keywords: ...`` mechanism; Steps 3--8 do not call it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .audio import load_audio
from .config import ModelConfig


@dataclass
class TranscriptionResult:
    prediction: str
    raw_output: str
    n_generated_tokens: int
    hit_token_limit: bool
    runtime_s: float


class Granite4ASRTranscriber:
    """Granite 4 Speech ASR with the project-wide transcriber interface.

    The model card documents one audio item per chat request. ``transcribe_batch``
    therefore preserves the public batch interface while processing paths in a
    deterministic sequence. This is deliberately conservative until real-model
    validation establishes a safe batched Granite request format.
    """

    BASELINE_USER_PROMPT = "<|audio|>can you transcribe the speech into a written format?"

    def __init__(self, cfg: ModelConfig):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

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
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            cfg.name, revision=cfg.revision, torch_dtype=dtype, device_map=device
        ).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()

    def _chat_prompt(self, keywords: str | None) -> str:
        user_prompt = self.BASELINE_USER_PROMPT
        if keywords and keywords.strip():
            keyword_text = keywords.strip()
            if keyword_text.lower().startswith("keywords:"):
                keyword_text = keyword_text.split(":", 1)[1].strip()
            if keyword_text:
                user_prompt += f" Keywords: {keyword_text}"
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _transcribe_one(self, audio_path: str, keywords: str | None) -> TranscriptionResult:
        # Granite's model card expects mono 16 kHz audio shaped as (channels, samples).
        waveform = self._torch.from_numpy(load_audio(audio_path)).unsqueeze(0)
        chat_prompt = self._chat_prompt(keywords)
        start = time.perf_counter()
        inputs = self.processor(chat_prompt, waveform, device=self.device, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        generated = outputs[:, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.batch_decode(
            generated,
            add_special_tokens=False,
            skip_special_tokens=True,
        )[0].strip()
        elapsed = time.perf_counter() - start
        pad = self.tokenizer.pad_token_id
        ids = [token for token in generated[0].tolist() if token != pad]
        ended = bool(ids) and ids[-1] in self.eos_ids
        return TranscriptionResult(
            prediction=text,
            raw_output=text,
            n_generated_tokens=len(ids),
            hit_token_limit=(len(ids) >= self.cfg.max_new_tokens and not ended),
            runtime_s=elapsed,
        )

    def transcribe_batch(self, audio_paths: list[str], prompt: str | None = None) -> list[TranscriptionResult]:
        return [self._transcribe_one(path, prompt) for path in audio_paths]
