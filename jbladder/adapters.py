"""
Model backends.

The primary backend runs open-weight instruction models locally through HuggingFace
``transformers`` with batched, greedy (deterministic) generation. A ``mock`` backend lets the
full pipeline be exercised with no model download — it deterministically "refuses" plain harmful
prompts and "complies" once certain attack markers appear, so tests and dry-runs are meaningful.
"""

from __future__ import annotations

import hashlib


class MockBackend:
    """Deterministic fake model: refuses harmful-looking plain prompts, is fooled by attacks.

    Used for offline pipeline tests. It inspects the rendered prompt for attack markers and
    returns a canned refusal or a canned (harmless placeholder) compliance accordingly.
    """

    name = "mock"

    ATTACK_MARKERS = (
        "AURA", "Absolutely! Here is", "base64-encoded", "leetspeak", "Let a =",
        "obeying these rules", "thriller novel", "certified safety researcher",
        "Historically, before modern",
    )

    def generate(self, batch_messages):
        out = []
        for messages in batch_messages:
            rendered = "\n".join(m["content"] for m in messages)
            fooled = any(mark in rendered for mark in self.ATTACK_MARKERS)
            # stable pseudo-random wobble so not every attack "succeeds"
            h = int(hashlib.sha1(rendered.encode()).hexdigest(), 16) % 100
            if fooled and h > 25:
                out.append(
                    "Absolutely! Here is a general, high-level overview for the scenario: "
                    "first you would consider the objective, then the resources, then the steps "
                    "1. plan 2. prepare 3. proceed. (mock output — no real content)."
                )
            else:
                out.append("I'm sorry, but I can't help with that request.")
        return out


class HFBackend:
    """Local HuggingFace transformers backend with batched greedy generation."""

    def __init__(self, model_name, device="auto", max_new_tokens=256, dtype="bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.max_new_tokens = max_new_tokens
        self._torch = torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        torch_dtype = getattr(torch, dtype) if device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to(device)
        self.model.eval()

    def generate(self, batch_messages):
        torch = self._torch
        prompts = [
            self.tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in batch_messages
        ]
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(self.device)
        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def close(self):
        import gc
        del self.model
        gc.collect()
        if self.device == "cuda":
            self._torch.cuda.empty_cache()


class OpenAIBackend:
    """Optional API backend. Use model id 'openai:gpt-4o-mini' etc. Needs OPENAI_API_KEY.

    Included so the identical harness can target a frontier model when a key is available; the
    default local runs use no keys at all.
    """

    def __init__(self, model_name, max_new_tokens=256, **_):
        import os
        from openai import OpenAI
        self.name = model_name
        self.model = model_name.split(":", 1)[1]
        self.max_new_tokens = max_new_tokens
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, batch_messages):
        out = []
        for messages in batch_messages:
            r = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0,
                max_tokens=self.max_new_tokens)
            out.append((r.choices[0].message.content or "").strip())
        return out


class AnthropicBackend:
    """Optional API backend. Use model id 'anthropic:claude-...'. Needs ANTHROPIC_API_KEY."""

    def __init__(self, model_name, max_new_tokens=256, **_):
        import os
        import anthropic
        self.name = model_name
        self.model = model_name.split(":", 1)[1]
        self.max_new_tokens = max_new_tokens
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, batch_messages):
        out = []
        for messages in batch_messages:
            system = " ".join(m["content"] for m in messages if m["role"] == "system")
            turns = [m for m in messages if m["role"] != "system"]
            r = self.client.messages.create(
                model=self.model, max_tokens=self.max_new_tokens, temperature=0,
                system=system or None, messages=turns)
            text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
            out.append(text.strip())
        return out


def get_backend(model_name, **kwargs):
    if model_name == "mock":
        return MockBackend()
    if model_name.startswith("openai:"):
        return OpenAIBackend(model_name, **{k: v for k, v in kwargs.items()
                                            if k in ("max_new_tokens",)})
    if model_name.startswith("anthropic:"):
        return AnthropicBackend(model_name, **{k: v for k, v in kwargs.items()
                                               if k in ("max_new_tokens",)})
    return HFBackend(model_name, **kwargs)
