"""LLM Abstraction Layer — Unified interface to any language model.

Supports local (Ollama, HuggingFace) and cloud (OpenAI, Azure, Anthropic) providers.
v0.1: Ollama + OpenAI-compatible + HuggingFace Transformers.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 2048) -> LLMResponse:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass


class OllamaProvider(BaseLLMProvider):
    """Local LLM via Ollama (free, private, offline)."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    @property
    def provider_name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 2048) -> LLMResponse:
        import urllib.request

        start = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())

            latency = time.time() - start
            text = result.get("message", {}).get("content", "")
            input_tokens = result.get("prompt_eval_count", 0)
            output_tokens = result.get("eval_count", 0)

            return LLMResponse(
                text=text,
                model=self.model,
                provider="ollama",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=latency,
            )
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise ConnectionError(f"Ollama error: {e}. Is Ollama running at {self.base_url}?")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible API provider (works with OpenAI, Grok, Groq, Together, vLLM, etc.)."""

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = None
        self._use_sdk = False
        self._init_client()

    def _init_client(self):
        """Try to use the openai SDK (handles auth, retries, streaming properly)."""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self._use_sdk = True
            logger.info(f"Using openai SDK for {self.base_url}")
        except ImportError:
            self._use_sdk = False
            logger.info("openai SDK not installed, using urllib fallback")

    @property
    def provider_name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 2048) -> LLMResponse:
        if self._use_sdk:
            return self._generate_sdk(prompt, system_prompt, temperature, max_tokens)
        return self._generate_urllib(prompt, system_prompt, temperature, max_tokens)

    def _generate_sdk(self, prompt: str, system_prompt: Optional[str] = None,
                      temperature: float = 0.1, max_tokens: int = 2048) -> LLMResponse:
        """Generate using the openai Python SDK (recommended)."""
        start = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency = time.time() - start
            text = response.choices[0].message.content or ""
            usage = response.usage

            return LLMResponse(
                text=text,
                model=self.model,
                provider="openai",
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_seconds=latency,
            )
        except Exception as e:
            logger.error(f"OpenAI SDK generation failed: {e}")
            raise ConnectionError(f"OpenAI API error: {e}")

    def _generate_urllib(self, prompt: str, system_prompt: Optional[str] = None,
                         temperature: float = 0.1, max_tokens: int = 2048) -> LLMResponse:
        """Fallback: generate using urllib (no SDK dependency)."""
        import urllib.request

        start = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())

            latency = time.time() - start
            choice = result["choices"][0]
            text = choice["message"]["content"]
            usage = result.get("usage", {})

            return LLMResponse(
                text=text,
                model=self.model,
                provider="openai",
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                latency_seconds=latency,
            )
        except Exception as e:
            logger.error(f"OpenAI urllib generation failed: {e}")
            raise ConnectionError(f"OpenAI API error: {e}")


class HuggingFaceProvider(BaseLLMProvider):
    """Local HuggingFace Transformers provider (research, any model)."""

    def __init__(self, model: str = "microsoft/phi-2", device: str = "auto"):
        self.model_name = model
        self.device = device
        self._model = None
        self._tokenizer = None

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def is_available(self) -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False

    def _load_model(self):
        """Lazy model loading."""
        if self._model is not None:
            return

        logger.info(f"Loading HuggingFace model: {self.model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device if self.device != "auto" else ("auto" if torch.cuda.is_available() else "cpu"),
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        logger.info(f"Model loaded: {self.model_name}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 2048) -> LLMResponse:
        self._load_model()
        import torch

        start = time.time()

        # Use chat template if available, else simple format
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            full_prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\n\nAssistant:"

        inputs = self._tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 1024),
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
            )

        generated = outputs[0][input_len:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Clean garbled output
        text = self._clean_output(text)

        latency = time.time() - start

        return LLMResponse(
            text=text,
            model=self.model_name,
            provider="huggingface",
            input_tokens=input_len,
            output_tokens=len(generated),
            latency_seconds=latency,
        )

    @staticmethod
    def _clean_output(text: str) -> str:
        """Remove garbled/repetitive content from model output."""
        if not text:
            return text

        lines = text.split("\n")
        clean_lines = []
        seen = set()

        for line in lines:
            stripped = line.strip()
            # Skip empty or very short repeated lines
            if not stripped:
                if clean_lines and clean_lines[-1] != "":
                    clean_lines.append("")
                continue

            # Skip if line is mostly non-alphabetic (garbled)
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in stripped) / max(len(stripped), 1)
            if alpha_ratio < 0.5 and len(stripped) > 20:
                break  # Stop at first garbled line

            # Skip repeated lines
            key = stripped.lower()[:80]
            if key in seen and len(key) > 10:
                continue
            seen.add(key)

            # Stop if line is just repeated characters
            if len(set(stripped)) < 3 and len(stripped) > 5:
                break

            clean_lines.append(line)

        result = "\n".join(clean_lines).strip()

        # If result is too short after cleaning, return what we have
        if len(result) < 10 and len(text) > 10:
            # Take first 500 chars of original, cut at last sentence
            result = text[:500]
            last_period = result.rfind(".")
            if last_period > 50:
                result = result[:last_period + 1]

        return result


class LLMManager:
    """Factory and manager for LLM providers."""

    PROVIDER_MAP = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "azure_openai": OpenAIProvider,
        "huggingface": HuggingFaceProvider,
        "anthropic": OpenAIProvider,  # Uses OpenAI-compatible API
        "groq": OpenAIProvider,
        "together": OpenAIProvider,
        "custom": OpenAIProvider,
    }

    BASE_URLS = {
        "openai": "https://api.openai.com/v1",
        "azure_openai": None,  # Set by user
        "anthropic": "https://api.anthropic.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "together": "https://api.together.xyz/v1",
    }

    @classmethod
    def create(cls, backend: str, model: str, api_key: Optional[str] = None,
               base_url: Optional[str] = None, **kwargs) -> BaseLLMProvider:
        """Create an LLM provider instance."""
        backend = backend.lower().replace("-", "_")

        if backend == "ollama":
            url = base_url or "http://localhost:11434"
            return OllamaProvider(model=model, base_url=url)

        if backend == "huggingface":
            return HuggingFaceProvider(model=model, device=kwargs.get("device", "auto"))

        # All OpenAI-compatible providers
        if backend in cls.BASE_URLS:
            url = base_url or cls.BASE_URLS.get(backend, "https://api.openai.com/v1")
        else:
            url = base_url or "https://api.openai.com/v1"

        if backend == "azure_openai":
            endpoint = kwargs.get("azure_endpoint", base_url)
            deployment = kwargs.get("deployment_name", model)
            if endpoint:
                url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}"

        return OpenAIProvider(model=model, api_key=api_key, base_url=url)
