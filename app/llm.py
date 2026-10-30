"""LLM providers behind one interface: Claude (Anthropic API) or OpenRouter."""

from dataclasses import dataclass
from typing import Protocol

from .config import settings


@dataclass
class Completion:
    text: str
    refused: bool = False
    model: str = ""  # model that actually answered, when the provider reports it


class LLMError(Exception):
    """Provider failure, mapped to an HTTP status for the API layer."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> Completion: ...


class ClaudeLLM:
    def __init__(self, model: str | None = None, client=None):
        import anthropic

        self._anthropic = anthropic
        self.model = model or settings.claude_model
        self.name = f"anthropic/{self.model}"
        self._client = client or anthropic.Anthropic()

    def complete(self, system: str, user: str) -> Completion:
        anthropic = self._anthropic
        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low"},
                # On a safety decline, re-run the request on a fallback model chosen by the API.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError:
            raise LLMError(500, "Claude credentials are invalid (check ANTHROPIC_API_KEY).")
        except anthropic.RateLimitError:
            raise LLMError(429, "Rate limited by the Claude API, try again shortly.")
        except anthropic.APIStatusError as e:
            raise LLMError(502, f"Claude API error: {e.message}")
        except anthropic.APIConnectionError:
            raise LLMError(502, "Could not reach the Claude API.")
        except TypeError as e:
            # The SDK raises TypeError before sending when no credentials are configured.
            if "authentication" not in str(e):
                raise
            raise LLMError(500, "Claude credentials are missing (set ANTHROPIC_API_KEY or run `ant auth login`).")
        if response.stop_reason == "refusal":
            return Completion("", refused=True)
        return Completion("".join(b.text for b in response.content if b.type == "text"))


class OpenRouterLLM:
    """OpenRouter exposes an OpenAI-compatible endpoint; free models end in ':free'."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str | None = None, api_key: str | None = None, client=None):
        import openai

        self._openai = openai
        self.model = model or settings.openrouter_model
        self.name = f"openrouter/{self.model}"
        if client is None:
            api_key = api_key or settings.openrouter_api_key
            if not api_key:
                raise LLMError(500, "OPENROUTER_API_KEY is not set (get a free key at openrouter.ai/keys).")
            client = openai.OpenAI(base_url=self.BASE_URL, api_key=api_key, max_retries=2)
        self._client = client

    def complete(self, system: str, user: str) -> Completion:
        openai = self._openai
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.1,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                extra_headers={"X-Title": "Enterprise RAG"},
            )
        except openai.AuthenticationError:
            raise LLMError(500, "OpenRouter key is invalid (check OPENROUTER_API_KEY).")
        except openai.RateLimitError as e:
            raise LLMError(429, f"OpenRouter rate limit for {self.model}, retry shortly ({e.message}).")
        except openai.APIStatusError as e:
            raise LLMError(502, f"OpenRouter error ({e.status_code}): {e.message}")
        except openai.APIConnectionError:
            raise LLMError(502, "Could not reach OpenRouter.")
        if not response.choices:
            raise LLMError(502, "OpenRouter returned no choices.")
        choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            return Completion("", refused=True)
        return Completion(choice.message.content or "", model=f"openrouter/{response.model}")


def get_llm() -> LLM:
    if settings.llm_provider == "openrouter":
        return OpenRouterLLM()
    if settings.llm_provider == "anthropic":
        return ClaudeLLM()
    raise ValueError(f"Unknown RAG_LLM_PROVIDER '{settings.llm_provider}' (use 'anthropic' or 'openrouter')")
