"""LLM providers behind one interface: Claude (Anthropic API) or OpenRouter."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from .config import settings


@dataclass
class Completion:
    text: str
    refused: bool = False
    model: str = ""  # model that actually answered, when the provider reports it


@dataclass
class Thinking:
    """A piece of the model's reasoning, streamed before the answer by
    reasoning models. Shown to the user as progress, never as the answer."""

    text: str


class LLMError(Exception):
    """Provider failure, mapped to an HTTP status for the API layer."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> Completion: ...

    def stream(self, system: str, user: str) -> Iterator[str | Thinking | Completion]:
        """Yield answer text pieces (str) and reasoning pieces (Thinking) as
        they arrive, then one final Completion holding the full answer."""
        ...


class ClaudeLLM:
    def __init__(self, model: str | None = None, client=None):
        import anthropic

        self._anthropic = anthropic
        self.model = model or settings.claude_model
        self.name = f"anthropic/{self.model}"
        self._client = client or anthropic.Anthropic()

    def _params(self, system: str, user: str) -> dict:
        return dict(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},
            # On a safety decline, re-run the request on a fallback model chosen by the API.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    @contextmanager
    def _errors(self):
        anthropic = self._anthropic
        try:
            yield
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

    @staticmethod
    def _completion(message) -> Completion:
        if message.stop_reason == "refusal":
            return Completion("", refused=True)
        return Completion("".join(b.text for b in message.content if b.type == "text"))

    def complete(self, system: str, user: str) -> Completion:
        with self._errors():
            response = self._client.beta.messages.create(**self._params(system, user))
        return self._completion(response)

    def stream(self, system: str, user: str) -> Iterator[str | Thinking | Completion]:
        params = self._params(system, user)
        # Summarized thinking gives the user something to watch while the
        # model reasons; the default ("omitted") streams empty thinking blocks.
        params["thinking"] = {"type": "adaptive", "display": "summarized"}
        with self._errors():
            with self._client.beta.messages.stream(**params) as stream:
                for event in stream:
                    if event.type != "content_block_delta":
                        continue
                    if event.delta.type == "text_delta":
                        yield event.delta.text
                    elif event.delta.type == "thinking_delta" and event.delta.thinking:
                        yield Thinking(event.delta.thinking)
                final = stream.get_final_message()
        yield self._completion(final)


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

    def _params(self, system: str, user: str) -> dict:
        return dict(
            model=self.model,
            max_tokens=2048,
            temperature=0.1,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            extra_headers={"X-Title": "Enterprise RAG"},
        )

    @contextmanager
    def _errors(self):
        openai = self._openai
        try:
            yield
        except openai.AuthenticationError:
            raise LLMError(500, "OpenRouter key is invalid (check OPENROUTER_API_KEY).")
        except openai.RateLimitError as e:
            raise LLMError(429, f"OpenRouter rate limit for {self.model}, retry shortly ({e.message}).")
        except openai.APIStatusError as e:
            raise LLMError(502, f"OpenRouter error ({e.status_code}): {e.message}")
        except openai.APIConnectionError:
            raise LLMError(502, "Could not reach OpenRouter.")
        except openai.APIError as e:
            # Errors reported inside an already-open stream.
            raise LLMError(502, f"OpenRouter error: {e.message}")

    def complete(self, system: str, user: str) -> Completion:
        with self._errors():
            response = self._client.chat.completions.create(**self._params(system, user))
        if not response.choices:
            raise LLMError(502, "OpenRouter returned no choices.")
        choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            return Completion("", refused=True)
        return Completion(choice.message.content or "", model=f"openrouter/{response.model}")

    def stream(self, system: str, user: str) -> Iterator[str | Thinking | Completion]:
        parts: list[str] = []
        model = ""
        finish_reason = None
        with self._errors():
            for chunk in self._client.chat.completions.create(**self._params(system, user), stream=True):
                model = chunk.model or model
                if not chunk.choices:  # e.g. a trailing usage-only chunk
                    continue
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                # Reasoning models (common among free ones) stream their
                # reasoning in a non-standard "reasoning" field first.
                reasoning = (getattr(choice.delta, "model_extra", None) or {}).get("reasoning")
                if reasoning:
                    yield Thinking(reasoning)
                if choice.delta.content:
                    parts.append(choice.delta.content)
                    yield choice.delta.content
        if finish_reason == "content_filter":
            yield Completion("", refused=True)
            return
        yield Completion("".join(parts), model=f"openrouter/{model}" if model else "")


def get_llm() -> LLM:
    if settings.llm_provider == "openrouter":
        return OpenRouterLLM()
    if settings.llm_provider == "anthropic":
        return ClaudeLLM()
    raise ValueError(f"Unknown RAG_LLM_PROVIDER '{settings.llm_provider}' (use 'anthropic' or 'openrouter')")
