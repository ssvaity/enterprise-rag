from types import SimpleNamespace

import json

import pytest

from app.ingest import Page, UnsupportedFileError, chunk_pages, parse
from app.llm import ClaudeLLM, Completion, LLMError, OpenRouterLLM, Thinking
from app.rag import NOT_FOUND, RAGPipeline, attach_citations
from app.store import Hit, VectorStore


def test_chunks_overlap_and_keep_page_numbers():
    words = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_pages([Page(words, page=3)], chunk_tokens=40, overlap=0.25)  # 30 words, step 23
    assert [c.page for c in chunks] == [3] * len(chunks)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert len(first) == 30 and second[0] == "w23"
    assert chunks[-1].text.split()[-1] == "w99"


def test_chunks_do_not_span_pages():
    chunks = chunk_pages([Page("alpha beta", 1), Page("", 2), Page("gamma", 3)], 700, 0.15)
    assert [(c.text, c.page) for c in chunks] == [("alpha beta", 1), ("gamma", 3)]


def test_parse_rejects_unknown_types():
    with pytest.raises(UnsupportedFileError):
        parse("sheet.xlsx", b"")


def test_parse_html_strips_markup():
    pages = parse("a.html", b"<html><script>x()</script><p>Hello <b>world</b></p></html>")
    assert "x()" not in pages[0].text and "Hello" in pages[0].text


def _hit(name, page=None):
    return Hit(text=f"text of {name}", doc_id="d", filename=name, page=page, chunk_index=0, score=0.9)


def test_attach_citations_drops_unknown_tags_and_orders_by_use():
    hits = [_hit("a.pdf", 1), _hit("b.docx")]
    text, cites = attach_citations("Fact one [S2]. Fact two [S1][S2]. Bogus [S7].", hits)
    assert "[S7]" not in text
    assert [c.tag for c in cites] == ["S2", "S1"]
    assert cites[1].page == 1 and cites[0].page is None


class FakeLLM:
    name = "fake/model"

    def __init__(self, reply, refused=False, fail_after=None, thinking=()):
        self.reply, self.refused, self.calls = reply, refused, []
        self.fail_after = fail_after  # raise LLMError after this many streamed pieces
        self.thinking = thinking  # reasoning pieces streamed before the answer

    def complete(self, system, user):
        self.calls.append({"system": system, "user": user})
        return Completion(self.reply, refused=self.refused)

    def stream(self, system, user):
        self.calls.append({"system": system, "user": user})
        for t in self.thinking:
            yield Thinking(t)
        pieces = self.reply.split(" ")
        for i, word in enumerate(pieces):
            if self.fail_after is not None and i == self.fail_after:
                raise LLMError(502, "stream broke")
            yield word if i == 0 else " " + word
        yield Completion(self.reply, refused=self.refused)


@pytest.fixture
def store(tmp_path):
    s = VectorStore(path=str(tmp_path / "chroma"), collection="test")
    leave = chunk_pages([Page("New employees receive 18 days of paid annual leave per year.", 3)], 700, 0.15)
    it = chunk_pages([Page("Laptops are refreshed every three years by the IT team.", 1)], 700, 0.15)
    s.add_document("leave", "Leave_Policy.pdf", leave)
    s.add_document("it", "IT_Handbook.pdf", it)
    return s


def test_search_ranks_semantically_relevant_chunk_first(store):
    hits = store.search("How much vacation do new staff get?", k=2)
    assert hits[0].filename == "Leave_Policy.pdf" and hits[0].page == 3


def test_ask_returns_answer_with_citations(store):
    client = FakeLLM("New employees get 18 days of paid annual leave [S1].")
    answer = RAGPipeline(store, client).ask("How many leave days do new employees get?", k=2)
    assert answer.citations[0].filename == "Leave_Policy.pdf"
    assert answer.citations[0].page == 3
    prompt = client.calls[0]["user"]
    assert '<source id="S1" label="Leave_Policy.pdf, p.3">' in prompt


def test_ask_uncited_answer_becomes_not_found(store):
    answer = RAGPipeline(store, FakeLLM("Probably 20 days.")).ask("leave?", k=2)
    assert answer.answer == NOT_FOUND and answer.citations == []


def test_ask_with_empty_store_skips_llm(tmp_path):
    client = FakeLLM("unused")
    answer = RAGPipeline(VectorStore(str(tmp_path / "c"), "empty"), client).ask("anything")
    assert answer.answer == NOT_FOUND and client.calls == []


def test_delete_document(store):
    store.delete_document("leave")
    assert [d["doc_id"] for d in store.list_documents()] == ["it"]


def test_refusal_is_reported(store):
    answer = RAGPipeline(store, FakeLLM("", refused=True)).ask("leave?", k=2)
    assert "declined" in answer.answer and answer.citations == []


def _fake_openai(content=None, finish_reason="stop", error=None):
    calls = []

    def create(**kw):
        calls.append(kw)
        if error:
            raise error
        msg = SimpleNamespace(content=content)
        choices = [SimpleNamespace(message=msg, finish_reason=finish_reason)]
        return SimpleNamespace(model=kw["model"], choices=choices)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls


def test_openrouter_sends_system_and_user_messages():
    client, calls = _fake_openai("Answer [S1].")
    llm = OpenRouterLLM(model="google/gemma-4-31b-it:free", client=client)
    assert llm.complete("sys", "usr").text == "Answer [S1]."
    assert calls[0]["model"] == "google/gemma-4-31b-it:free"
    assert calls[0]["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]


def test_openrouter_content_filter_counts_as_refusal():
    client, _ = _fake_openai(None, finish_reason="content_filter")
    assert OpenRouterLLM(client=client).complete("s", "u").refused


def test_openrouter_rate_limit_maps_to_429():
    import httpx2
    import openai

    req = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    err = openai.RateLimitError("rate limited", response=httpx2.Response(429, request=req), body=None)
    client, _ = _fake_openai(error=err)
    with pytest.raises(LLMError) as exc:
        OpenRouterLLM(client=client).complete("s", "u")
    assert exc.value.status == 429


def test_openrouter_requires_key(monkeypatch):
    monkeypatch.setattr("app.llm.settings", SimpleNamespace(openrouter_model="m", openrouter_api_key=""))
    with pytest.raises(LLMError):
        OpenRouterLLM()


# ---------------------------------------------------------------- streaming


def test_ask_stream_emits_sources_tokens_then_validated_answer(store):
    llm = FakeLLM("New employees get 18 days [S1]. Also [S9].")
    events = list(RAGPipeline(store, llm).ask_stream("leave for new staff?", k=2))
    names = [e for e, _ in events]
    assert names[0] == "sources" and names[-1] == "done"
    assert set(names[1:-1]) == {"token"}
    assert "".join(d["text"] for e, d in events if e == "token") == "New employees get 18 days [S1]. Also [S9]."
    done = events[-1][1]
    assert "[S9]" not in done["answer"]  # bogus tag removed in the final answer
    assert [c["tag"] for c in done["citations"]] == ["S1"]
    assert done == RAGPipeline(store, FakeLLM("New employees get 18 days [S1]. Also [S9].")).ask(
        "leave for new staff?", k=2
    ).to_dict()


def test_ask_stream_sends_reasoning_as_thinking_events_before_tokens(store):
    llm = FakeLLM("18 days [S1].", thinking=["Looking at S1", " for leave."])
    events = list(RAGPipeline(store, llm).ask_stream("leave?", k=2))
    names = [e for e, _ in events]
    assert names[:3] == ["sources", "thinking", "thinking"] and names[3] == "token"
    assert "".join(d["text"] for e, d in events if e == "thinking") == "Looking at S1 for leave."
    assert events[-1][1]["answer"] == "18 days [S1]."  # reasoning never leaks into the answer


def test_ask_stream_uncited_answer_becomes_not_found(store):
    *_, (event, done) = RAGPipeline(store, FakeLLM("Probably 20 days.")).ask_stream("leave?", k=2)
    assert event == "done" and done["answer"] == NOT_FOUND


def test_ask_stream_with_empty_store_skips_llm(tmp_path):
    llm = FakeLLM("unused")
    events = list(RAGPipeline(VectorStore(str(tmp_path / "c"), "empty"), llm).ask_stream("anything"))
    assert [e for e, _ in events] == ["sources", "done"]
    assert events[-1][1]["answer"] == NOT_FOUND and llm.calls == []


def _sse_events(text):
    events = []
    for raw in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in raw.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def client(store, monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main

    def use(llm):
        monkeypatch.setattr(main, "get_pipeline", lambda: RAGPipeline(store, llm))
        return TestClient(main.app)

    return use


def test_stream_endpoint_sends_server_sent_events(client):
    res = client(FakeLLM("18 days [S1].")).post("/api/ask/stream", json={"question": "leave?", "top_k": 2})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(res.text)
    assert [e for e, _ in events] == ["sources", "token", "token", "token", "done"]
    assert events[-1][1]["citations"][0]["filename"] == "Leave_Policy.pdf"


def test_stream_endpoint_reports_mid_stream_failure_as_error_event(client):
    res = client(FakeLLM("one two three", fail_after=2)).post("/api/ask/stream", json={"question": "leave?"})
    events = _sse_events(res.text)
    assert [e for e, _ in events] == ["sources", "token", "token", "error"]
    assert events[-1][1] == {"status": 502, "detail": "stream broke"}


def test_stream_endpoint_returns_http_error_when_llm_is_not_configured(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main

    def broken():
        raise LLMError(500, "OPENROUTER_API_KEY is not set")

    monkeypatch.setattr(main, "get_pipeline", broken)
    res = TestClient(main.app).post("/api/ask/stream", json={"question": "hi"})
    assert res.status_code == 500 and res.json()["detail"] == "OPENROUTER_API_KEY is not set"


def _chunk(content=None, finish_reason=None, model="vendor/model:free", choices=True, reasoning=None):
    delta = SimpleNamespace(content=content, model_extra={"reasoning": reasoning} if reasoning else {})
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)] if choices else [],
    )


def _fake_openai_stream(chunks):
    calls = []

    def create(**kw):
        calls.append(kw)
        return iter(chunks)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def test_openrouter_stream_yields_text_then_completion():
    chunks = [_chunk("Hello"), _chunk(" world"), _chunk(None, "stop"), _chunk(choices=False)]
    client, calls = _fake_openai_stream(chunks)
    out = list(OpenRouterLLM(model="openrouter/free", client=client).stream("s", "u"))
    assert out[:-1] == ["Hello", " world"]
    assert out[-1] == Completion("Hello world", model="openrouter/vendor/model:free")
    assert calls[0]["stream"] is True


def test_openrouter_stream_passes_reasoning_through_as_thinking():
    chunks = [_chunk("", reasoning="Let me"), _chunk("", reasoning=" check."), _chunk("Answer."), _chunk(None, "stop")]
    client, _ = _fake_openai_stream(chunks)
    out = list(OpenRouterLLM(client=client).stream("s", "u"))
    assert out[:3] == [Thinking("Let me"), Thinking(" check."), "Answer."]
    assert out[-1].text == "Answer."  # reasoning is not part of the answer


def test_openrouter_stream_content_filter_counts_as_refusal():
    client, _ = _fake_openai_stream([_chunk("partial"), _chunk(None, "content_filter")])
    assert list(OpenRouterLLM(client=client).stream("s", "u"))[-1].refused


def test_openrouter_stream_maps_errors_raised_mid_stream():
    import httpx2
    import openai

    def chunks():
        yield _chunk("partial")
        req = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        raise openai.APIError("upstream died", request=req, body=None)

    client, _ = _fake_openai_stream(chunks())
    gen = OpenRouterLLM(client=client).stream("s", "u")
    assert next(gen) == "partial"
    with pytest.raises(LLMError) as exc:
        next(gen)
    assert exc.value.status == 502


class _FakeClaudeStream:
    """Mimics the SDK's beta message stream: iterating yields raw events."""

    def __init__(self, pieces, stop_reason="end_turn", thinking=()):
        delta = lambda **kw: SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(**kw))
        self._events = [SimpleNamespace(type="message_start")]
        self._events += [delta(type="thinking_delta", thinking=t) for t in thinking]
        self._events += [delta(type="text_delta", text=p) for p in pieces]
        self._events += [SimpleNamespace(type="message_stop")]
        text = SimpleNamespace(type="text", text="".join(pieces))
        self._final = SimpleNamespace(stop_reason=stop_reason, content=[text])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final


def _fake_anthropic(stream):
    calls = []

    def make_stream(**kw):
        calls.append(kw)
        return stream

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=make_stream))), calls


def test_claude_stream_yields_text_then_completion():
    client, calls = _fake_anthropic(_FakeClaudeStream(["18 ", "days [S1]."]))
    out = list(ClaudeLLM(model="claude-opus-5-5", client=client).stream("sys", "usr"))
    assert out == ["18 ", "days [S1].", Completion("18 days [S1].")]
    assert calls[0]["system"] == "sys" and calls[0]["fallbacks"] == "default"
    assert calls[0]["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_claude_stream_passes_thinking_through():
    client, _ = _fake_anthropic(_FakeClaudeStream(["Answer."], thinking=["Checking S1."]))
    out = list(ClaudeLLM(client=client).stream("s", "u"))
    assert out == [Thinking("Checking S1."), "Answer.", Completion("Answer.")]


def test_claude_stream_refusal():
    client, _ = _fake_anthropic(_FakeClaudeStream(["par"], stop_reason="refusal"))
    assert list(ClaudeLLM(client=client).stream("s", "u"))[-1].refused
