from types import SimpleNamespace

import pytest

from app.ingest import Page, UnsupportedFileError, chunk_pages, parse
from app.llm import Completion, LLMError, OpenRouterLLM
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

    def __init__(self, reply, refused=False):
        self.reply, self.refused, self.calls = reply, refused, []

    def complete(self, system, user):
        self.calls.append({"system": system, "user": user})
        return Completion(self.reply, refused=self.refused)


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
