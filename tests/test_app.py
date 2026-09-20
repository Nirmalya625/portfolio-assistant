import io
from unittest.mock import patch, MagicMock

import pytest
from reportlab.pdfgen import canvas

import app as app_module

def make_pdf_bytes(lines):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for line in lines:
        c.drawString(50, y, line)
        y -= 20
    c.save()
    buf.seek(0)
    return buf


def make_fake_stream(tokens):
    chunks = []
    for token in tokens:
        chunk = MagicMock()
        chunk.choices[0].delta.content = token
        chunks.append(chunk)
    return chunks


SID = {"X-Session-Id": "test-session"}


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    app_module.SESSIONS.clear()
    app_module.RATE.clear()
    with app_module.app.test_client() as c:
        yield c


def upload_pdf(client, lines, name="policy.pdf", headers=SID):
    return client.post(
        "/api/upload",
        data={"file": (make_pdf_bytes(lines), name)},
        content_type="multipart/form-data",
        headers=headers,
    )


def chat(client, message, reply, headers=SID):
    """POST a chat message with a mocked Groq stream; return (response_text, kwargs_sent_to_groq)."""
    with patch.object(
        app_module.client.chat.completions, "create", return_value=make_fake_stream(reply)
    ) as mock_create:
        res = client.post("/api/chat/stream", json={"message": message}, headers=headers)
        body = res.get_data(as_text=True)
    return res, body, mock_create.call_args.kwargs if mock_create.called else None


def test_index_page_loads(client):
    res = client.get("/")
    assert res.status_code == 200


def test_upload_rejects_missing_file(client):
    res = client.post("/api/upload", data={}, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "No file" in res.get_json()["error"]


def test_upload_rejects_non_pdf(client):
    data = {"file": (io.BytesIO(b"not a pdf"), "notes.txt")}
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "PDF" in res.get_json()["error"]


def test_upload_indexes_a_real_pdf(client):
    res = upload_pdf(client, ["Employees get twenty five days of annual leave."])
    body = res.get_json()

    assert res.status_code == 200
    assert body["filename"] == "policy.pdf"
    assert body["chunks_added"] >= 1
    assert "policy.pdf" in body["documents"]


def test_uploads_are_private_to_each_session(client):
    upload_pdf(client, ["Employees get twenty five days of annual leave."], headers={"X-Session-Id": "a"})
    res = client.get("/api/documents", headers={"X-Session-Id": "b"})
    assert res.get_json()["documents"] == []


def test_knowledge_base_is_loaded():
    assert app_module.knowledge.has_documents()
    titles = {c["title"] for c in app_module.knowledge.chunks}
    assert any("Fraud Detection" in t for t in titles)


def test_chat_about_nirmalya_is_grounded_in_knowledge_base(client):
    res, body, kwargs = chat(client, "Tell me about the fraud detection project", ["It uses a Random Forest."])
    assert res.status_code == 200
    last = kwargs["messages"][-1]["content"]
    assert "Document excerpts" in last
    assert "About Nirmalya: Project: Fraud Detection Console" in last
    assert kwargs["messages"][0]["role"] == "system"


def test_chat_unrelated_question_is_not_grounded(client):
    res, body, kwargs = chat(client, "what is the airspeed velocity of a swallow", ["An African or European swallow?"])
    assert res.status_code == 200
    assert body == "An African or European swallow?"
    assert kwargs["messages"][-1]["content"] == "what is the airspeed velocity of a swallow"


def test_chat_before_upload_answers_normally(client):
    res, body, kwargs = chat(client, "what is 2 + 2?", ["4"])
    assert res.status_code == 200
    assert body == "4"
    assert kwargs["messages"][-1]["content"] == "what is 2 + 2?"


def test_chat_rejects_empty_message(client):
    res = client.post("/api/chat/stream", json={"message": "   "}, headers=SID)
    assert res.status_code == 400


def test_chat_rejects_overlong_message(client):
    res = client.post("/api/chat/stream", json={"message": "x" * 5000}, headers=SID)
    assert res.status_code == 400


def test_chat_answers_from_uploaded_document_with_citation(client):
    upload_pdf(client, ["Employees get twenty five days of annual leave every year."])
    res, body, kwargs = chat(
        client, "how many leave days do employees get?",
        ["You get 25 days ", "[Source: policy.pdf, chunk 0]"],
    )
    assert res.status_code == 200
    assert "[Source: policy.pdf" in body
    last = kwargs["messages"][-1]["content"]
    assert "Document excerpts" in last
    assert "[Source: policy.pdf, chunk" in last


def test_follow_up_reuses_previous_question_for_retrieval(client):
    chat(client, "Tell me about the fraud detection project", ["It uses a Random Forest."])
    res, body, kwargs = chat(client, "and what else?", ["It also uses SHAP."])
    assert "Fraud Detection Console" in kwargs["messages"][-1]["content"]


def test_history_keeps_plain_question_not_excerpts(client):
    chat(client, "Tell me about the fraud detection project", ["Random Forest."])
    history = app_module.SESSIONS["test-session"]["history"]
    assert history[0] == {"role": "user", "content": "Tell me about the fraud detection project"}
    assert history[1]["role"] == "assistant"


def test_llm_failure_streams_friendly_error_and_skips_history(client):
    with patch.object(app_module.client.chat.completions, "create", side_effect=RuntimeError("boom")):
        res = client.post("/api/chat/stream", json={"message": "hello there"}, headers=SID)
        body = res.get_data(as_text=True)
    assert "Sorry" in body
    assert app_module.SESSIONS["test-session"]["history"] == []


def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(app_module, "RATE_LIMIT", 2)
    codes = []
    for _ in range(3):
        res, _body, _kw = chat(client, "hello there", ["hi"])
        codes.append(res.status_code)
    assert codes == [200, 200, 429]


def test_cors_allows_configured_origin_only(client, monkeypatch):
    monkeypatch.setattr(app_module, "ALLOWED_ORIGINS", ["https://me.example"])
    ok = client.get("/healthz", headers={"Origin": "https://me.example"})
    bad = client.get("/healthz", headers={"Origin": "https://evil.example"})
    assert ok.headers.get("Access-Control-Allow-Origin") == "https://me.example"
    assert "Access-Control-Allow-Origin" not in bad.headers


def test_preflight_succeeds(client, monkeypatch):
    monkeypatch.setattr(app_module, "ALLOWED_ORIGINS", ["https://me.example"])
    res = client.options("/api/chat/stream", headers={
        "Origin": "https://me.example",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,x-session-id",
    })
    assert res.status_code in (200, 204)
    assert "X-Session-Id" in res.headers["Access-Control-Allow-Headers"]


def test_reset_clears_session_but_keeps_knowledge_base(client):
    upload_pdf(client, ["Some policy content here."])
    chat(client, "hello there", ["hi"])
    kb_chunks = len(app_module.knowledge.chunks)

    res = client.post("/api/reset", headers=SID)
    assert res.status_code == 200
    sess = app_module.SESSIONS["test-session"]
    assert sess["history"] == []
    assert not sess["docs"].has_documents()
    assert len(app_module.knowledge.chunks) == kb_chunks
