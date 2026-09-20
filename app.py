import os
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response, send_from_directory, stream_with_context
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

from rag import DocumentIndex

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024  # 3 MB uploads

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Origins allowed to call the API from the browser (your portfolio's URL).
# Comma-separated in .env, e.g. ALLOWED_ORIGINS=https://nirmalya625.github.io
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

MAX_MESSAGE_CHARS = 1000
MAX_HISTORY_TURNS = 10
MAX_SESSIONS = 200
MAX_UPLOADS = 3
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))

KB_DIR = Path(__file__).parent / "knowledge"

SYSTEM_PROMPT = (
    "You are the assistant on Nirmalya Sarkar's machine learning portfolio site. "
    "Visitors are mostly recruiters, hiring managers, and engineers. Talk about "
    "Nirmalya in the third person. Be concise and concrete: lead with the answer, "
    "2-5 sentences unless asked for more, name specific projects, models, and "
    "numbers.\n\n"
    "Background: Nirmalya builds end-to-end ML projects (data, notebook, backend, "
    "app) across fraud detection, forecasting, recommendation, simulation, and NLP. "
    "Stack: Python, Flask, scikit-learn, Node.js. GitHub: "
    "https://github.com/Nirmalya625/nirmalya-sarkar  LinkedIn: "
    "https://www.linkedin.com/in/nirmalyasarkar/\n\n"
    "Rules:\n"
    "- State facts about Nirmalya only if they appear in this prompt or in a "
    "'Document excerpts' section labelled 'About Nirmalya'. If something isn't "
    "there (salary, availability, education, contact details, private matters), "
    "say you don't have it and point to his LinkedIn. Never invent.\n"
    "- Excerpts labelled '[Source: <file>, chunk N]' come from a PDF the visitor "
    "uploaded. Answer from them and cite like [Source: filename, chunk N]. If the "
    "visitor uploaded a job description, map specific projects to specific "
    "requirements and say plainly where there are gaps.\n"
    "- Don't cite anything for 'About Nirmalya' excerpts; just answer naturally.\n"
    "- For unrelated general questions, answer briefly, then offer to steer back "
    "to Nirmalya's work."
)

# Read-only knowledge base about Nirmalya, indexed once at startup.
knowledge = DocumentIndex()


def load_knowledge():
    knowledge.clear()
    for path in sorted(KB_DIR.glob("*.md")):
        knowledge.add_sections(path.name, path.read_text(encoding="utf-8"))


load_knowledge()

# Per-visitor state: chat history + that visitor's uploaded PDFs. Keyed by the
# X-Session-Id header so one visitor's uploads are never searchable by another.
SESSIONS = OrderedDict()
RATE = defaultdict(deque)


def client_ip():
    # Last X-Forwarded-For entry = the address the platform's proxy saw
    # (the first entry can be spoofed by the client).
    return request.headers.get("X-Forwarded-For", "").split(",")[-1].strip() or request.remote_addr or "?"


def get_session():
    # Clients without an X-Session-Id header get a per-IP session instead of one shared by everyone.
    sid = (request.headers.get("X-Session-Id") or f"ip:{client_ip()}")[:64]
    sess = SESSIONS.get(sid)
    if sess is None:
        sess = {"history": [], "docs": DocumentIndex()}
        SESSIONS[sid] = sess
        while len(SESSIONS) > MAX_SESSIONS:
            SESSIONS.popitem(last=False)
    SESSIONS.move_to_end(sid)
    return sess


def rate_limited():
    ip = client_ip()
    now = time.time()
    if len(RATE) > 5000:
        RATE.clear()
    hits = RATE[ip]
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        return True
    hits.append(now)
    return False


@app.after_request
def add_cors(resp):
    origin = request.headers.get("Origin")
    if origin and ("*" in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS):
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Session-Id"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Vary"] = "Origin"
    return resp


def extract_pdf_text(file_storage):
    reader = PdfReader(file_storage)
    parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(parts)


@app.route("/")
def portfolio_page():
    # The portfolio frontend (templates/my_web.html). Sent as a plain file, not run
    # through Jinja, so braces in its CSS/JS can never break it.
    return send_from_directory(Path(app.root_path) / "templates", "my_web.html")


@app.route("/assistant")
def assistant_page():
    # Full-page version of the assistant. PORTFOLIO_URL overrides the "Portfolio" link target.
    return render_template("index.html", portfolio_url=os.getenv("PORTFOLIO_URL", "/"))


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "kb_chunks": len(knowledge.chunks)})


@app.route("/api/upload", methods=["POST"])
def upload():
    if rate_limited():
        return jsonify({"error": "Too many requests — try again in a minute."}), 429

    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file provided"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    sess = get_session()
    if len(sess["docs"].documents) >= MAX_UPLOADS:
        return jsonify({"error": f"Limit of {MAX_UPLOADS} PDFs per session — reset to start over."}), 400

    try:
        text = extract_pdf_text(file)
    except Exception as exc:  # noqa: BLE001 — surface any parse failure to the user
        return jsonify({"error": f"Could not read PDF: {exc}"}), 400

    if not text.strip():
        return jsonify({
            "error": "No extractable text found — this PDF may be scanned/image-only."
        }), 400

    added = sess["docs"].add_document(file.filename, text)

    return jsonify({
        "status": "indexed",
        "filename": file.filename,
        "chunks_added": added,
        "documents": sess["docs"].documents,
    })


@app.route("/api/documents", methods=["GET"])
def documents():
    return jsonify({"documents": get_session()["docs"].documents})


def retrieve(query, sess, top_k=4):
    results = knowledge.search(query, top_k=3) + sess["docs"].search(query, top_k=3)
    results.sort(key=lambda pair: pair[0], reverse=True)
    return results[:top_k]


def build_user_message(message, sess, prev_question=""):
    """
    Wrap the message with retrieved excerpts (Nirmalya's knowledge base and/or
    the visitor's uploaded PDFs). If nothing is relevant, send it plain.
    Short follow-ups ("tell me more") are retried with the previous question.
    """
    results = retrieve(message, sess)
    if not results and prev_question:
        results = retrieve(f"{prev_question} {message}", sess)

    if not results:
        return {"role": "user", "content": message}

    blocks = []
    for _score, chunk in results:
        if "title" in chunk:
            label = f"About Nirmalya: {chunk['title']}"
        else:
            label = f"Source: {chunk['doc']}, chunk {chunk['chunk_id']}"
        blocks.append(f"[{label}]\n{chunk['text']}")

    return {
        "role": "user",
        "content": (
            "Document excerpts:\n\n" + "\n\n---\n\n".join(blocks) + "\n\n"
            f"Question: {message}\n\n"
            "Answer using the excerpts above where they are relevant. Cite "
            "[Source: filename, chunk N] only for uploaded-PDF excerpts."
        ),
    }


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "message is required"}), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify({"error": f"Message too long (max {MAX_MESSAGE_CHARS} characters)."}), 400
    if rate_limited():
        return jsonify({"error": "Too many requests — try again in a minute."}), 429

    sess = get_session()
    history = sess["history"]
    prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")

    model_message = build_user_message(message, sess, prev)
    messages_for_model = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history[-MAX_HISTORY_TURNS * 2:]
        + [model_message]
    )

    def generate():
        reply, failed = "", False
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages_for_model,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    reply += delta
                    yield delta
        except Exception:  # noqa: BLE001
            failed = True
            app.logger.exception("Groq request failed")
            yield "Sorry, the language model didn't respond. Please try again in a moment."
        finally:
            # Store the plain question (not the excerpt-wrapped one) so history stays small.
            if reply and not failed:
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": reply})
                del history[:-MAX_HISTORY_TURNS * 4]

    resp = Response(stream_with_context(generate()), mimetype="text/plain")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"  # don't let proxies buffer the stream
    return resp


@app.route("/api/reset", methods=["POST"])
def reset():
    sess = get_session()
    sess["history"].clear()
    sess["docs"].clear()  # only this visitor's uploads; the knowledge base stays
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=int(os.getenv("PORT", "5050")))