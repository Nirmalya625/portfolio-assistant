# Portfolio Assistant :-

A chatbot that answers questions about my machine learning projects, embedded in my portfolio site. Recruiters can also attach a job description (PDF) and ask how my work matches it.

Flask backend, Groq LLM (streaming), and a from-scratch TF-IDF retrieval layer. No vector database or embedding API.

## How it works

```
portfolio page (widget)  ──►  Flask API  ──►  Groq
                                 │
            ┌────────────────────┴────────────────────┐
            │ knowledge/*.md  (about me, read-only)   │
            │ visitor's uploaded PDFs (per session)   │
            └──────────── TF-IDF cosine search ───────┘
```

1. `knowledge/*.md` is split into one chunk per `## ` section and indexed at startup.
2. Each question is searched against that index plus the visitor's own PDFs. The top matches are added to the prompt.
3. Short follow-ups ("and what else?") are retried together with the previous question.
4. If nothing matches, the model answers normally and is told not to invent facts about me.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
python app.py               # http://localhost:5050
```

Open `my_web.html` directly in a browser: on `localhost` and `file://` the widget talks to `http://localhost:5050`. Set `ALLOWED_ORIGINS=*` in `.env` for local testing.

## Deploy (Render)

- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app --workers 1 --threads 4`  (one worker: sessions live in memory)
- Environment: `GROQ_API_KEY`, `ALLOWED_ORIGINS=https://<your-portfolio-domain>`, optional `PORTFOLIO_URL`
- Then set `PROD_API` in `my_web.html` to the Render URL.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key | required |
| `GROQ_MODEL` | Model name | `openai/gpt-oss-20b` |
| `ALLOWED_ORIGINS` | Comma-separated origins allowed to call the API | none |
| `RATE_LIMIT_PER_MIN` | Requests per IP per minute | `20` |
| `PORTFOLIO_URL` | Adds a "Portfolio" link to the full-page UI | none |

## Update what it knows

Edit `knowledge/nirmalya.md` and restart. Start each new topic with a `## ` heading. Anything not written there, the assistant will say it doesn't know.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

Covers retrieval, grounding, per-visitor session isolation, CORS, rate limiting, and LLM failure handling. The Groq client is mocked, so no API key or network is needed.

## Layout

```
app.py              Flask API, sessions, prompt building
rag.py              TF-IDF index (chunking, cosine search)
knowledge/          Markdown the assistant answers from
templates/index.html  Full-page version of the assistant
my_web.html         Portfolio page with the embedded widget
tests/              pytest suite
```
