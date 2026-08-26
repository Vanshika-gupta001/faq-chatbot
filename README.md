# FAQ Bot — Customizable AI Chatbot for Any Business

A lightweight, deployable FAQ chatbot built with **FastAPI** + **scikit-learn**.
Give it a business's FAQ list, and it instantly answers customer questions —
no fine-tuning, no paid LLM API key required to run the core engine.

This is a **portfolio/demo project** meant to show a working, end-to-end
product: backend API, retrieval logic, and an embeddable chat widget — the
exact shape of a "custom FAQ bot for your website" freelance gig.

---

## What it does

- Loads a business's FAQs (`data/faqs.json`)
- Matches a customer's question to the closest known FAQ using TF-IDF +
  cosine similarity
- Returns the matched answer, or a graceful fallback if nothing matches well
- Ships with a ready-to-use chat widget (`static/widget.html`) that can be
  embedded on any website via an iframe
- New businesses can swap in their own FAQs via the `POST /faqs` endpoint —
  this is what makes the bot reusable across different freelance clients
  without touching the code

## Tech stack

`FastAPI` · `scikit-learn` (TF-IDF retrieval) · `Pydantic` · vanilla
`HTML/CSS/JS` widget (no frontend framework needed — easy to embed anywhere)

---

## Run it locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open:
- `http://127.0.0.1:8000/widget` — the demo chat widget
- `http://127.0.0.1:8000/docs` — interactive API docs (Swagger UI)

Try asking the widget things like *"what are your store hours?"* or
*"how long does shipping take?"*

---

## API endpoints

| Method | Path      | What it does                                  |
|--------|-----------|------------------------------------------------|
| GET    | `/`       | Health check + current business info           |
| POST   | `/chat`   | Send `{"message": "..."}`, get back an answer  |
| GET    | `/faqs`   | View the currently loaded FAQ set               |
| POST   | `/faqs`   | Replace the FAQ set (for onboarding a new client) |
| GET    | `/widget` | Serves the demo chat widget page                |

---

## Customizing for a new client

Send a `POST /faqs` request with the client's own business name and FAQ
list, in this shape:

```json
{
  "business_name": "Client's Business Name",
  "faqs": [
    { "question": "Do you offer free trials?", "answer": "Yes, 14 days, no card required." }
  ]
}
```

The bot immediately starts answering from the new FAQ set — this is the
core pitch for freelance clients: **"send me your FAQs, get a working
chatbot the same day."**

---

## Deploying for free (to show clients a live link)

This project is a plain FastAPI app, so it deploys cleanly to any of these
free tiers:

- **Render** (render.com) — connect the GitHub repo, set the start command
  to `uvicorn main:app --host 0.0.0.0 --port $PORT`, done.
- **Railway** (railway.app) — similar one-click deploy from GitHub.

Once deployed, the `/widget` URL is something you can literally send a
prospective client to try live.

---

## Known limitation (and how to talk about it in an interview/pitch)

TF-IDF matches on shared words, so it won't catch every paraphrase (e.g.
"can I get my money back" won't match "refund policy" unless the wording
overlaps). For production use with a paying client, the natural upgrade
path is swapping the retrieval step for sentence-embedding similarity
(e.g. `sentence-transformers`) or adding an LLM call for answer generation
on top of the matched FAQ context (retrieval-augmented generation). This
version intentionally keeps the core lightweight and dependency-free so it
runs anywhere without needing a paid API key.

---

## Project structure

```
faq-bot/
├── main.py              # FastAPI app + TF-IDF retrieval engine
├── requirements.txt
├── data/
│   └── faqs.json         # sample FAQ set (swap this per client)
├── static/
│   └── widget.html        # embeddable chat widget frontend
└── README.md
```
