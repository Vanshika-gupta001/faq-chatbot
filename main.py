
"""
FAQ Bot - A customizable AI chatbot for business FAQs
--------------------------------------------------------
Built with FastAPI + scikit-learn (TF-IDF retrieval).

Any business can drop in their own FAQ list (JSON) and get a working
chatbot that answers customer questions from that content - no fine-tuning,
no external LLM API key required to run the core matching engine.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/widget to try the demo chat widget.
"""

import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "faqs.json"

app = FastAPI(
    title="FAQ Bot API",
    description="A pluggable FAQ chatbot for any business - upload your FAQs, get instant Q&A.",
    version="1.0.0",
)

# Allow the widget to be embedded on any website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ---------- Data models ----------

class FAQItem(BaseModel):
    question: str
    answer: str


class FAQSet(BaseModel):
    business_name: str
    faqs: List[FAQItem]


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    matched_question: Optional[str] = None
    confidence: float
    business_name: str


# ---------- Retrieval engine ----------

class FAQEngine:
    """Wraps a FAQ set with a TF-IDF retriever so questions can be matched
    against the closest known FAQ using cosine similarity."""

    def __init__(self, faq_set: FAQSet):
        self.business_name = faq_set.business_name
        self.faqs = faq_set.faqs
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None
        self._fit()

    def _fit(self):
        questions = [f.question for f in self.faqs]
        if questions:
            self._matrix = self._vectorizer.fit_transform(questions)
        else:
            self._matrix = None

    def answer(self, user_message: str, threshold: float = 0.2) -> ChatResponse:
        if not self.faqs or self._matrix is None:
            return ChatResponse(
                answer="No FAQs are loaded yet for this business.",
                confidence=0.0,
                business_name=self.business_name,
            )

        user_vec = self._vectorizer.transform([user_message])
        scores = cosine_similarity(user_vec, self._matrix)[0]
        best_idx = scores.argmax()
        best_score = float(scores[best_idx])

        if best_score < threshold:
            return ChatResponse(
                answer="I'm not sure about that one. Could you rephrase, or "
                       "contact our support team directly for help?",
                confidence=round(best_score, 3),
                business_name=self.business_name,
            )

        matched = self.faqs[best_idx]
        return ChatResponse(
            answer=matched.answer,
            matched_question=matched.question,
            confidence=round(best_score, 3),
            business_name=self.business_name,
        )


def load_faq_engine() -> FAQEngine:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    faq_set = FAQSet(**raw)
    return FAQEngine(faq_set)


engine = load_faq_engine()


# ---------- Routes ----------

@app.get("/", tags=["meta"])
def root():
    return {
        "status": "ok",
        "business_name": engine.business_name,
        "faq_count": len(engine.faqs),
        "try_it": "/widget",
        "docs": "/docs",
    }


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    return engine.answer(req.message)


@app.get("/faqs", response_model=FAQSet, tags=["admin"])
def get_faqs():
    return FAQSet(business_name=engine.business_name, faqs=engine.faqs)


@app.post("/faqs", response_model=FAQSet, tags=["admin"])
def replace_faqs(faq_set: FAQSet):
    """Swap in a brand new FAQ set for a different business - this is what
    makes the bot reusable across clients without touching the code."""
    global engine
    engine = FAQEngine(faq_set)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(faq_set.model_dump(), f, indent=2)
    return faq_set


@app.get("/widget", response_class=HTMLResponse, tags=["demo"])
def widget_page():
    return FileResponse(str(BASE_DIR / "static" / "widget.html"))
