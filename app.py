import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from groq import RateLimitError
from pydantic import BaseModel

from agent import analyze_text

BASE_DIR = Path(__file__).parent
PAGE = BASE_DIR / "docs" / "index.html"

# Only these websites may call the API from a browser.
# On Render, set ALLOWED_ORIGINS=https://<your-username>.github.io
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    if o.strip()
]

# Simple per-visitor limit so strangers can't burn through your Groq quota.
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "10"))
_recent: dict[str, deque] = defaultdict(deque)

app = FastAPI(title="Toxic Language Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class AnalyzeRequest(BaseModel):
    text: str


def _client_ip(request: Request) -> str:
    # Render/most hosts put the real visitor IP in X-Forwarded-For.
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def _check_rate_limit(ip: str) -> None:
    now = time.time()
    q = _recent[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Slow down! Try again in a minute.")
    q.append(now)


@app.get("/")
def index():
    return FileResponse(PAGE)


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, request: Request):
    _check_rate_limit(_client_ip(request))
    try:
        return analyze_text(req.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Too many requests right now. Wait a few seconds and try again.")
    except Exception:
        raise HTTPException(status_code=502, detail="Analysis failed. Please try again.")


@app.get("/health")
def health():
    return {"status": "ok"}
