"""
Syntra.AI — Roadmap Factory Microservice
POST /generate-roadmap
"""

import os
import re
import json
import asyncio
import logging
import threading
from typing import List

import google.generativeai as genai
from duckduckgo_search import DDGS
from youtubesearchpython import VideosSearch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("roadmap_factory")

# ── Config ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
YT_LIMIT = int(os.getenv("YT_SEARCH_LIMIT", "5"))
YT_MIN_SEC = int(os.getenv("YT_MIN_DURATION_SEC", "600"))
CONCURRENCY = int(os.getenv("ENRICH_CONCURRENCY", "10"))

SKIP_DOMAINS = ("pinterest.", "facebook.", "twitter.", "x.com", "instagram.", "tiktok.")
SEARCH_PAGES = ("/search?", "/search/", "google.com/search", "youtube.com/results", "bing.com/search", "duckduckgo.com/")
YT_GOOD = ("tutorial", "course", "learn", "explained", "guide", "introduction", "basics", "fundamentals", "full", "complete")
YT_BAD = ("shorts", "in 5 minutes", "in 10 minutes", "speed run", "trailer", "reaction", "tiktok")
ART_GOOD = ("tutorial", "guide", "documentation", "docs", "learn", "introduction")

_ddg_lock = threading.Lock()

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Check your .env file.")
genai.configure(api_key=GEMINI_API_KEY)
logger.info("Gemini model: %s", GEMINI_MODEL)

# ── Schemas ───────────────────────────────────────────────────────────────────
class RoadmapRequest(BaseModel):
    track_name: str = Field(..., example="Frontend Web Development")
    hours_per_week: int = Field(..., ge=1, le=168, example=10)

class ResourceSchema(BaseModel):
    youtube_link: str
    book_reference: str
    article_link: str

class SkillSchema(BaseModel):
    skill_name: str
    estimated_hours: float
    resources: ResourceSchema

class WeekSchema(BaseModel):
    week_number: int
    skills: List[SkillSchema]

class RoadmapResponse(BaseModel):
    track_name: str
    user_hours_per_week: int
    total_weeks_calculated: int
    roadmap: List[WeekSchema]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _norm(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", (text or "").lower()).strip()

def _dur(s: str) -> int:
    if not s:
        return 0
    try:
        p = [int(x) for x in s.strip().split(":")]
        if len(p) == 3:
            return p[0] * 3600 + p[1] * 60 + p[2]
        if len(p) == 2:
            return p[0] * 60 + p[1]
        return p[0]
    except ValueError:
        return 0

def _ok_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    if u.lower() in {"", "null", "none", "n/a", "#", '""', "https://", "http://"}:
        return False
    if not u.startswith(("http://", "https://")) or len(u) <= 12:
        return False
    return not any(p in u.lower() for p in SEARCH_PAGES)

def _skip_url(url: str) -> bool:
    return any(d in url.lower() for d in SKIP_DOMAINS)

def _yt_watch(url: str) -> bool:
    ul = url.lower()
    return "youtube.com/watch" in ul or "youtu.be/" in ul

def _score(text: str, skill: str, good: tuple, bad: tuple = (), duration: int = 0) -> float:
    t, sn = _norm(text), _norm(skill)
    tokens = {w for w in sn.split() if len(w) > 2}
    s = len(tokens & set(t.split())) * 3.0 + (10.0 if sn and sn in t else 0.0)
    s += sum(1.5 for k in good if k in t)
    s -= sum(4.0 for k in bad if k in t)
    if duration >= 3600: s += 10
    elif duration >= 1800: s += 6
    elif duration >= YT_MIN_SEC: s += 2
    elif 0 < duration < 180: s -= 10
    return s

def _ddg(query: str) -> list[dict]:
    try:
        with _ddg_lock:
            with DDGS() as d:
                return list(d.text(query, max_results=5))
    except Exception as exc:
        logger.debug("DDG fail '%s': %s", query, exc)
        return []

# ── Step 1: Gemini ────────────────────────────────────────────────────────────
_GEMINI_SYSTEM = """\
You are an expert curriculum designer. When given a learning track name, you generate
a comprehensive, ordered list of skills a learner needs to master, together with
accurate learning-resource metadata.

You MUST respond with a single valid JSON object — no markdown fences, no prose.
The JSON schema is:

{
  "track_name": "<string>",
  "skills": [
    {
      "skill_name": "<string>",
      "estimated_hours": <number>,
      "resources": {
        "youtube_link": "",
        "book_reference": "<Author – Book Title – Chapter X>",
        "article_link": "<https://...>"
      }
    }
  ]
}

Rules:
- Include 10-20 skills ordered from fundamentals to advanced.
- estimated_hours must be a realistic positive number (e.g. 2, 4, 6…).
- youtube_link MUST always be an empty string "".
- book_reference must be a real reference (Author – Book Title – Chapter X). Never leave it empty.
- article_link is REQUIRED for every skill. It MUST be a full https:// URL to a real article,
  tutorial, or official documentation page. NEVER leave article_link as "" or a placeholder.
- Prefer official docs and reputable sources (MDN, react.dev, python.org docs, freeCodeCamp, Dev.to).
- Do NOT wrap the JSON in triple backticks or any markdown.

CRITICAL — track alignment (must follow):
- Every skill MUST belong directly to the requested track_name. Do NOT output a generic or wrong track.
- Set track_name in JSON to exactly the track you were asked for.
- "Backend Web Development" → server-side only: HTTP/REST, databases, SQL, APIs, auth, backends, Docker, testing, deployment.
  Do NOT list HTML, CSS, React, or frontend UI as skills.
- "Frontend Web Development" → client-side: HTML, CSS, JavaScript, frameworks, UI, browser APIs.
  Do NOT list backend-only topics unless directly needed for frontend.
- Match the track domain exactly (DevOps, Data Science, Mobile, etc.) — never swap tracks.
"""

_TRACK_HINTS = {
    "backend": "Focus on HTTP/REST, SQL, APIs, auth, server frameworks, Docker. Exclude HTML/CSS/React.",
    "frontend": "Focus on HTML, CSS, JavaScript, React/Vue, UI. Exclude backend-only topics.",
}

async def _call_gemini(track_name: str) -> dict:
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=_GEMINI_SYSTEM)

    hint = ""
    tn = track_name.lower()
    if "backend" in tn:
        hint = " " + _TRACK_HINTS["backend"]
    elif "frontend" in tn:
        hint = " " + _TRACK_HINTS["frontend"]

    prompt = (
        f'Generate a complete learning roadmap ONLY for the track: "{track_name}". '
        f"Every skill must be specific to this track.{hint} "
        f'Set track_name in JSON to exactly: "{track_name}".'
    )
    logger.info("Calling Gemini for track: %s", track_name)
    try:
        resp = await asyncio.to_thread(
            model.generate_content, prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=8192),
        )
    except Exception as exc:
        msg = str(exc)
        if "429" in msg or "quota" in msg.lower():
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Gemini quota exceeded (model: {GEMINI_MODEL}). "
                    "Free tier is ~20 requests/day per model. "
                    "Wait a minute and retry, try GEMINI_MODEL=gemini-2.0-flash or gemini-2.5-flash-lite in .env, "
                    "or enable billing: https://ai.google.dev/gemini-api/docs/rate-limits"
                ),
            )
        raise HTTPException(status_code=502, detail=f"LLM service error: {exc}")

    raw = resp.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="LLM returned malformed JSON. Please retry.")
    if not isinstance(data.get("skills"), list):
        raise HTTPException(status_code=502, detail="LLM response missing 'skills' array.")
    data["track_name"] = track_name
    return data

# ── Step 2: Fast resource enrichment ─────────────────────────────────────────
def _fetch_yt(query: str) -> list:
    try:
        return VideosSearch(query, limit=YT_LIMIT).result().get("result", [])
    except Exception:
        return []

def _best_yt(videos: list, skill: str) -> dict | None:
    pool = []
    for v in videos:
        title = (v.get("title") or "").lower()
        link = v.get("link") or ""
        d = _dur(v.get("duration") or "")
        if "#shorts" in title or "/shorts" in link or not _yt_watch(link):
            continue
        if d and d < YT_MIN_SEC:
            continue
        pool.append(v)
    if not pool:
        pool = [v for v in videos if _yt_watch(v.get("link") or "") and "/shorts" not in (v.get("link") or "")]
    if not pool:
        return None
    return max(pool, key=lambda v: (_score(v.get("title") or "", skill, YT_GOOD, YT_BAD, _dur(v.get("duration") or "")), _dur(v.get("duration") or "")))

def _search_youtube(skill: str) -> str:
    for q in (f"{skill} full course tutorial", f"{skill} tutorial"):
        pick = _best_yt(_fetch_yt(q), skill)
        if pick and pick.get("link"):
            return pick["link"]
    return ""

def _article_from_ddg(skill: str) -> str:
    best_url, best_score = "", -1.0
    for r in _ddg(f"{skill} tutorial"):
        url, title = r.get("href", ""), r.get("title", "")
        if not _ok_url(url) or _skip_url(url):
            continue
        sc = _score(f"{title} {url}", skill, ART_GOOD)
        if sc > best_score:
            best_score, best_url = sc, url
    return best_url

def _resolve_article(skill: str, candidate: str) -> str:
    if _ok_url(candidate) and not _skip_url(candidate):
        return candidate.strip()
    return _article_from_ddg(skill)

def _enrich_skill(skill: dict) -> dict:
    res = skill.setdefault("resources", {})
    name = skill["skill_name"]
    res["youtube_link"] = _search_youtube(name)
    res["article_link"] = _resolve_article(name, res.get("article_link", ""))
    return skill

async def _enrich_resources(skills: list) -> list:
    sem = asyncio.Semaphore(CONCURRENCY)
    async def run(s):
        async with sem:
            return await asyncio.to_thread(_enrich_skill, s)
    return list(await asyncio.gather(*[run(s) for s in skills]))

# ── Step 3: Weekly schedule ───────────────────────────────────────────────────
def _build_weekly_schedule(skills: list, hours_per_week: int) -> tuple[List[WeekSchema], int]:
    weeks, cur, hrs, num = [], [], 0.0, 1
    for raw in skills:
        try:
            skill = SkillSchema(skill_name=raw["skill_name"], estimated_hours=float(raw["estimated_hours"]),
                                resources=ResourceSchema(**raw["resources"]))
        except Exception as exc:
            logger.warning("Skipping skill %s: %s", raw.get("skill_name"), exc)
            continue
        if cur and hrs + skill.estimated_hours > hours_per_week:
            weeks.append(WeekSchema(week_number=num, skills=cur))
            num, cur, hrs = num + 1, [], 0.0
        cur.append(skill)
        hrs += skill.estimated_hours
    if cur:
        weeks.append(WeekSchema(week_number=num, skills=cur))
    return weeks, len(weeks)

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Syntra.AI — Roadmap Factory", version="1.0.0",
              description="Generates a time-scaled roadmap with verified resources.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Syntra.AI Roadmap Factory"}

@app.post("/generate-roadmap", response_model=RoadmapResponse)
async def generate_roadmap(request: RoadmapRequest) -> RoadmapResponse:
    logger.info("=== track='%s' | %d hrs/week ===", request.track_name, request.hours_per_week)
    data = await _call_gemini(request.track_name)
    skills = await _enrich_resources(data["skills"])
    weeks, total = _build_weekly_schedule(skills, request.hours_per_week)
    logger.info("Done: %d skills, %d weeks", len(skills), total)
    return RoadmapResponse(track_name=request.track_name, user_hours_per_week=request.hours_per_week,
                           total_weeks_calculated=total, roadmap=weeks)
