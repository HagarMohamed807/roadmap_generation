"""
Syntra.AI — Roadmap Factory Microservice
POST /generate-roadmap

Pipeline:
  1. Intelligence  → Gemini 1.5 Flash generates a structured roadmap JSON
  2. Verification  → youtube-search-python fills in real tutorial links
  3. Personalization → hours_per_week groups skills into weekly schedules
"""

import os
import json
import asyncio
import logging
from typing import List

import google.generativeai as genai
from youtubesearchpython import VideosSearch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config & Logging
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("roadmap_factory")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
YT_MAX_RESULTS = int(os.getenv("YT_MAX_RESULTS", "3"))

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class RoadmapRequest(BaseModel):
    track_name: str      = Field(..., example="Frontend Web Development")
    hours_per_week: int  = Field(..., ge=1, le=168, example=10)


class ResourceSchema(BaseModel):
    youtube_link:    str
    book_reference:  str
    article_link:    str


class SkillSchema(BaseModel):
    skill_name:      str
    estimated_hours: float
    resources:       ResourceSchema


class WeekSchema(BaseModel):
    week_number: int
    skills:      List[SkillSchema]


class RoadmapResponse(BaseModel):
    track_name:             str
    user_hours_per_week:    int
    total_weeks_calculated: int
    roadmap:                List[WeekSchema]


# ---------------------------------------------------------------------------
# Step 1 — Intelligence: Gemini generates a flat skill list
# ---------------------------------------------------------------------------

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
- book_reference and article_link must be real, accurate references.
- Do NOT wrap the JSON in triple backticks or any markdown.
"""


async def _call_gemini(track_name: str) -> dict:
    """Call Gemini 1.5 Flash and parse response as JSON."""
    model  = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=_GEMINI_SYSTEM,
    )

    prompt = f'Generate a complete learning roadmap for the track: "{track_name}"'

    logger.info("Calling Gemini for track: %s", track_name)
    try:
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=8192,
            ),
        )
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"LLM service error: {exc}",
        )

    raw_text = response.text.strip()

    # Strip accidental markdown fences if Gemini disobeys
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Gemini JSON: %s\nRaw:\n%s", exc, raw_text)
        raise HTTPException(
            status_code=502,
            detail="LLM returned malformed JSON. Please retry.",
        )

    if "skills" not in data or not isinstance(data["skills"], list):
        raise HTTPException(
            status_code=502,
            detail="LLM response missing 'skills' array.",
        )

    return data


# ---------------------------------------------------------------------------
# Step 2 — Verification: YouTube search fills in real links
# ---------------------------------------------------------------------------

def _search_youtube(query: str) -> str:
    """Return the first YouTube watch URL for `query`, or empty string on failure."""
    try:
        results = VideosSearch(query, limit=YT_MAX_RESULTS).result()
        videos  = results.get("result", [])
        if videos:
            link = videos[0].get("link", "")
            logger.info("YT link found for '%s': %s", query, link)
            return link
    except Exception as exc:
        logger.warning("YouTube search failed for '%s': %s", query, exc)
    return ""


async def _enrich_with_youtube(skills: list, track_name: str) -> list:
    """Run YouTube searches in a thread pool to avoid blocking the event loop."""

    def _enrich_single(skill: dict) -> dict:
        query = f"{skill['skill_name']} {track_name} tutorial"
        skill["resources"]["youtube_link"] = _search_youtube(query)
        return skill

    enriched = await asyncio.gather(
        *[asyncio.to_thread(_enrich_single, skill) for skill in skills]
    )
    return list(enriched)


# ---------------------------------------------------------------------------
# Step 3 — Personalization: group skills into weekly schedule
# ---------------------------------------------------------------------------

def _build_weekly_schedule(
    skills: list,
    hours_per_week: int,
) -> tuple[List[WeekSchema], int]:
    """
    Pack skills greedily into weeks.

    A skill is never split across weeks — it is placed into the current week
    if it fits, otherwise a new week is started.  This keeps each week
    coherent.  The last week may run slightly over the target if a single
    skill exceeds hours_per_week; in that case the skill occupies its own week.
    """
    weeks: List[WeekSchema] = []
    current_week_skills: List[SkillSchema] = []
    current_week_hours: float = 0.0
    week_num = 1

    for raw in skills:
        try:
            res = ResourceSchema(**raw["resources"])
            skill = SkillSchema(
                skill_name=raw["skill_name"],
                estimated_hours=float(raw["estimated_hours"]),
                resources=res,
            )
        except Exception as exc:
            logger.warning("Skipping malformed skill %s: %s", raw.get("skill_name"), exc)
            continue

        # If adding this skill would exceed the weekly budget AND we already
        # have skills in the current week, close the week first.
        if current_week_skills and (current_week_hours + skill.estimated_hours) > hours_per_week:
            weeks.append(WeekSchema(week_number=week_num, skills=current_week_skills))
            week_num += 1
            current_week_skills = []
            current_week_hours  = 0.0

        current_week_skills.append(skill)
        current_week_hours += skill.estimated_hours

    # Flush the last week
    if current_week_skills:
        weeks.append(WeekSchema(week_number=week_num, skills=current_week_skills))

    return weeks, len(weeks)


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Syntra.AI — Roadmap Factory",
    description=(
        "Generates a time-scaled learning roadmap with verified YouTube resources. "
        "Called by the backend only when no cached roadmap exists."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "service": "Syntra.AI Roadmap Factory"}


@app.post(
    "/generate-roadmap",
    response_model=RoadmapResponse,
    tags=["Roadmap"],
    summary="Generate a complete, time-scaled roadmap with verified resources",
)
async def generate_roadmap(request: RoadmapRequest) -> RoadmapResponse:
    """
    **Pipeline:**
    1. Gemini 1.5 Flash → structured skill list (JSON)
    2. youtube-search-python → fills real tutorial links per skill
    3. Time-scaling → groups skills into weekly schedule by `hours_per_week`
    """
    # --- Step 1: Intelligence ---
    logger.info("=== NEW REQUEST | track='%s' | hours_per_week=%d ===",
                request.track_name, request.hours_per_week)

    gemini_data = await _call_gemini(request.track_name)
    raw_skills: list = gemini_data["skills"]
    logger.info("Gemini returned %d skills.", len(raw_skills))

    # --- Step 2: Verification ---
    logger.info("Enriching skills with YouTube links …")
    enriched_skills = await _enrich_with_youtube(raw_skills, request.track_name)

    # --- Step 3: Personalization ---
    logger.info("Building weekly schedule (budget: %d hrs/week) …", request.hours_per_week)
    weeks, total_weeks = _build_weekly_schedule(enriched_skills, request.hours_per_week)
    logger.info("Schedule built: %d weeks total.", total_weeks)

    return RoadmapResponse(
        track_name=request.track_name,
        user_hours_per_week=request.hours_per_week,
        total_weeks_calculated=total_weeks,
        roadmap=weeks,
    )
