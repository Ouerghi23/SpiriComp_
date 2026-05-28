"""
src/nlp/nlp_api.py
FIX-CITY: if NLP doesn't detect a city from text, use user-entered city as fallback
All other code identical.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.nlp.multilingual_nlp_pipeline import MultilingualNLPPipeline
from src.nlp.complaint_db import ComplaintDB

router = APIRouter(tags=["Complaints", "NLP", "Analytics"])
_pipe = MultilingualNLPPipeline()
_db   = ComplaintDB()


class ComplaintSubmit(BaseModel):
    text:    str           = Field(..., min_length=5, max_length=3000)
    msisdn:  Optional[str] = None
    city:    Optional[str] = None
    segment: Optional[str] = None
    channel: Optional[str] = "web"

class StatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(open|in_progress|resolved)$")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """<html><body style="font-family:Arial;max-width:500px;margin:40px auto">
    <h2>Huawei SpiriCom NLP API</h2>
    <ul>
      <li><a href="/form">Complaint Form</a></li>
      <li><a href="/docs">API Documentation</a></li>
      <li><a href="/api/complaints/stats">Live Stats</a></li>
      <li><a href="/api/complaints">All Complaints</a></li>
    </ul>
    </body></html>"""


@router.post("/api/complaints/submit", tags=["Complaints"])
async def submit_complaint(c: ComplaintSubmit):
    nlp = _pipe.analyze(c.text)
    cid = _db._generate_id()

    record = {
        "complaint_id":  cid,
        "submitted_at":  datetime.now().isoformat(),
        "msisdn":        c.msisdn,
        "city_input":    c.city,
        "segment":       c.segment,
        "channel":       c.channel or "web",
        "text_original": c.text,
        **nlp,
    }

    # ── FIX-CITY ──────────────────────────────────────────────────────
    # If the NLP pipeline didn't detect a city from the message text,
    # fall back to the city the user typed in the form field.
    # Without this, nlp_city is always empty when text doesn't mention
    # a Tunisian city name explicitly.
    if not record.get("city") and c.city:
        record["city"] = c.city
    # ──────────────────────────────────────────────────────────────────

    _db.insert(record)

    resp_hours = {"très urgent": 2, "urgent": 8, "normal": 24}.get(nlp["urgency_level"], 24)
    lang_label = {"ar": "العربية", "fr": "Français", "en": "English"}.get(nlp["language"], nlp["language"])

    return {
        "complaint_id":             cid,
        "is_complaint":             nlp.get("is_complaint"),
        "language_detected":        lang_label,
        "category":                 nlp["category"],
        "sentiment":                nlp["sentiment"],
        "urgency_level":            nlp["urgency_level"],
        "urgency_score":            nlp["urgency_score"],
        "city_detected":            record.get("city"),   # ← returns user city if NLP found nothing
        "estimated_response_hours": resp_hours,
        "message": (
            f"{'Réclamation' if nlp.get('is_complaint') else 'Feedback'} "
            f"enregistré (ID: {cid}). "
            f"{'Délai: ' + str(resp_hours) + 'h.' if nlp.get('is_complaint') else 'Merci.'}"
        ),
    }


@router.post("/api/complaints/analyze", tags=["NLP"])
async def analyze_only(c: ComplaintSubmit):
    return _pipe.analyze(c.text)


@router.get("/api/complaints/stats", tags=["Analytics"])
async def get_stats():
    return _db.stats()


@router.get("/api/complaints", tags=["Complaints"])
async def list_complaints(
    language:     Optional[str]  = Query(None, examples=["ar"]),
    urgency:      Optional[str]  = Query(None, examples=["urgent"]),
    sentiment:    Optional[str]  = Query(None, examples=["critique"]),
    status:       Optional[str]  = Query(None, examples=["open"]),
    is_complaint: Optional[bool] = Query(None),
    limit:        int             = Query(100, le=500),
):
    df = _db.to_dataframe(
        language=language, urgency=urgency, sentiment=sentiment,
        status=status, is_complaint=is_complaint, limit=limit,
    )
    if df.empty:
        return {"total": 0, "complaints": []}
    return {"total": len(df), "complaints": df.to_dict(orient="records")}


@router.get("/api/complaints/{complaint_id}", tags=["Complaints"])
async def get_complaint(complaint_id: str):
    df  = _db.to_dataframe(limit=10000)
    row = df[df["complaint_id"] == complaint_id]
    if row.empty:
        raise HTTPException(404, f"Complaint {complaint_id} not found")
    return row.iloc[0].to_dict()


@router.put("/api/complaints/{complaint_id}/status", tags=["Complaints"])
async def update_status(complaint_id: str, body: StatusUpdate):
    _db.update_status(complaint_id, body.status)
    return {"complaint_id": complaint_id, "status": body.status}


@router.delete("/api/complaints/{complaint_id}", tags=["Complaints"])
async def delete_complaint(complaint_id: str):
    with _db._conn() as conn:
        cursor = conn.execute(
            "DELETE FROM complaints WHERE complaint_id = ?", (complaint_id,)
        )
        if cursor.rowcount == 0:
            raise HTTPException(404, f"Complaint {complaint_id} not found")
    return {"complaint_id": complaint_id, "deleted": True}