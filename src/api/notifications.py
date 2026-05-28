# src/api/notifications.py
# ─────────────────────────────────────────────────────────────────────
# Stub notification router — extend with SSE/WebSocket when needed.
# This file MUST exist to prevent analytics_api.py from crashing on import.
# ─────────────────────────────────────────────────────────────────────
from fastapi import APIRouter

router = APIRouter(prefix="/api/notifications", tags=["Notifications"])


@router.get("/health")
async def notif_health():
    """Notification system health check."""
    return {"status": "ok", "message": "Notifications stub active"}