"""
complaint_db.py
================
SQLite database manager for Ooredoo NLP complaint storage.

Schema (v2 — adds is_complaint):
  complaints (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id        TEXT UNIQUE,
    submitted_at        TEXT,
    msisdn              TEXT,
    city_input          TEXT,
    segment             TEXT,
    channel             TEXT,
    text_original       TEXT,
    language            TEXT,
    nlp_category        TEXT,
    nlp_sentiment       TEXT,
    nlp_urgency_score   REAL,
    nlp_urgency_level   TEXT,
    nlp_city            TEXT,
    nlp_network_type    TEXT,
    nlp_keywords        TEXT,
    status              TEXT,
    is_complaint        INTEGER   ← NEW
                                        (1 = réclamation,
                                         0 = feedback,
                                         NULL = unknown)
  )

Migration:
    If the existing DB does not yet have is_complaint,
    _init_schema() adds it with ALTER TABLE … ADD COLUMN
    — safe to run on live databases.

NEW in stats():
    "complaint_count"
        — submissions where is_complaint = 1

    "non_complaint_count"
        — submissions where is_complaint = 0

    "by_type"
        — {"complaint": N, "feedback": N}
          (used by NLPAnalysis.jsx)
"""

from __future__ import annotations

import json
import sqlite3
import uuid

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

DB_PATH = Path("data/nlp/complaints.db")


class ComplaintDB:

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ─────────────────────────────────────────────────────────────
    # Context manager
    # ─────────────────────────────────────────────────────────────
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            yield conn
            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            conn.close()

    # ─────────────────────────────────────────────────────────────
    # Schema + migration
    # ─────────────────────────────────────────────────────────────
    def _init_schema(self) -> None:

        with self._conn() as conn:

            # Create table if it does not exist
            # (includes is_complaint)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS complaints (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    complaint_id        TEXT UNIQUE,
                    submitted_at        TEXT NOT NULL,
                    msisdn              TEXT,
                    city_input          TEXT,
                    segment             TEXT,
                    channel             TEXT DEFAULT 'web',
                    text_original       TEXT NOT NULL,
                    language            TEXT,
                    nlp_category        TEXT,
                    nlp_sentiment       TEXT,
                    nlp_urgency_score   REAL,
                    nlp_urgency_level   TEXT,
                    nlp_city            TEXT,
                    nlp_network_type    TEXT,
                    nlp_keywords        TEXT,
                    status              TEXT DEFAULT 'open',
                    is_complaint        INTEGER DEFAULT NULL
                )
            """)

            # ── Safe migration ────────────────────────────────
            # ALTER TABLE ADD COLUMN raises OperationalError
            # if already present.
            try:
                conn.execute("""
                    ALTER TABLE complaints
                    ADD COLUMN is_complaint INTEGER DEFAULT NULL
                """)

                logger.info(
                    "Migration: added is_complaint column "
                    "to existing table."
                )

            except sqlite3.OperationalError:
                pass

            # ── Indices ───────────────────────────────────────
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_submitted_at
                ON complaints(submitted_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_urgency
                ON complaints(nlp_urgency_level)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_language
                ON complaints(language)
            """)

            # NEW index — supports is_complaint filtering
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_is_complaint
                ON complaints(is_complaint)
            """)

        logger.info(f"DB ready: {self.db_path}")

    # ─────────────────────────────────────────────────────────────
    # Insert
    # ─────────────────────────────────────────────────────────────
    def insert(self, complaint: dict) -> str:
        """
        Insert one analyzed complaint.

        Accepts the dict returned by
        MultilingualNLPPipeline.analyze().

        Returns:
            complaint_id
        """

        cid = complaint.get("complaint_id") or self._generate_id()

        kw = (
            complaint.get("keywords")
            or complaint.get("nlp_keywords")
            or []
        )

        if isinstance(kw, list):
            kw = json.dumps(kw, ensure_ascii=False)

        # is_complaint:
        # accept True/False, 1/0, or None
        raw_is_complaint = complaint.get("is_complaint")

        if raw_is_complaint is None:
            is_complaint_val = None
        else:
            is_complaint_val = 1 if raw_is_complaint else 0

        row = {
            "complaint_id": cid,

            "submitted_at": complaint.get(
                "submitted_at",
                datetime.now().isoformat()
            ),

            "msisdn": complaint.get("msisdn"),
            "city_input": complaint.get("city_input"),
            "segment": complaint.get("segment"),

            "channel": complaint.get("channel", "web"),

            "text_original": (
                complaint.get("text_original")
                or complaint.get("text", "")
            ),

            "language": complaint.get("language", "fr"),

            "nlp_category": (
                complaint.get("category")
                or complaint.get("nlp_category")
            ),

            "nlp_sentiment": (
                complaint.get("sentiment")
                or complaint.get("nlp_sentiment")
            ),

            "nlp_urgency_score": (
                complaint.get("urgency_score")
                or complaint.get("nlp_urgency_score", 0.0)
            ),

            "nlp_urgency_level": (
                complaint.get("urgency_level")
                or complaint.get("nlp_urgency_level", "normal")
            ),

            "nlp_city": (
                complaint.get("city")
                or complaint.get("nlp_city")
            ),

            "nlp_network_type": (
                complaint.get("network_type")
                or complaint.get("nlp_network_type")
            ),

            "nlp_keywords": kw,

            "status": complaint.get("status", "open"),

            "is_complaint": is_complaint_val,
        }

        with self._conn() as conn:

            conn.execute("""
                INSERT OR IGNORE INTO complaints (
                    complaint_id,
                    submitted_at,
                    msisdn,
                    city_input,
                    segment,
                    channel,
                    text_original,
                    language,
                    nlp_category,
                    nlp_sentiment,
                    nlp_urgency_score,
                    nlp_urgency_level,
                    nlp_city,
                    nlp_network_type,
                    nlp_keywords,
                    status,
                    is_complaint
                )
                VALUES (
                    :complaint_id,
                    :submitted_at,
                    :msisdn,
                    :city_input,
                    :segment,
                    :channel,
                    :text_original,
                    :language,
                    :nlp_category,
                    :nlp_sentiment,
                    :nlp_urgency_score,
                    :nlp_urgency_level,
                    :nlp_city,
                    :nlp_network_type,
                    :nlp_keywords,
                    :status,
                    :is_complaint
                )
            """, row)

        return cid

    # ─────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────
    def to_dataframe(
        self,
        language: str | None = None,
        urgency: str | None = None,
        sentiment: str | None = None,
        status: str | None = None,
        is_complaint: bool | None = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """
        Load complaints from DB with optional filters.
        """

        conditions: list[str] = []
        params: list = []

        if language is not None:
            conditions.append("language = ?")
            params.append(language)

        if urgency is not None:
            conditions.append("nlp_urgency_level = ?")
            params.append(urgency)

        if sentiment is not None:
            conditions.append("nlp_sentiment = ?")
            params.append(sentiment)

        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        # NEW: filter by is_complaint
        if is_complaint is not None:
            conditions.append("is_complaint = ?")
            params.append(1 if is_complaint else 0)

        where = (
            "WHERE " + " AND ".join(conditions)
            if conditions else ""
        )

        sql = f"""
            SELECT *
            FROM complaints
            {where}
            ORDER BY submitted_at DESC
            LIMIT ?
        """

        params.append(limit)

        with self._conn() as conn:
            df = pd.read_sql_query(sql, conn, params=params)

        if not df.empty:

            if "nlp_keywords" in df.columns:
                df["nlp_keywords"] = df["nlp_keywords"].apply(
                    lambda x:
                        json.loads(x)
                        if x and isinstance(x, str)
                        else []
                )

            # SQLite 0/1/NULL → Python bool/None
            if "is_complaint" in df.columns:
                df["is_complaint"] = df["is_complaint"].apply(
                    lambda x:
                        None
                        if pd.isna(x)
                        else bool(int(x))
                )

        return df

    # ─────────────────────────────────────────────────────────────
    # Aggregated stats
    # ─────────────────────────────────────────────────────────────
    def count(self) -> int:

        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM complaints"
            ).fetchone()[0]

    def stats(self) -> dict:
        """
        Return aggregated stats for the NLP dashboard.

        NEW fields:
            complaint_count
                — is_complaint = 1

            non_complaint_count
                — is_complaint = 0

            by_type
                — {"complaint": N, "feedback": N}
                  consumed by NLPAnalysis.jsx
        """

        with self._conn() as conn:

            total = conn.execute(
                "SELECT COUNT(*) FROM complaints"
            ).fetchone()[0]

            if total == 0:
                return {
                    "total": 0,
                    "complaint_count": 0,
                    "non_complaint_count": 0,
                    "by_type": {
                        "complaint": 0,
                        "feedback": 0,
                    },
                }

            by_lang = dict(conn.execute("""
                SELECT language, COUNT(*)
                FROM complaints
                GROUP BY language
            """).fetchall())

            by_cat = dict(conn.execute("""
                SELECT nlp_category, COUNT(*)
                FROM complaints
                GROUP BY nlp_category
                ORDER BY 2 DESC
                LIMIT 10
            """).fetchall())

            by_sent = dict(conn.execute("""
                SELECT nlp_sentiment, COUNT(*)
                FROM complaints
                GROUP BY nlp_sentiment
            """).fetchall())

            by_urg = dict(conn.execute("""
                SELECT nlp_urgency_level, COUNT(*)
                FROM complaints
                GROUP BY nlp_urgency_level
            """).fetchall())

            by_city = dict(conn.execute("""
                SELECT nlp_city, COUNT(*)
                FROM complaints
                WHERE nlp_city IS NOT NULL
                GROUP BY nlp_city
                ORDER BY 2 DESC
                LIMIT 10
            """).fetchall())

            avg_urg = (
                conn.execute("""
                    SELECT AVG(nlp_urgency_score)
                    FROM complaints
                """).fetchone()[0]
                or 0.0
            )

            # Complaint / feedback counts
            complaint_count = conn.execute("""
                SELECT COUNT(*)
                FROM complaints
                WHERE is_complaint = 1
            """).fetchone()[0]

            non_complaint_count = conn.execute("""
                SELECT COUNT(*)
                FROM complaints
                WHERE is_complaint = 0
            """).fetchone()[0]

        return {
            "total": total,

            "by_language": by_lang,
            "by_category": by_cat,
            "by_sentiment": by_sent,
            "by_urgency_level": by_urg,
            "by_city": by_city,

            "mean_urgency": round(avg_urg, 3),

            # NEW — consumed by NLPAnalysis.jsx
            "complaint_count": complaint_count,
            "non_complaint_count": non_complaint_count,

            "by_type": {
                "complaint": complaint_count,
                "feedback": non_complaint_count,
            },
        }

    # ─────────────────────────────────────────────────────────────
    # Mutations
    # ─────────────────────────────────────────────────────────────
    def update_status(
        self,
        complaint_id: str,
        status: str
    ) -> None:

        with self._conn() as conn:
            conn.execute(
                """
                UPDATE complaints
                SET status = ?
                WHERE complaint_id = ?
                """,
                (status, complaint_id),
            )

    # ─────────────────────────────────────────────────────────────
    # Generate unique complaint ID
    # ─────────────────────────────────────────────────────────────
    def _generate_id(self) -> str:
        """
        Generate a unique complaint ID.

        Example:
            OOR-A1B2C3D4
        """
        import uuid
        short = uuid.uuid4().hex[:8].upper()
        return f"OOR-{short}"