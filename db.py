"""
db.py - SQLite Persistence Module

This module manages a local SQLite database (interview_coach.db) to record
every interview question, candidate answer, input mode (voice or text),
numerical score, and feedback from Gemini.

Features:
- Auto-creates the database table if it doesn't exist.
- Saves interview question attempts.
- Retrieves past session history into a Pandas DataFrame for dashboard rendering.
- Computes aggregate metrics (average score, total sessions, voice vs text count).
"""

import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd

DEFAULT_DB_PATH = "interview_coach.db"


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection with timeout and foreign key support."""
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initializes the SQLite database and creates the 'sessions' table if not already present.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                resume_name TEXT,
                jd_title TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                input_mode TEXT NOT NULL,
                score REAL NOT NULL,
                feedback TEXT NOT NULL
            )
        """)
        conn.commit()


def save_session(
    question: str,
    answer: str,
    input_mode: str,
    score: float,
    feedback: str,
    resume_name: Optional[str] = "Uploaded Resume",
    jd_title: Optional[str] = "Target Job Description",
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Saves a completed interview answer evaluation to the database.

    Args:
        question: The interview question asked.
        answer: The candidate's response (from voice transcription or text typing).
        input_mode: 'Voice' or 'Text'.
        score: Numerical score out of 10.
        feedback: Evaluation notes and constructive feedback.
        resume_name: Name of the uploaded resume document.
        jd_title: Job description title or label.
        db_path: SQLite database file path.

    Returns:
        The inserted row ID.
    """
    init_db(db_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (
                timestamp, resume_name, jd_title, question, answer, input_mode, score, feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_str,
            resume_name or "Resume",
            jd_title or "Job Description",
            question.strip(),
            answer.strip(),
            input_mode,
            float(score),
            feedback.strip()
        ))
        conn.commit()
        return cursor.lastrowid


def get_all_sessions(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Loads all recorded interview attempts from SQLite into a Pandas DataFrame.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        query = """
            SELECT 
                id,
                timestamp,
                resume_name AS "Resume",
                jd_title AS "Job Role",
                question AS "Question",
                answer AS "Answer",
                input_mode AS "Input Mode",
                score AS "Score / 10",
                feedback AS "Feedback"
            FROM sessions
            ORDER BY id DESC
        """
        df = pd.read_sql_query(query, conn)
    return df


def get_session_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Computes key performance metrics across all interview attempts.
    """
    df = get_all_sessions(db_path)
    if df.empty:
        return {
            "total_questions": 0,
            "average_score": 0.0,
            "voice_count": 0,
            "text_count": 0,
            "highest_score": 0.0,
        }

    scores = df["Score / 10"].dropna()
    total_q = len(df)
    avg_score = round(float(scores.mean()), 1) if not scores.empty else 0.0
    highest_score = round(float(scores.max()), 1) if not scores.empty else 0.0

    voice_count = int((df["Input Mode"].str.contains("Voice", case=False, na=False)).sum())
    text_count = total_q - voice_count

    return {
        "total_questions": total_q,
        "average_score": avg_score,
        "voice_count": voice_count,
        "text_count": text_count,
        "highest_score": highest_score,
    }
