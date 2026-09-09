"""
Student Memory Module for AI Learning & Study Assistant.
Handles persistent JSON-based tracking of topics studied, questions asked,
quiz history, and weak topics.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from config import MEMORY_FILE

DEFAULT_MEMORY_STRUCTURE: Dict[str, Any] = {
    "topics_studied": [],
    "quiz_history": [],
    "weak_topics": [],
    "recent_questions": []
}

def load_memory() -> Dict[str, Any]:
    """
    Loads student memory from the JSON file.
    Creates and returns the default structure if the file is missing or corrupted.
    """
    if not MEMORY_FILE.exists():
        save_memory(DEFAULT_MEMORY_STRUCTURE)
        return DEFAULT_MEMORY_STRUCTURE.copy()

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure all required keys exist
            for key in DEFAULT_MEMORY_STRUCTURE:
                if key not in data:
                    data[key] = []
            return data
    except (json.JSONDecodeError, OSError):
        # Fallback to default if file is corrupted
        save_memory(DEFAULT_MEMORY_STRUCTURE)
        return DEFAULT_MEMORY_STRUCTURE.copy()

def save_memory(data: Dict[str, Any]) -> None:
    """
    Saves student memory to the JSON file safely.
    """
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"Error saving memory to {MEMORY_FILE}: {e}")

def add_topic_studied(topic: str) -> None:
    """
    Adds a new topic or document title to topics_studied if not already present.
    """
    if not topic or not topic.strip():
        return
    clean_topic = topic.strip()
    data = load_memory()
    if clean_topic not in data["topics_studied"]:
        data["topics_studied"].append(clean_topic)
        save_memory(data)

def add_question(question: str, topic: str = "General") -> None:
    """
    Records a question asked by the student, along with an optional topic tag and timestamp.
    Keeps the last 30 recent questions.
    """
    if not question or not question.strip():
        return
    data = load_memory()
    entry = {
        "question": question.strip(),
        "topic": topic.strip() if topic else "General",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    data["recent_questions"].insert(0, entry)
    # Trim to 30 most recent
    data["recent_questions"] = data["recent_questions"][:30]
    save_memory(data)

def add_quiz_result(topic: str, score: int, total: int, percentage: float, weak_areas: List[str] = None) -> None:
    """
    Records quiz completion:
    - Appends entry to quiz_history
    - If percentage < 70% or specific weak areas are flagged, adds to weak_topics
    - If score >= 85% and topic was in weak_topics, resolves it
    """
    data = load_memory()
    entry = {
        "topic": topic.strip() if topic else "General Study Material",
        "score": score,
        "total": total,
        "percentage": round(percentage, 1),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weak_areas": weak_areas or []
    }
    data["quiz_history"].insert(0, entry)

    # Track topic as studied
    if entry["topic"] not in data["topics_studied"]:
        data["topics_studied"].append(entry["topic"])

    # Update weak topics based on performance
    weak_set = set(data.get("weak_topics", []))
    if percentage < 70.0:
        weak_set.add(entry["topic"])
    elif percentage >= 85.0 and entry["topic"] in weak_set:
        weak_set.discard(entry["topic"])

    if weak_areas:
        for area in weak_areas:
            if area and area.strip():
                weak_set.add(area.strip())

    data["weak_topics"] = sorted(list(weak_set))
    save_memory(data)

def get_student_memory() -> Dict[str, Any]:
    """
    Returns the full student memory dictionary.
    """
    return load_memory()

def get_progress_summary() -> Dict[str, Any]:
    """
    Calculates high-level progress metrics for display in Streamlit cards.
    """
    data = load_memory()
    total_quizzes = len(data.get("quiz_history", []))
    avg_score = 0.0
    if total_quizzes > 0:
        avg_score = sum(q.get("percentage", 0.0) for q in data["quiz_history"]) / total_quizzes

    return {
        "total_topics_studied": len(data.get("topics_studied", [])),
        "total_questions_asked": len(data.get("recent_questions", [])),
        "total_quizzes_taken": total_quizzes,
        "average_quiz_score": round(avg_score, 1),
        "topics_studied": data.get("topics_studied", []),
        "weak_topics": data.get("weak_topics", []),
        "quiz_history": data.get("quiz_history", []),
        "recent_questions": data.get("recent_questions", [])
    }

def clear_memory() -> None:
    """
    Resets the memory file to clean default structure.
    """
    save_memory(DEFAULT_MEMORY_STRUCTURE)
