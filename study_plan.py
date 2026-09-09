"""
Personalized Study Plan Generator Module.
Creates structured, day-by-day schedules tailored to exam dates, available study hours,
student knowledge level, and identified weak topics.
"""

from datetime import datetime, date
from typing import List, Dict, Any, Optional
from config import generate_gemini_text

def calculate_days_remaining(exam_date_str: str) -> int:
    """
    Calculates the number of days remaining until the target exam date.
    Returns at least 1 day.
    """
    try:
        exam_d = datetime.strptime(str(exam_date_str).strip()[:10], "%Y-%m-%d").date()
        today = date.today()
        diff = (exam_d - today).days
        return max(1, diff)
    except Exception:
        return 7  # default to a 7-day sprint if parsing fails

def generate_study_plan(
    subject: str,
    exam_date: str,
    daily_hours: float = 2.0,
    knowledge_level: str = "Intermediate",
    weak_topics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generates a day-by-day study schedule using Gemini with fallback capabilities.
    Returns:
    - subject: str
    - exam_date: str
    - days_remaining: int
    - daily_hours: float
    - plan_markdown: str
    """
    days_left = calculate_days_remaining(str(exam_date))
    # Cap plan days to a manageable demo sprint (e.g., up to 14 days detailed plan)
    plan_days = min(days_left, 14)
    weak_str = ", ".join(weak_topics) if weak_topics else "None specifically identified yet"

    system_prompt = (
        "You are an expert academic strategist and learning coach. "
        "Create realistic, actionable, high-retention day-by-day study plans. "
        "Use active recall, spaced repetition, and focused practice."
    )

    prompt = (
        f"Create a structured, day-by-day study plan for a student.\n\n"
        f"STUDENT DETAILS:\n"
        f"- Subject: {subject}\n"
        f"- Days Until Exam: {days_left} days (provide a detailed schedule for the upcoming {plan_days} days)\n"
        f"- Target Exam Date: {exam_date}\n"
        f"- Daily Study Time: {daily_hours} hours/day\n"
        f"- Current Knowledge Level: {knowledge_level}\n"
        f"- Identified Weak Topics / Priority Areas: {weak_str}\n\n"
        f"PLAN REQUIREMENTS:\n"
        f"1. Break down the time allocation each day into: Theory/Concepts, Practice/Problem Solving, and Quick Review/Quiz.\n"
        f"2. Give special priority to the weak topics in the first half of the schedule.\n"
        f"3. Dedicate the final days to comprehensive mock testing, formula review, and memory consolidation.\n"
        f"4. Format clearly with Markdown headers (e.g. ### Day 1: [Topic Title]), bullet points, and actionable tasks.\n"
        f"5. Conclude with 3 High-Impact Exam Tips."
    )

    try:
        plan_text = generate_gemini_text(prompt, system_instruction=system_prompt)
    except Exception as e:
        print(f"Gemini API error during study plan generation: {e}")
        plan_text = generate_fallback_study_plan(
            subject=subject,
            plan_days=plan_days,
            daily_hours=daily_hours,
            knowledge_level=knowledge_level,
            weak_topics=weak_topics
        )

    return {
        "subject": subject,
        "exam_date": str(exam_date),
        "days_remaining": days_left,
        "plan_days": plan_days,
        "daily_hours": daily_hours,
        "knowledge_level": knowledge_level,
        "weak_topics": weak_topics or [],
        "plan_markdown": plan_text
    }

def generate_fallback_study_plan(
    subject: str,
    plan_days: int,
    daily_hours: float,
    knowledge_level: str,
    weak_topics: Optional[List[str]] = None
) -> str:
    """
    Constructs a clean, structured offline study plan when the LLM is unreachable.
    """
    weak_str = ", ".join(weak_topics) if weak_topics else f"{subject} Core Concepts"
    half_time = round(daily_hours * 0.5, 1)
    practice_time = round(daily_hours * 0.35, 1)
    review_time = round(daily_hours * 0.15, 1)

    lines = [
        f"## 📅 {plan_days}-Day Accelerated Study Plan: {subject}",
        f"**Target Level:** {knowledge_level} | **Daily Commitment:** {daily_hours} hours | **Priority Areas:** {weak_str}\n",
        "---"
    ]

    for day in range(1, plan_days + 1):
        if day == 1:
            title = f"Diagnostic & Foundational Fundamentals ({subject})"
            focus = f"Review syllabus outline and tackle priority weak topic: {weak_str}"
        elif day == plan_days:
            title = "Final Review, Confidence Building & Mock Test"
            focus = "Comprehensive review of summary notes, flashcards, and a full-length timed mock quiz"
        elif day == plan_days - 1:
            title = "Weak Areas Consolidation & High-Yield Problems"
            focus = f"Targeted deep-dive into remaining stumbling blocks in {weak_str}"
        else:
            title = f"Core Topic Mastery - Module {day - 1}"
            focus = f"In-depth analysis of core {subject} chapters, diagrams, and case problems"

        lines.extend([
            f"### Day {day}: {title}",
            f"- 🎯 **Primary Focus:** {focus}",
            f"- 📖 **Theory & Notes ({half_time} hrs):** Read key definitions, derive core principles, and summarize formulas.",
            f"- ✍️ **Active Practice ({practice_time} hrs):** Solve 5-10 practice questions and take an interactive quiz.",
            f"- 🧠 **Spaced Recall ({review_time} hrs):** Write down top 5 takeaways from memory without looking at notes.\n"
        ])

    lines.extend([
        "---",
        "### 💡 3 High-Impact Exam Tips",
        "1. **Active Recall Beats Passive Reading:** Test yourself on concepts rather than repeatedly re-reading notes.",
        "2. **The Pomodoro Technique:** Work in 25-minute sprints with 5-minute pauses to maintain peak mental focus.",
        "3. **Sleep Consolidates Memory:** Avoid all-nighters before the exam to allow your brain to synthesize knowledge."
    ])

    return "\n".join(lines)
