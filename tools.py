"""
AI Agent Tools & Intent Routing Module.
Implements simple, deterministic agent behavior:
1. Tool definitions (search_study_material, generate_quiz, calculate_quiz_score, create_study_plan, get_student_memory)
2. Keyword-based intent detection router
3. Execution and response formatting
"""

import re
from typing import Dict, Any, Optional, List
import rag
import quiz
import memory
import study_plan

# 1. Modular Tool Implementations

def search_study_material(query: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Tool: Searches uploaded study materials and generates a grounded RAG answer.
    """
    return rag.answer_question(query=query, top_k=top_k)

def generate_quiz(topic: str, num_questions: int = 5, difficulty: str = "Medium") -> List[Dict[str, Any]]:
    """
    Tool: Generates MCQs with 4 options, explanations, and correct answers.
    """
    # Attempt to pull relevant context from uploaded materials if available
    context_chunks = rag.search_study_material(topic, top_k=3)
    context_str = "\n\n".join([c.get("text", "") for c in context_chunks]) if context_chunks else ""
    return quiz.generate_quiz_questions(
        topic=topic,
        num_questions=num_questions,
        difficulty=difficulty,
        context=context_str
    )

def calculate_quiz_score(questions: List[Dict[str, Any]], user_answers: Dict[int, str], topic: str) -> Dict[str, Any]:
    """
    Tool: Evaluates student answers, calculates percentage, and records performance.
    """
    return quiz.calculate_quiz_score(questions, user_answers, topic)

def create_study_plan(
    subject: str,
    exam_date: str,
    daily_hours: float = 2.0,
    knowledge_level: str = "Intermediate",
    weak_topics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Tool: Generates a day-by-day exam preparation schedule.
    """
    return study_plan.generate_study_plan(
        subject=subject,
        exam_date=exam_date,
        daily_hours=daily_hours,
        knowledge_level=knowledge_level,
        weak_topics=weak_topics
    )

def get_student_memory() -> Dict[str, Any]:
    """
    Tool: Retrieves the current student persistent memory summary.
    """
    return memory.get_progress_summary()


# 2. Agent Decision Flow & Intent Router

def analyze_intent(user_prompt: str) -> str:
    """
    Analyzes student prompt using direct keyword heuristics (no expensive LLM classifier).
    Returns tool name:
    - 'generate_quiz'
    - 'get_student_memory'
    - 'create_study_plan'
    - 'search_study_material' (default)
    """
    text = user_prompt.lower().strip()

    # Quiz intent triggers
    quiz_keywords = [
        "give me a quiz", "generate quiz", "create a quiz", "quiz me", "start a quiz",
        "make a quiz", "take a quiz", "test me", "practice mcq", "mcq quiz", "test questions"
    ]
    if any(kw in text for kw in quiz_keywords) or (re.search(r"\bquiz\b", text) and ("make" in text or "give" in text or "create" in text)):
        return "generate_quiz"

    # Memory & progress intent triggers
    progress_keywords = [
        "how am i performing", "show my progress", "my progress", "view memory",
        "student memory", "my score", "my performance", "weak topics", "topics studied",
        "what are my weak", "quiz history", "learning stats"
    ]
    if any(kw in text for kw in progress_keywords) or (re.search(r"\b(progress|performance|stats|weak topics)\b", text)):
        return "get_student_memory"

    # Study plan intent triggers
    study_plan_keywords = [
        "create a study plan", "make a study plan", "create study plan", "generate study plan",
        "study schedule", "revision schedule", "plan my study", "plan my exam",
        "prepare for exam", "exam timetable", "study routine"
    ]
    if any(kw in text for kw in study_plan_keywords) or (re.search(r"\b(study plan|study schedule|timetable)\b", text)):
        return "create_study_plan"

    # Default to study material Q&A
    return "search_study_material"

def route_and_execute(user_prompt: str, context_topic: str = "General Study") -> Dict[str, Any]:
    """
    Agent execution pipeline:
    1. Analyze user request
    2. Select appropriate tool
    3. Execute tool
    4. Format and return response
    """
    tool_name = analyze_intent(user_prompt)

    if tool_name == "generate_quiz":
        # Extract topic if mentioned, else use context_topic
        topic = context_topic
        clean = re.sub(r"(give me a quiz on|generate quiz for|quiz me on|make a quiz about|quiz)", "", user_prompt, flags=re.IGNORECASE).strip()
        if len(clean) > 2:
            topic = clean

        questions = generate_quiz(topic=topic, num_questions=5, difficulty="Medium")
        formatted = (
            f"🎯 **Tool Selected:** `generate_quiz`\n\n"
            f"I have generated a 5-question quiz for **{topic}**! "
            f"You can practice it directly in the **📝 Quiz** tab."
        )
        return {
            "tool_selected": "generate_quiz",
            "tool_input": {"topic": topic, "num_questions": 5},
            "tool_output": questions,
            "response": formatted,
            "sources": []
        }

    elif tool_name == "get_student_memory":
        mem = get_student_memory()
        studied = ", ".join(mem["topics_studied"]) if mem["topics_studied"] else "None recorded yet"
        weak = ", ".join(mem["weak_topics"]) if mem["weak_topics"] else "None (great job!)"
        
        formatted = (
            f"📊 **Tool Selected:** `get_student_memory`\n\n"
            f"### 📈 Your Learning Progress Snapshot\n"
            f"- **Topics Studied:** {studied}\n"
            f"- **Quizzes Completed:** {mem['total_quizzes_taken']} (Avg Score: {mem['average_quiz_score']}%)\n"
            f"- **Questions Asked:** {mem['total_questions_asked']}\n"
            f"- **Focus Areas / Weak Topics:** {weak}\n\n"
            f"*Check the **📊 My Progress** tab for full analytics!*"
        )
        return {
            "tool_selected": "get_student_memory",
            "tool_input": {},
            "tool_output": mem,
            "response": formatted,
            "sources": []
        }

    elif tool_name == "create_study_plan":
        mem = memory.load_memory()
        subject = context_topic if context_topic and context_topic != "General Study" else "Core Subject"
        plan = create_study_plan(
            subject=subject,
            exam_date="2026-09-20",
            daily_hours=2.0,
            knowledge_level="Intermediate",
            weak_topics=mem.get("weak_topics", [])
        )
        formatted = (
            f"📅 **Tool Selected:** `create_study_plan`\n\n"
            f"Here is a personalized study schedule for **{subject}** based on your performance:\n\n"
            f"{plan['plan_markdown']}\n\n"
            f"*You can configure custom exam dates and study hours in the **📅 Study Plan** tab.*"
        )
        return {
            "tool_selected": "create_study_plan",
            "tool_input": {"subject": subject},
            "tool_output": plan,
            "response": formatted,
            "sources": []
        }

    else:
        # Default: search_study_material
        rag_res = search_study_material(query=user_prompt)
        formatted = (
            f"🔍 **Tool Selected:** `search_study_material`\n\n"
            f"{rag_res.get('answer', '')}"
        )
        return {
            "tool_selected": "search_study_material",
            "tool_input": {"query": user_prompt},
            "tool_output": rag_res,
            "response": formatted,
            "sources": rag_res.get("sources", [])
        }
