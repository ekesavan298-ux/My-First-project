"""
Interactive Quiz Generator and Grading Module.
Generates 5 or 10 Multiple Choice Questions (MCQs) via Gemini based on study materials,
grades student responses, generates explanations, and records performance in memory.
"""

import json
import re
from typing import List, Dict, Any, Optional
from config import generate_gemini_text
import memory

def clean_json_string(raw_text: str) -> str:
    """
    Extracts JSON array or object substring from raw Gemini LLM response.
    Handles Markdown ```json blocks and extra conversational text.
    """
    if not raw_text:
        return ""
    # Remove markdown code block fences
    text = re.sub(r"^```(?:json)?", "", raw_text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
    text = text.strip()

    # Look for [ ... ] array
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        return match.group(0)

    # Look for single { ... } object
    match_obj = re.search(r"\{\s*\"questions\"\s*:\s*\[.*\]\s*\}", text, re.DOTALL)
    if match_obj:
        return match_obj.group(0)

    return text

def generate_quiz_questions(
    topic: str,
    num_questions: int = 5,
    difficulty: str = "Medium",
    context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generates structured MCQs using Gemini.
    Each item in the returned list contains:
    - id: int
    - question: str
    - options: {"A": "...", "B": "...", "C": "...", "D": "..."}
    - correct_answer: "A" | "B" | "C" | "D"
    - explanation: str
    - subtopic: str
    """
    num_questions = 10 if num_questions >= 10 else 5
    difficulty = difficulty.capitalize() if difficulty in ["Easy", "Medium", "Hard"] else "Medium"

    context_prompt = ""
    if context and context.strip():
        context_prompt = (
            f"Ground the quiz primarily on the following study context:\n"
            f"----------------------------------------\n"
            f"{context.strip()[:4000]}\n"
            f"----------------------------------------\n\n"
        )

    system_prompt = (
        "You are an academic test maker and tutor. "
        "Your task is to generate high-quality, unambiguous multiple-choice questions (MCQs). "
        "You must output ONLY valid JSON format with no additional text or Markdown wrapping."
    )

    prompt = (
        f"{context_prompt}"
        f"Generate exactly {num_questions} multiple choice questions on the topic: '{topic}'.\n"
        f"Difficulty level: {difficulty}.\n\n"
        f"CRITICAL FORMAT RULES:\n"
        f"Output must be a valid JSON array of objects. Each object must have these exact keys:\n"
        f"- \"question\": string with the question text\n"
        f"- \"options\": object with exactly 4 keys \"A\", \"B\", \"C\", \"D\" and their corresponding option strings\n"
        f"- \"correct_answer\": a single character string \"A\", \"B\", \"C\", or \"D\"\n"
        f"- \"explanation\": concise explanation of why the correct answer is right\n"
        f"- \"subtopic\": short concept or subtopic name (used for tracking weak areas)\n\n"
        f"Example JSON structure:\n"
        f"[\n"
        f"  {{\n"
        f"    \"question\": \"What is the primary benefit of cloud elasticity?\",\n"
        f"    \"options\": {{\n"
        f"      \"A\": \"Manual server purchasing\",\n"
        f"      \"B\": \"Dynamically adapting resources to demand\",\n"
        f"      \"C\": \"Eliminating all security concerns\",\n"
        f"      \"D\": \"Replacing internet connectivity\"\n"
        f"    }},\n"
        f"    \"correct_answer\": \"B\",\n"
        f"    \"explanation\": \"Elasticity allows cloud systems to scale resources up or down dynamically based on workload demand.\",\n"
        f"    \"subtopic\": \"Cloud Elasticity\"\n"
        f"  }}\n"
        f"]"
    )

    raw_response = ""
    try:
        raw_response = generate_gemini_text(prompt, system_instruction=system_prompt)
        clean_json = clean_json_string(raw_response)
        parsed = json.loads(clean_json)
        if isinstance(parsed, dict) and "questions" in parsed:
            questions_list = parsed["questions"]
        elif isinstance(parsed, list):
            questions_list = parsed
        else:
            raise ValueError("Parsed JSON is neither a list nor an object with a 'questions' key.")

        formatted_questions = []
        for i, q in enumerate(questions_list, start=1):
            options = q.get("options", {})
            # Ensure 4 options A, B, C, D
            if not isinstance(options, dict) or not all(k in options for k in ["A", "B", "C", "D"]):
                # If options came as a list
                if isinstance(options, list) and len(options) >= 4:
                    options = {
                        "A": str(options[0]),
                        "B": str(options[1]),
                        "C": str(options[2]),
                        "D": str(options[3])
                    }
                else:
                    continue  # skip invalid question

            correct = str(q.get("correct_answer", "A")).strip().upper()
            if correct not in ["A", "B", "C", "D"]:
                # Try matching option text
                matched = False
                for k, v in options.items():
                    if correct.lower() in v.lower():
                        correct = k
                        matched = True
                        break
                if not matched:
                    correct = "A"

            formatted_questions.append({
                "id": i,
                "question": q.get("question", f"Question {i}"),
                "options": options,
                "correct_answer": correct,
                "explanation": q.get("explanation", "No explanation provided."),
                "subtopic": q.get("subtopic", topic)
            })

        if not formatted_questions:
            raise ValueError("No valid questions could be formatted from Gemini response.")

        return formatted_questions

    except Exception as e:
        # Fallback graceful handler to avoid crashing
        print(f"Error parsing Gemini quiz JSON: {e}\nRaw text:\n{raw_response[:300]}")
        return generate_fallback_quiz(topic, num_questions)

def generate_fallback_quiz(topic: str, num_questions: int) -> List[Dict[str, Any]]:
    """
    Reliable built-in fallback questions in case API or parsing fails.
    """
    fallback_pool = [
        {
            "id": 1,
            "question": f"Which of the following best defines the core principle of {topic}?",
            "options": {
                "A": f"A foundational framework for understanding and applying {topic}",
                "B": "A deprecated legacy method rarely utilized today",
                "C": "An isolated hardware component only",
                "D": "A purely theoretical construct with no practical relevance"
            },
            "correct_answer": "A",
            "explanation": f"{topic} is built upon core principles that enable practical application and structured learning.",
            "subtopic": f"{topic} Fundamentals"
        },
        {
            "id": 2,
            "question": f"What is a primary advantage of adopting systematic approaches in {topic}?",
            "options": {
                "A": "Increased errors and inconsistency",
                "B": "Predictable outcomes, scalability, and enhanced efficiency",
                "C": "Uncontrolled resource consumption",
                "D": "Slower development cycles without measurable benefits"
            },
            "correct_answer": "B",
            "explanation": f"Systematic methodology in {topic} ensures reproducibility, high quality, and optimal resource usage.",
            "subtopic": f"{topic} Best Practices"
        },
        {
            "id": 3,
            "question": f"When evaluating key metrics in {topic}, what is most essential?",
            "options": {
                "A": "Ignoring baseline standards",
                "B": "Measuring verifiable accuracy, performance, and reliability",
                "C": "Focusing solely on subjective intuition",
                "D": "Disregarding student feedback"
            },
            "correct_answer": "B",
            "explanation": "Reliable quantitative and qualitative evaluation ensures continuous improvement.",
            "subtopic": f"{topic} Evaluation"
        },
        {
            "id": 4,
            "question": f"Which strategy is recommended for mastering challenging concepts in {topic}?",
            "options": {
                "A": "Passive reading without practice",
                "B": "Active recall, practical exercises, and spaced repetition",
                "C": "Cramming the night before an exam",
                "D": "Skipping prerequisite fundamentals"
            },
            "correct_answer": "B",
            "explanation": "Active learning and spaced repetition are proven study techniques for long-term retention.",
            "subtopic": f"{topic} Study Strategy"
        },
        {
            "id": 5,
            "question": f"How do modern tools transform the practice of {topic}?",
            "options": {
                "A": "By automating routine tasks and providing intelligent insights",
                "B": "By eliminating the need for foundational understanding",
                "C": "By increasing manual data entry",
                "D": "By restricting access to information"
            },
            "correct_answer": "A",
            "explanation": "Modern AI and software tools empower students and practitioners by accelerating comprehension.",
            "subtopic": f"{topic} Modern Tools"
        }
    ]

    if num_questions > 5:
        # Extend with 5 more variations
        for i in range(6, 11):
            fallback_pool.append({
                "id": i,
                "question": f"In {topic}, what consideration is vital during implementation phase {i-5}?",
                "options": {
                    "A": "Rigorous validation and continuous monitoring",
                    "B": "Omitting documentation",
                    "C": "Assuming zero edge cases",
                    "D": "Bypassing safety protocols"
                },
                "correct_answer": "A",
                "explanation": "Validation and continuous monitoring maintain system integrity.",
                "subtopic": f"{topic} Implementation"
            })

    return fallback_pool[:num_questions]

def calculate_quiz_score(
    questions: List[Dict[str, Any]],
    user_answers: Dict[int, str],
    topic: str
) -> Dict[str, Any]:
    """
    Grades user answers against questions.
    Returns:
    - score: int
    - total: int
    - percentage: float
    - details: List[Dict] with question, selected, correct, is_correct, explanation
    - weak_areas: List[str]
    - saved_to_memory: bool
    """
    total = len(questions)
    score = 0
    details = []
    weak_areas = []

    for q in questions:
        q_id = q["id"]
        correct_opt = q["correct_answer"].strip().upper()
        user_opt = user_answers.get(q_id, "").strip().upper()
        is_correct = (user_opt == correct_opt)

        if is_correct:
            score += 1
        else:
            sub = q.get("subtopic", topic)
            if sub and sub not in weak_areas:
                weak_areas.append(sub)

        details.append({
            "id": q_id,
            "question": q["question"],
            "options": q["options"],
            "selected_option": user_opt,
            "correct_option": correct_opt,
            "selected_text": q["options"].get(user_opt, "No answer selected"),
            "correct_text": q["options"].get(correct_opt, ""),
            "is_correct": is_correct,
            "explanation": q.get("explanation", ""),
            "subtopic": q.get("subtopic", topic)
        })

    percentage = round((score / total * 100.0), 1) if total > 0 else 0.0

    # Save to persistent memory
    memory.add_quiz_result(
        topic=topic,
        score=score,
        total=total,
        percentage=percentage,
        weak_areas=weak_areas
    )

    return {
        "score": score,
        "total": total,
        "percentage": percentage,
        "details": details,
        "weak_areas": weak_areas,
        "saved_to_memory": True
    }
