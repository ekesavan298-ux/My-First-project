import rag
import memory
import quiz
import study_plan
import tools

print("=== 1. Testing PDF Indexing ===")
res = rag.process_and_index_pdf("uploads/sample_lecture.pdf", "sample_lecture.pdf")
print("Index result:", res)

print("\n=== 2. Testing FAISS Search ===")
matches = rag.search_study_material("What is virtualization?", top_k=2)
print("Found matches:", len(matches))
for m in matches:
    print(f"Match score: {m['score']:.4f}, snippet: {m['text'][:60]}...")

print("\n=== 3. Testing Memory ===")
memory.add_question("What is virtualization?", topic="Cloud Computing")
summary = memory.get_progress_summary()
print("Topics studied:", summary["topics_studied"])
print("Recent questions:", len(summary["recent_questions"]))

print("\n=== 4. Testing Quiz Evaluation ===")
fallback_questions = quiz.generate_fallback_quiz("Cloud Computing", 5)
user_answers = {1: "A", 2: "B", 3: "A", 4: "B", 5: "A"}
eval_res = quiz.calculate_quiz_score(fallback_questions, user_answers, "Cloud Computing")
print(f"Quiz Score: {eval_res['score']}/{eval_res['total']} ({eval_res['percentage']}%)")
print("Weak areas identified:", eval_res["weak_areas"])

print("\n=== 5. Testing Tools & Intent Router ===")
r1 = tools.route_and_execute("Give me a quiz on Cloud Computing")
print("Router (Give me a quiz) ->", r1["tool_selected"])

r2 = tools.route_and_execute("How am I performing in my studies?")
print("Router (How am I performing) ->", r2["tool_selected"])

r3 = tools.route_and_execute("Create study plan for cloud computing")
print("Router (Create study plan) ->", r3["tool_selected"])

r4 = tools.route_and_execute("What does virtualization mean?")
print("Router (What does virtualization mean) ->", r4["tool_selected"])

print("\n=== 6. Testing Study Plan Generation ===")
plan = study_plan.generate_study_plan(
    subject="Cloud Computing",
    exam_date="2026-09-20",
    daily_hours=2.5,
    knowledge_level="Intermediate",
    weak_topics=["Cloud Service Models"]
)
print("Study plan days remaining:", plan["days_remaining"])
print("Plan preview (first 150 chars):")
print(plan["plan_markdown"][:150].encode('ascii', 'replace').decode('ascii'))

print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
