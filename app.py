"""
AI Learning & Study Assistant - Main Streamlit Application.
Provides an all-in-one student workspace:
1. 📚 Study Materials (PDF upload, text chunking, FAISS vector indexing)
2. 💬 Ask AI (RAG Q&A with source attribution & Tool Agent mode)
3. 📝 Quiz (5/10 MCQ generator, interactive test taking, instant grading & explanations)
4. 📊 My Progress (Student memory analytics, weak topics, quiz history)
5. 📅 Study Plan (Customized day-by-day exam schedules with export)
"""

import os
import time
from datetime import date, timedelta
from pathlib import Path
import streamlit as st

# Configure page
st.set_page_config(
    page_title="AI Learning & Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, student-friendly interface
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .source-box {
        background: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    .tool-badge {
        display: inline-block;
        background: #EEF2FF;
        color: #4F46E5;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        border: 1px solid #C7D2FE;
        margin-bottom: 0.4rem;
    }
    .badge-correct {
        color: #16A34A;
        font-weight: 700;
    }
    .badge-wrong {
        color: #DC2626;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Imports after styling to maintain clean module separation
import config
import rag
import quiz
import memory
import study_plan
import tools

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = None
if "quiz_topic" not in st.session_state:
    st.session_state.quiz_topic = ""
if "generated_plan" not in st.session_state:
    st.session_state.generated_plan = None

# ==========================================
# SIDEBAR SETUP & NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("## 🎓 AI Study Assistant")
    st.caption("Personalized learning with your own materials.")
    st.markdown("---")

    # Navigation Radio
    nav_choice = st.radio(
        "Navigation",
        [
            "📚 Study Materials",
            "💬 Ask AI",
            "📝 Quiz",
            "📊 My Progress",
            "📅 Study Plan"
        ],
        index=0
    )

    # Vector store status
    v_status = rag.get_vector_store_status()
    st.caption(f"📦 Index: {v_status['total_chunks']} chunks | {len(v_status['indexed_files'])} file(s)")

# Main Header
st.markdown('<div class="main-title">AI Learning & Study Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Learn smarter with your own study materials.</div>', unsafe_allow_html=True)


# ==========================================
# 1. 📚 STUDY MATERIALS SECTION
# ==========================================
if nav_choice == "📚 Study Materials":
    st.markdown("### 📚 Upload & Index Study Materials")
    st.write(
        "Upload your lecture notes, textbooks, or course slide PDFs. "
        "The system extracts text, generates embeddings with Sentence Transformers, "
        "and creates a local FAISS vector index for fast semantic search."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Select one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload college materials, lecture slides, or textbook chapters."
        )

        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) selected.")
            process_btn = st.button("🚀 Process & Index Materials", type="primary", use_container_width=True)

            if process_btn:
                for uploaded_file in uploaded_files:
                    file_path = config.UPLOADS_DIR / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    with st.spinner(f"Processing '{uploaded_file.name}'... (Extracting, Chunking & Embedding)"):
                        try:
                            result = rag.process_and_index_pdf(file_path, uploaded_file.name)
                            st.success(f"✅ Successfully processed **{result['file_name']}**")
                            
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("Pages Extracted", result["total_pages"])
                            with c2:
                                st.metric("Chunks Created", result["total_chunks"])
                            with c3:
                                st.metric("Total Vector Chunks", result["total_index_size"])

                        except ValueError as ve:
                            st.error(f"⚠️ {ve}")
                        except Exception as ex:
                            st.error(f"❌ Error processing '{uploaded_file.name}': {ex}")

    with col2:
        st.markdown("#### 📑 Current Vector Store Status")
        status = rag.get_vector_store_status()

        if status["is_ready"]:
            st.success(f"**Index Active:** {status['total_chunks']} searchable chunks")
            st.markdown("**Indexed Materials:**")
            for f in status["indexed_files"]:
                st.markdown(f"- 📄 `{f}`")

            st.markdown("---")
            if st.button("🗑️ Clear Vector Index", help="Deletes all indexed study chunks to start fresh"):
                rag.clear_vector_store()
                st.rerun()
        else:
            st.warning("No study materials indexed yet.\n\nPlease upload a PDF to enable RAG Question Answering.")


# ==========================================
# 2. 💬 ASK AI (RAG & AGENT)
# ==========================================
elif nav_choice == "💬 Ask AI":
    st.markdown("### 💬 Ask Questions on Your Materials")

    # Mode selector
    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        interaction_mode = st.radio(
            "Interaction Mode",
            ["Direct Study Q&A (RAG)", "AI Agent (Intent Router)"],
            horizontal=True,
            help="Direct RAG answers strictly from materials; Agent mode detects whether you want a quiz, study plan, or progress."
        )
    with col_m2:
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Quick sample prompt chips
    st.caption("💡 Quick prompts:")
    qp_cols = st.columns(3)
    quick_q = None
    with qp_cols[0]:
        if st.button("📌 Summarize core concepts", use_container_width=True):
            quick_q = "What is the main concept discussed in this material?"
    with qp_cols[1]:
        if st.button("📝 Give me a quiz", use_container_width=True):
            quick_q = "Give me a quiz"
    with qp_cols[2]:
        if st.button("📅 Plan my exam study", use_container_width=True):
            quick_q = "Create a study plan"

    # Display message history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg.get("tool"):
                st.markdown(f'<span class="tool-badge">Tool: {msg["tool"]}</span>', unsafe_allow_html=True)
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("🔍 View Sources & Citations", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**File:** `{src['file_name']}` | **Page:** {src['page']} | **Relevance:** {src['score']}\n\n"
                            f"> *\"{src['snippet']}\"*"
                        )

    # Input handling
    user_input = st.chat_input("Ask a question about your study materials...")
    prompt_to_run = quick_q if quick_q else user_input

    if prompt_to_run:
        # Check service availability
        if not config.is_gemini_configured():
            st.error("⚠️ AI assistant service is not configured. Please ensure GEMINI_API_KEY is set in the server environment.")
            st.stop()

        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_run})
        with st.chat_message("user"):
            st.markdown(prompt_to_run)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing study materials..."):
                try:
                    if interaction_mode == "AI Agent (Intent Router)":
                        res = tools.route_and_execute(prompt_to_run)
                        st.markdown(f'<span class="tool-badge">Tool: {res["tool_selected"]}</span>', unsafe_allow_html=True)
                        st.markdown(res["response"])
                        if res.get("sources"):
                            with st.expander("🔍 View Sources & Citations", expanded=False):
                                for src in res["sources"]:
                                    st.markdown(
                                        f"**File:** `{src['file_name']}` | **Page:** {src['page']} | **Score:** {src['score']}\n\n"
                                        f"> *\"{src['snippet']}\"*"
                                    )
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": res["response"],
                            "tool": res["tool_selected"],
                            "sources": res.get("sources", [])
                        })
                    else:
                        # Direct RAG
                        rag_out = rag.answer_question(prompt_to_run)
                        st.markdown(rag_out["answer"])
                        if rag_out.get("sources"):
                            with st.expander("🔍 View Sources & Citations", expanded=True):
                                for src in rag_out["sources"]:
                                    st.markdown(
                                        f"**File:** `{src['file_name']}` | **Page:** {src['page']} | **Similarity:** {src['score']}\n\n"
                                        f"> *\"{src['snippet']}\"*"
                                    )
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": rag_out["answer"],
                            "sources": rag_out.get("sources", [])
                        })
                except Exception as err:
                    st.error(f"❌ Error: {err}")


# ==========================================
# 3. 📝 QUIZ SECTION
# ==========================================
elif nav_choice == "📝 Quiz":
    st.markdown("### 📝 Interactive MCQ Quiz Generator")
    st.write("Generate practice tests with multiple choice questions, instant grading, and detailed explanations.")

    # Determine default topic from memory or indexed files
    mem_data = memory.load_memory()
    default_topic = mem_data["topics_studied"][-1] if mem_data["topics_studied"] else "Cloud Computing"

    with st.expander("⚙️ Quiz Setup Options", expanded=(st.session_state.current_quiz is None)):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            quiz_topic = st.text_input("Quiz Topic", value=default_topic, help="Concept or chapter to test on")
        with c2:
            num_q = st.selectbox("Number of Questions", [5, 10], index=0)
        with c3:
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)

        gen_quiz_btn = st.button("⚡ Generate Quiz", type="primary", use_container_width=True)

    if gen_quiz_btn:
        if not config.is_gemini_configured():
            st.error("⚠️ Quiz generation service is currently unavailable. Please verify the server environment configuration.")
        else:
            with st.spinner(f"Generating {num_q} {difficulty} questions on '{quiz_topic}'..."):
                try:
                    # Pull relevant context chunks from FAISS if available
                    context_chunks = rag.search_study_material(quiz_topic, top_k=3)
                    context_str = "\n\n".join([c["text"] for c in context_chunks]) if context_chunks else ""
                    
                    questions = quiz.generate_quiz_questions(
                        topic=quiz_topic,
                        num_questions=num_q,
                        difficulty=difficulty,
                        context=context_str
                    )
                    st.session_state.current_quiz = questions
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_results = None
                    st.session_state.quiz_topic = quiz_topic
                    st.success(f"Generated {len(questions)} questions on '{quiz_topic}'!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Failed to generate quiz: {ex}")

    # Render Interactive Quiz
    if st.session_state.current_quiz:
        st.markdown(f"#### ✍️ Quiz: {st.session_state.quiz_topic} ({len(st.session_state.current_quiz)} Questions)")

        # Render form for taking the quiz
        with st.form("quiz_form"):
            for q in st.session_state.current_quiz:
                st.markdown(f"**Q{q['id']}. {q['question']}**")
                options_list = [f"{k}. {v}" for k, v in q["options"].items()]

                # Current selection
                prev_choice = st.session_state.quiz_answers.get(q["id"], None)
                idx = 0
                if prev_choice:
                    for opt_idx, opt_str in enumerate(options_list):
                        if opt_str.startswith(prev_choice):
                            idx = opt_idx
                            break

                selected = st.radio(
                    f"Select answer for Q{q['id']}:",
                    options_list,
                    index=idx if prev_choice else None,
                    key=f"q_radio_{q['id']}",
                    label_visibility="collapsed"
                )
                if selected:
                    # Extract the option letter (A, B, C, D)
                    st.session_state.quiz_answers[q["id"]] = selected[0]

                st.markdown("---")

            submit_quiz = st.form_submit_button("📊 Submit Quiz & See Results", type="primary", use_container_width=True)

        if submit_quiz:
            with st.spinner("Grading your quiz and updating progress..."):
                results = quiz.calculate_quiz_score(
                    questions=st.session_state.current_quiz,
                    user_answers=st.session_state.quiz_answers,
                    topic=st.session_state.quiz_topic
                )
                st.session_state.quiz_results = results
                st.rerun()

    # Display Quiz Results
    if st.session_state.quiz_results:
        res = st.session_state.quiz_results
        st.markdown("### 🏆 Quiz Results")

        # Score cards
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Score", f"{res['score']} / {res['total']}")
        with sc2:
            st.metric("Percentage", f"{res['percentage']}%")
        with sc3:
            rating = "🌟 Excellent!" if res['percentage'] >= 80 else ("👍 Good Job!" if res['percentage'] >= 60 else "⚠️ Needs Review")
            st.metric("Performance", rating)

        # Progress bar
        st.progress(res['percentage'] / 100.0)

        # Question Breakdown
        st.markdown("#### 📋 Detailed Question Review")
        for item in res["details"]:
            with st.container():
                if item["is_correct"]:
                    st.success(
                        f"✅ **Q{item['id']}: Correct!**\n\n"
                        f"**Question:** {item['question']}\n\n"
                        f"**Your Answer:** {item['selected_option']}. {item['selected_text']}\n\n"
                        f"💡 *Explanation:* {item['explanation']}"
                    )
                else:
                    st.error(
                        f"❌ **Q{item['id']}: Incorrect**\n\n"
                        f"**Question:** {item['question']}\n\n"
                        f"**Your Answer:** {item['selected_option']}. {item['selected_text']}\n\n"
                        f"**Correct Answer:** {item['correct_option']}. {item['correct_text']}\n\n"
                        f"💡 *Explanation:* {item['explanation']}"
                    )

        if res["weak_areas"]:
            st.warning(f"⚠️ **Identified Focus Areas:** {', '.join(res['weak_areas'])}")

        st.info("✅ This quiz result has been recorded in your **📊 My Progress** memory.")


# ==========================================
# 4. 📊 MY PROGRESS (STUDENT MEMORY)
# ==========================================
elif nav_choice == "📊 My Progress":
    st.markdown("### 📊 Student Learning Memory & Progress")
    st.write("Tracks your active recall performance, weak topics, and questions over time.")

    summary = memory.get_progress_summary()

    # High-level metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Topics Studied", summary["total_topics_studied"])
    with m2:
        st.metric("Questions Asked", summary["total_questions_asked"])
    with m3:
        st.metric("Quizzes Completed", summary["total_quizzes_taken"])
    with m4:
        st.metric("Average Quiz Score", f"{summary['average_quiz_score']}%")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 📚 Topics Studied")
        if summary["topics_studied"]:
            for t in summary["topics_studied"]:
                st.markdown(f"- 📘 **{t}**")
        else:
            st.info("No topics recorded yet. Upload materials or ask questions to begin tracking.")

        st.markdown("#### ⚠️ Focus & Weak Topics")
        if summary["weak_topics"]:
            for wt in summary["weak_topics"]:
                st.markdown(f"- 🔴 **{wt}** *(needs practice)*")
        else:
            st.success("🎉 No weak topics flagged! Keep up the great work.")

    with col_right:
        st.markdown("#### 📝 Quiz History")
        if summary["quiz_history"]:
            for qh in summary["quiz_history"]:
                badge = "🟢" if qh["percentage"] >= 70 else "🔴"
                with st.expander(f"{badge} {qh['topic']} - {qh['score']}/{qh['total']} ({qh['percentage']}%)"):
                    st.write(f"📅 **Date:** {qh['timestamp']}")
                    st.write(f"🎯 **Score:** {qh['score']} of {qh['total']} ({qh['percentage']}%)")
                    if qh.get("weak_areas"):
                        st.write(f"⚠️ **Weak Areas:** {', '.join(qh['weak_areas'])}")
        else:
            st.info("No quizzes taken yet. Try the '📝 Quiz' tab!")

        st.markdown("#### 🕒 Recent Inquiries")
        if summary["recent_questions"]:
            for rq in summary["recent_questions"][:7]:
                st.markdown(f"- 💬 *\"{rq['question']}\"* ({rq.get('topic', 'General')})")
        else:
            st.info("No questions asked yet.")

    st.markdown("---")
    if st.button("🗑️ Reset All Student Memory", help="Wipes all recorded topics, quiz history, and weak areas"):
        memory.clear_memory()
        st.success("Student memory reset.")
        st.rerun()


# ==========================================
# 5. 📅 STUDY PLAN SECTION
# ==========================================
elif nav_choice == "📅 Study Plan":
    st.markdown("### 📅 Personalized Study Plan Generator")
    st.write("Generate an actionable, day-by-day study schedule based on your exam date, daily hours, and weak topics.")

    mem_data = memory.load_memory()
    default_sub = mem_data["topics_studied"][-1] if mem_data["topics_studied"] else "Cloud Computing"

    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        subject = st.text_input("Subject / Course Name", value=default_sub)
        exam_date = st.date_input("Target Exam Date", value=date.today() + timedelta(days=7), min_value=date.today())
        daily_hours = st.slider("Daily Study Hours Available", min_value=1.0, max_value=8.0, value=2.5, step=0.5)

    with col_p2:
        knowledge_level = st.selectbox("Current Knowledge Level", ["Beginner", "Intermediate", "Advanced"], index=1)
        
        # Pull weak topics from memory as suggestions
        suggested_weak = mem_data.get("weak_topics", [])
        selected_weak = st.multiselect(
            "Priority Weak Topics to Address",
            options=suggested_weak + ["Exam Review", "Core Theory"],
            default=suggested_weak if suggested_weak else []
        )
        custom_weak = st.text_input("Additional Focus Topics (optional)", placeholder="e.g., Virtualization, Service Models")
        if custom_weak.strip():
            selected_weak.append(custom_weak.strip())

    gen_plan_btn = st.button("📅 Generate Study Plan", type="primary", use_container_width=True)

    if gen_plan_btn:
        if not config.is_gemini_configured():
            st.error("⚠️ Study plan generator service is currently unavailable. Please verify the server environment configuration.")
        else:
            with st.spinner(f"Creating personalized study schedule for '{subject}'..."):
                try:
                    plan = study_plan.generate_study_plan(
                        subject=subject,
                        exam_date=str(exam_date),
                        daily_hours=daily_hours,
                        knowledge_level=knowledge_level,
                        weak_topics=selected_weak
                    )
                    st.session_state.generated_plan = plan
                    st.success("Study plan generated successfully!")
                except Exception as ex:
                    st.error(f"Error generating study plan: {ex}")

    if st.session_state.generated_plan:
        plan = st.session_state.generated_plan
        st.markdown("---")
        st.markdown(f"### 📋 Study Schedule: {plan['subject']}")

        c_d1, c_d2, c_d3 = st.columns(3)
        with c_d1:
            st.metric("Days Remaining", plan["days_remaining"])
        with c_d2:
            st.metric("Daily Commitment", f"{plan['daily_hours']} hours")
        with c_d3:
            st.metric("Knowledge Level", plan["knowledge_level"])

        st.markdown(plan["plan_markdown"])

        # Export / Download Plan
        st.download_button(
            label="📥 Download Study Plan (.md)",
            data=plan["plan_markdown"],
            file_name=f"Study_Plan_{plan['subject'].replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )
