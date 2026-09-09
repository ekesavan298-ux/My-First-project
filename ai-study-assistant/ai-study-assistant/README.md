# 🎓 AI Learning & Study Assistant

A lightweight, reliable, and intelligent AI study companion built specifically for students. Upload your course materials, lecture notes, or textbooks in PDF format, ask questions grounded strictly in your documents using Retrieval-Augmented Generation (RAG), generate personalized multiple-choice quizzes with instant grading, track learning progress and weak topics over time, and build day-by-day exam study schedules.

---

## 🚀 Key Features

- **📚 PDF Ingestion & Semantic Vector Store:**
  - Extracts text page-by-page from multiple PDFs using `PyPDF`.
  - Cleans, normalizes, and chunks text into structured segments with page and file citations.
  - Generates dense semantic embeddings with `Sentence Transformers` (`all-MiniLM-L6-v2`).
  - Indexes chunks in a fast, CPU-optimized `FAISS` vector database using cosine similarity (`IndexFlatIP` on normalized vectors).
  - Gracefully detects scanned or empty PDFs with user-friendly alerts.

- **💬 Grounded RAG Question Answering:**
  - Semantic similarity search against the FAISS index to find the most relevant document chunks.
  - Formulates grounded prompts for Google Gemini with strict anti-hallucination guardrails.
  - If information is absent from the study material, cleanly states: *"I couldn't find this information in the uploaded study material."*
  - Expandable **Sources & Citations** box beneath every answer displaying the source file, page number, relevance score, and text snippet.

- **📝 Interactive MCQ Quiz Generator:**
  - Generates 5 or 10 multiple-choice questions (MCQs) tailored to the uploaded material or chosen topic.
  - Configurable difficulty (Easy, Medium, Hard).
  - Interactive radio-button test form in Streamlit.
  - Instant scoring, percentage calculation, detailed answer explanations, and weak-area diagnosis.
  - Automatically updates student memory with quiz scores.

- **📊 Student Memory (Persistent Progress):**
  - Stored locally in `data/memory.json`.
  - Logs topics studied, questions asked, quiz history, and diagnosed weak topics.
  - Identifies weak areas when quiz scores fall below 70% and marks mastery when scores exceed 85%.
  - High-level progress dashboard with metrics cards.

- **📅 Personalized Study Plan Generator:**
  - Inputs: Course subject, target exam date, daily study hours, knowledge level, and priority weak topics.
  - Produces an actionable day-by-day study schedule with theory allocations, practice exercises, spaced recall, and exam tips.
  - Instant Markdown export/download button.

- **🤖 AI Agent & Intent Router:**
  - Simple, deterministic keyword router executing dedicated tools without slow or expensive LLM classifiers:
    - `search_study_material(query)`
    - `generate_quiz(topic, num_questions, difficulty)`
    - `calculate_quiz_score(questions, user_answers, topic)`
    - `create_study_plan(subject, exam_date, daily_hours, knowledge_level, weak_topics)`
    - `get_student_memory()`

---

## 🏛️ System Architecture

```
                               +-----------------------------+
                               |     User / Streamlit UI     |
                               +--------------+--------------+
                                              |
                     +------------------------+------------------------+
                     |                        |                        |
             [Study Materials]            [Ask AI]               [Quiz / Plan]
                     |                        |                        |
             PyPDF Extraction                 |                        |
                     |                 Intent Router                   |
              Text Chunking                   |                        |
                     |            +-----------+-----------+            |
           SentenceTransformers   |                       |            |
            (all-MiniLM-L6-v2)    v                       v            v
                     |       Vector Search         Direct Tools     Gemini API
                     v       (FAISS Index)       (quiz, memory,    (Flash Model)
               FAISS Index        |               study plan)          |
                     |            v                       |            |
                     +---> Retrieved Context              +------------+
                                  |
                                  v
                          Google Gemini LLM
                                  |
                                  v
                        Grounded Answer + Sources
                                  |
                                  v
                      Local Memory (data/memory.json)
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.11+ (Tested on Python 3.12)
- **Frontend / UI:** Streamlit
- **LLM:** Google Gemini API (`google-generativeai`)
- **PDF Extraction:** PyPDF
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2` - CPU friendly, fast)
- **Vector Search:** FAISS (`faiss-cpu`)
- **Environment Management:** python-dotenv
- **Persistence:** Local JSON (`data/memory.json`) & Binary FAISS (`data/vector_store/`)

> **Note:** Zero reliance on LangChain or LangGraph. The RAG pipeline and agent router are written directly in modular Python for simplicity, full visibility, and quick debugging.

---

## 📁 Project Structure

```
ai-study-assistant/
│
├── app.py               # Streamlit application UI & navigation
├── rag.py               # PDF extraction, chunking, FAISS index & RAG QA
├── quiz.py              # MCQ generation, interactive grading & explanations
├── memory.py            # Persistent student memory (data/memory.json)
├── study_plan.py        # Day-by-day exam schedule generator & export
├── tools.py             # Modular tools & keyword intent router
├── config.py            # Environment variables, paths, and Gemini client
├── requirements.txt     # Minimal, pinned Python dependencies
├── .env.example         # Example environment file
├── README.md            # Complete documentation & demo guide
│
├── data/
│   ├── memory.json      # Persistent student history and metrics
│   └── vector_store/    # Saved FAISS index & chunk metadata
│
└── uploads/             # Locally stored uploaded PDFs
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.11 or Python 3.12 installed on your system.
- A free Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/).

### 2. Clone or Navigate to the Directory
```bash
cd d:\inten\ai-study-assistant
```

### 3. Set Up a Virtual Environment
**Windows (PowerShell or CMD):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Your Gemini API Key
Create a `.env` file in the root directory:
```bash
copy .env.example .env
```
Open `.env` and add your API key:
```env
GEMINI_API_KEY=AIzaSy...your_real_api_key_here
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

*(Configured securely via server-side environment variables or .env file)*

---

## 🚀 Running the Application

Ensure your virtual environment is active, then launch Streamlit:

```bash
streamlit run app.py
```

The application will open automatically in your browser at:
`http://localhost:8501`

---

## 🔍 How Each Component Works

### 1. How RAG Works
1. **Extraction:** When a PDF is uploaded, `pypdf` reads each page. Empty pages or scanned images without selectable text are flagged gracefully.
2. **Chunking:** Pages are split into 700-character chunks with a 100-character sliding overlap. Each chunk retains its metadata: source filename, page number, and chunk ID.
3. **Embedding:** Chunks are passed through `all-MiniLM-L6-v2` to produce 384-dimensional dense vectors. Vectors are normalized to unit length.
4. **Indexing:** Normalized vectors are inserted into a `faiss.IndexFlatIP`. (Inner product of unit vectors equals exact cosine similarity).
5. **Retrieval:** When a question is asked, the query is embedded, normalized, and searched in FAISS for the top-4 most similar chunks.
6. **Generation:** Retrieved passages are formatted into a grounded system prompt for Gemini. If similarity scores fall below threshold or the information is not present, the model responds with: *"I couldn't find this information in the uploaded study material."*
7. **Citations:** Source details (file, page number, relevance score, text snippet) are rendered in an interactive expander.

### 2. How Student Memory Works
- Persistent storage is maintained in `data/memory.json`.
- When a user asks a question, it is logged with a timestamp into `recent_questions`.
- When a PDF is ingested, the title is added to `topics_studied`.
- When a quiz is completed:
  - The score, total, percentage, timestamp, and topic are appended to `quiz_history`.
  - If the score is `< 70%`, the topic and flagged concepts are added to `weak_topics`.
  - If the score reaches `>= 85%`, previously flagged weak topics are resolved.
- The **📊 My Progress** section computes summary statistics (average score, total questions asked, quizzes taken) in real time.

### 3. How Tools & Agent Flow Works
Instead of unpredictable multi-step LLM chains, `tools.py` implements a fast, deterministic keyword router:
- Queries containing "quiz", "test me", "give me a quiz" ➔ `generate_quiz()`
- Queries containing "progress", "score", "performance", "how am i performing" ➔ `get_student_memory()`
- Queries containing "study plan", "schedule", "timetable" ➔ `create_study_plan()`
- General questions ➔ `search_study_material()` (RAG)

---

## 🎬 Step-by-Step Demo Flow

Follow these exact steps during your demonstration:

1. **Launch App:**
   Run `streamlit run app.py`. Point out the clean sidebar and the 5 modular sections.
2. **Upload Study Material:**
   Go to **📚 Study Materials**. Upload a course PDF (e.g., `Cloud_Computing_Lecture.pdf`). Click **🚀 Process & Index Materials**. Show the metrics: Pages Extracted, Chunks Created, and Total Vector Chunks.
3. **Ask a RAG Question:**
   Go to **💬 Ask AI**. Type:
   `"What is the main concept discussed in this material?"`
   Point out the AI's grounded explanation and expand the **🔍 View Sources & Citations** box to show the exact page number and text snippet.
4. **Test Anti-Hallucination Guardrail:**
   Ask an unrelated question (e.g., *"Who won the 1994 football world cup?"*).
   Observe the strict grounded response: *"I couldn't find this information in the uploaded study material."*
5. **Generate a Quiz:**
   Go to **📝 Quiz**. Select your topic, choose 5 questions and Medium difficulty, then click **⚡ Generate Quiz**.
6. **Take & Submit Quiz:**
   Answer the MCQs interactively. Click **📊 Submit Quiz & See Results**.
   Highlight the score card, percentage bar, correct vs incorrect cards, and detailed explanations.
7. **Inspect Student Memory:**
   Go to **📊 My Progress**. Show that the quiz score, topics studied, recent inquiries, and any weak topics were automatically recorded in memory.
8. **Generate Personalized Study Plan:**
   Go to **📅 Study Plan**. Select your exam date (e.g. 7 days from today), set daily hours (e.g. 2.5 hrs), pick your knowledge level, and click **📅 Generate Study Plan**. Show the day-by-day plan and demonstrate clicking **📥 Download Study Plan (.md)**.

---

## ❓ Troubleshooting

| Issue | Likely Cause | Solution |
|---|---|---|
| `GEMINI_API_KEY is not set` | Missing API key in environment | Set `GEMINI_API_KEY` in `.env` or server environment variables. |
| `Gemini model not found (404)` | Configured model not supported in region | Change `GEMINI_MODEL=gemini-2.5-flash` (or `gemini-flash-latest`, `gemini-3.5-flash`) in `.env`. |
| `PDF contains no extractable text` | PDF is scanned images or empty | Upload a digital PDF with selectable text. |
| `No study materials uploaded yet` | Asking questions with empty FAISS index | Ingest at least one PDF in **📚 Study Materials** first. |
| Memory corrupted or invalid | Broken JSON syntax | Click **🗑️ Reset All Student Memory** in the progress tab to auto-regenerate. |
