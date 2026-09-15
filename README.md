# 🎙️ AI Mock Interview Coach

A complete, zero-cost, voice-enabled **Mock Interview Coach** powered by **Streamlit**, local **RAG** (Retrieval-Augmented Generation with `sentence-transformers` & `FAISS`), the **Google Gemini API (Free Tier)**, **gTTS + pygame** for Text-to-Speech, **SpeechRecognition** for voice answers, and **SQLite** for progress tracking.

---

## 📌 Features

1. **📄 Document Ingestion**:
   - Upload candidate resume (PDF format).
   - Provide target job description (either upload PDF or paste raw text).
   - Text extracted cleanly via `pdfplumber` and chunked with semantic overlap.

2. **🧠 Local RAG Analysis (Zero-Cost & Offline)**:
   - Encodes text chunks into dense 384-dimensional embeddings using `sentence-transformers/all-MiniLM-L6-v2` (runs 100% locally on CPU).
   - Builds an in-memory **FAISS** vector store (`IndexFlatIP` on L2-normalized vectors = cosine similarity).
   - Automatically computes:
     - **🟢 Matching Skills**: High-similarity areas where candidate experience satisfies JD requirements.
     - **🔴 Skill Gaps**: Low-similarity areas representing missing or unmentioned JD skills to probe in the interview.

3. **🤖 Batched Gemini Question Generation (Free-Tier Friendly)**:
   - Makes **one single batched call** to Google Gemini 1.5 Flash to generate 5 tailored interview questions:
     - 2 questions on matching skills & candidate projects
     - 2 questions on identified skill gaps & JD requirements
     - 1 role-specific behavioral/problem-solving question
   - Avoids multiple individual calls to respect free-tier rate limits.

4. **🔊 Voice Interaction**:
   - **TTS**: Reads each question aloud using Google Text-to-Speech (`gTTS`) and `pygame.mixer` through local speakers (plus in-browser `st.audio` replay).
   - **Dual-Mode Answering**: Candidates can choose between:
     - 🎤 **Voice Input**: Speaks their answer into the microphone, transcribed using `SpeechRecognition` via the free Google Web Speech backend with an editable confirmation box.
     - ⌨️ **Text Input**: Types directly into an expandable text area.

5. **📋 Intelligent Answer Evaluation**:
   - Gemini evaluates the response against the question context and role requirements.
   - Outputs a numerical score (out of 10) and constructive feedback on strong points and missing areas.

6. **📊 Local SQLite Persistence & Analytics Dashboard**:
   - Records session timestamp, question, answer, input mode (Voice/Text), score, and feedback into a local SQLite database (`interview_coach.db`).
   - "Past Sessions" dashboard displays score progression over time, overall averages, input breakdown, and searchable past attempts.

---

## 🏗️ Project Architecture

```
d:/resume_project/
├── .env.example          # Sample environment variables template
├── .gitignore            # Excludes .env, virtualenv, databases, and audio cache
├── requirements.txt      # Pinned dependency specifications
├── README.md             # Documentation, interview walkthrough, and deployment guide
├── parsing.py            # PDF text extraction and sliding-window chunking
├── rag.py                # sentence-transformers embeddings and FAISS similarity retrieval
├── tts_stt.py            # Text-to-speech (gTTS/pygame) and speech-to-text (SpeechRecognition)
├── db.py                 # SQLite database storage and pandas analytics helpers
└── app.py                # Streamlit UI, Gemini LLM orchestration, and progress dashboard
```

---

## 🔑 How to Get a Free Google Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click **"Create API key"** (no credit card or paid tier required).
4. Copy the API key generated for your account.

---

## 🚀 Local Setup & Installation

### 1. Prerequisites
- Python 3.10 to 3.13 installed on your system.
- Git installed.
- Working microphone and speakers for voice features.

### 2. Clone or Navigate to the Repository
```bash
cd d:/resume_project
```

### 3. Create and Activate a Virtual Environment
On Windows (PowerShell / Command Prompt):
```powershell
python -m venv .venv
.venv\Scripts\activate
```

On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Your Gemini API Key
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Open `.env` and add your key:
```ini
GEMINI_API_KEY=your_actual_api_key_here
```
*(Note: You can also enter the API key directly into the Streamlit sidebar input if you prefer not to use a `.env` file).*

### 6. Run the Application
```bash
streamlit run app.py
```
Streamlit will launch and open the app in your browser at `http://localhost:8501`.

---

## 🎓 College Placement / Interview Walkthrough

If an interviewer asks you to walk through the codebase line by line, here is how you can explain each module:

### 1. `parsing.py` (Document Processing)
- **Why `pdfplumber`?** Unlike basic text dumpers, `pdfplumber` preserves character flow and layout, cleanly extracting text page by page.
- **Why chunking with overlap?** Transformer models have fixed context windows and perform better when matching specific paragraphs. Overlap (e.g., 40–50 characters) ensures that sentences spanning chunk boundaries aren't split mid-thought.

### 2. `rag.py` (Local Vector RAG)
- **Embedding Model**: We use `all-MiniLM-L6-v2` from `sentence-transformers`. It produces 384-dimensional vectors, runs on CPU in milliseconds, and requires zero paid API tokens.
- **FAISS Indexing**: We normalize vectors using L2 norm (`faiss.normalize_L2`), allowing `IndexFlatIP` (inner product) to compute cosine similarity:
  $$\text{Cosine Similarity} = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$
- **Match vs. Gap Detection**:
  - We index the candidate's resume chunks in FAISS.
  - For each chunk of the job description, we retrieve the highest cosine similarity score against the resume.
  - Chunks with high similarity are marked as **Matching Skills** (strengths).
  - Chunks with low similarity are flagged as **Skill Gaps** (topics where the candidate has little or no stated experience).

### 3. `tts_stt.py` (Speech Engine)
- **TTS**: `gTTS` converts text to an MP3 file via Google's free TTS endpoint, which `pygame.mixer` loads and plays through the local speaker. We wrap device initialization in a `try/except` block so that running in headless cloud environments (e.g., Streamlit Cloud) does not crash the server.
- **STT**: `SpeechRecognition` records audio via `PyAudio` and submits the audio snippet to Google Web Speech API (`recognizer.recognize_google()`) for free transcription.

### 4. `db.py` (SQLite Persistence)
- Standard lightweight SQLite storage using Python's built-in `sqlite3` module.
- We explicitly close database connections using `try...finally` blocks to prevent Windows file-locking issues.
- `get_all_sessions()` reads the table directly into a Pandas DataFrame for analysis.

### 5. `app.py` (Streamlit Frontend & Gemini Orchestration)
- **Rate-Limit Optimization**: Rather than calling Gemini once per question (which risks hitting free-tier per-minute rate limits), we send retrieved RAG context in **one batched prompt** asking for 5 questions structured as JSON.
- **Dual Input**: A Streamlit radio toggle lets the user speak or type. Both paths output to a single `final_answer_text` variable, maintaining a unified downstream evaluation logic.

---

## 🌐 Deploying to Streamlit Community Cloud (Free)

Follow these exact steps to push this project to GitHub and deploy it for free:

### Step 1: Push to GitHub
1. Go to [GitHub](https://github.com) and click **"New repository"**.
2. Name it `ai-mock-interview-coach`, choose **Public**, and do **NOT** initialize with a README (we already have our local repo ready).
3. In your local terminal (`d:/resume_project`), run:
   ```bash
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/ai-mock-interview-coach.git
   git branch -M main
   git push -u origin main
   ```
   *(Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username).*

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `YOUR_GITHUB_USERNAME/ai-mock-interview-coach`.
4. Set Branch to `main` and Main file path to `app.py`.

### Step 3: Add Your Gemini API Key as a Secret (Crucial for Security!)
1. Before clicking Deploy, click **"Advanced settings..."** (or navigate to **App Settings > Secrets** after creating the app).
2. In the **Secrets** text box, enter your Gemini key in TOML format:
   ```toml
   GEMINI_API_KEY = "AIzaSyYourActualSecretKeyHere"
   ```
3. Click **Save** and then click **Deploy!**.

> [!NOTE]
> Streamlit Cloud reads this secret securely via `st.secrets["GEMINI_API_KEY"]`. Your `.env` file will **never** be exposed in your public GitHub repository because it is included in `.gitignore`.

---

## 📄 License
MIT License. Built for student portfolios and technical interview preparation.
