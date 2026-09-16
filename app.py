"""
app.py - AI Mock Interview Coach (Streamlit Application)

This is the main entry point of the project. It orchestrates:
1. Document upload (Resume PDF + JD PDF/text) and text extraction (parsing.py).
2. Local basic RAG analysis (rag.py) with sentence-transformers & FAISS to identify matching skills and gap areas.
3. Batched question generation via Google Gemini API free tier (1 batched call for 5 questions to respect rate limits).
4. Interactive question delivery with Text-to-Speech (gTTS + pygame) and dual-mode answering:
   - Voice mode (SpeechRecognition via Google Web Speech backend)
   - Text mode (Streamlit text_area)
5. Gemini-based answer evaluation (score out of 10 + constructive feedback).
6. Local SQLite session logging and analytics dashboard (db.py).
"""

import os
import time
import uuid
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

import importlib
# Import modular backend components
import parsing
import rag
import tts_stt
import db

# Force fresh reload of custom modules on every Streamlit script execution
importlib.reload(parsing)
importlib.reload(rag)
importlib.reload(tts_stt)
importlib.reload(db)

# -----------------------------------------------------------------------------
# Configuration & Environment Setup
# -----------------------------------------------------------------------------
# Load environment variables from .env file (if present)
load_dotenv()

st.set_page_config(
    page_title="AI Mock Interview Coach",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize the local SQLite database table on startup
db.init_db()


def get_gemini_api_key() -> Optional[str]:
    """
    Safely retrieves the Gemini API key from environment, Streamlit Secrets,
    or sidebar input. Never prints or logs the key.
    """
    # 1. Check local environment variable (.env)
    api_key = os.getenv("GEMINI_API_KEY")

    # 2. Check Streamlit Cloud Secrets (for deployment)
    if not api_key:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    return api_key


# -----------------------------------------------------------------------------
# Gemini LLM Helper Functions (Free Tier)
# -----------------------------------------------------------------------------
# Ordered list of candidate models for resilience against model lifecycle deprecations.
# (Google has retired 1.5/2.0/2.5-flash; active models include 3.6-flash, flash-latest, 3.8-flash).
ACTIVE_GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
]


def generate_with_gemini(api_key: str, prompt: str) -> str:
    """
    Executes a content generation request using the current active Gemini Flash model.
    Tries candidate models in order to guarantee seamless operation across API updates.
    """
    genai.configure(api_key=api_key)
    last_error = None

    for model_name in ACTIVE_GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
            continue

    if last_error:
        raise last_error
    raise RuntimeError("Could not obtain a response from Gemini API.")


def get_configured_gemini_model(api_key: str):
    """
    Configures and returns an active Gemini GenerativeModel instance.
    """
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-3.6-flash")


def create_fallback_skills_and_gaps(
    raw_matches: List[Dict[str, Any]],
    raw_gaps: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Transforms raw RAG chunks into clean, non-technical human-readable cards
    if LLM synthesis format is unavailable.
    """
    clean_matches = []
    for idx, m in enumerate(raw_matches[:3]):
        jd_text = m.get("jd_chunk", "").strip()
        sim = float(m.get("similarity_score", 0.5))
        # Convert cosine similarity (typically 0.35-0.70) into intuitive 70-95%
        pct = int(min(max((sim - 0.2) / 0.5 * 35 + 60, 68), 96))
        words = jd_text.split()
        short_title = " ".join(words[:5]).rstrip(".,;:") if words else f"Core Competency {idx + 1}"
        clean_matches.append({
            "skill": short_title,
            "match_score": pct,
            "job_requirement": jd_text,
            "candidate_evidence": m.get("best_resume_match", "Your resume experience aligns with this requirement.")
        })

    clean_gaps = []
    for idx, g in enumerate(raw_gaps[:3]):
        jd_text = g.get("jd_chunk", "").strip()
        sim = float(g.get("similarity_score", 0.3))
        # Convert cosine similarity into intuitive 25-50%
        pct = int(min(max((sim - 0.1) / 0.4 * 30 + 20, 22), 52))
        words = jd_text.split()
        short_title = " ".join(words[:5]).rstrip(".,;:") if words else f"Growth Area {idx + 1}"
        clean_gaps.append({
            "skill": short_title,
            "match_score": pct,
            "job_requirement": jd_text,
            "advice": "This requirement is less visible on your resume. Expect questions testing your ability to learn and adapt."
        })

    return clean_matches, clean_gaps


def generate_interview_questions(
    api_key: str,
    resume_summary: str,
    jd_summary: str,
    matching_skills: List[Dict[str, Any]],
    skill_gaps: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generates non-technical human-readable skill match synthesis AND 5 tailored
    interview questions in ONE single batched Gemini call to preserve free-tier limits.
    """
    # Format retrieved RAG context for prompt injection
    matches_text = "\n".join([
        f"- JD Need: {m.get('jd_chunk', '')} (Match similarity: {m.get('similarity_score', 0.5)})"
        for m in matching_skills
    ]) if matching_skills else "General software and domain competencies."

    gaps_text = "\n".join([
        f"- JD Need: {g.get('jd_chunk', '')} (Match similarity: {g.get('similarity_score', 0.3)})"
        for g in skill_gaps
    ]) if skill_gaps else "Specific advanced domain requirements."

    prompt = f"""You are an expert career coach and hiring manager.
Compare the candidate's resume against the job description.
Identify:
1. Candidate's matching skills (strengths where resume clearly aligns with the job).
2. Candidate's skill gaps (job requirements where resume has the least visible experience).
Write ALL explanations in simple, everyday language so that anyone from a non-technical background can understand their strengths, their gaps, and how to prepare.
Then generate 5 tailored interview questions.

=== RETRIEVED RAG CONTEXT ===
Strongest Matches from Job Description:
{matches_text}

Weakest Matches / Gaps from Job Description:
{gaps_text}

Resume Snippet:
{resume_summary[:1000]}

Job Description Snippet:
{jd_summary[:1000]}

=== OUTPUT INSTRUCTIONS ===
Return strictly a valid JSON object with keys:
1. "matching_skills": A list of 2 to 3 objects:
   - "skill": Short, clean title of the matched skill (e.g. "Problem Solving & Clarity of Thought", "Python Programming")
   - "match_score": Integer percentage between 70 and 95 (e.g. 85)
   - "job_requirement": 1 simple sentence explaining what the employer is looking for.
   - "candidate_evidence": 1 simple sentence explaining how the candidate's background demonstrates this.
2. "skill_gaps": A list of 2 to 3 objects:
   - "skill": Short, clean title of the gap skill (e.g. "Big Data Architecture", "Cross-Functional Collaboration")
   - "match_score": Integer percentage between 25 and 55 (e.g. 35)
   - "job_requirement": 1 simple sentence explaining what the employer is looking for.
   - "advice": 1 practical sentence explaining why the interviewer may ask about this and how the candidate can address it (e.g. highlighting adaptability and quick learning).
3. "questions": A list of 5 objects:
   - "id": integer (1 to 5)
   - "category": string ("Matching Skill", "Technical Deep-Dive", "Skill Gap", "Adaptability", "Behavioral")
   - "question": string (the exact wording of the question)

Do NOT include any markdown formatting, backticks, or extra text outside the JSON object.
"""

    fallback_matches, fallback_gaps = create_fallback_skills_and_gaps(matching_skills, skill_gaps)
    default_questions = [
        {"id": 1, "category": "Matching Skill", "question": "Can you walk me through one of the primary technical projects mentioned in your resume that aligns with this role?"},
        {"id": 2, "category": "Technical Deep-Dive", "question": "What was the most challenging technical roadblock you encountered in your recent work, and how did you resolve it?"},
        {"id": 3, "category": "Skill Gap", "question": "This role requires familiarity with key technologies from the job description. What is your experience with them, and how do you approach learning new tools?"},
        {"id": 4, "category": "Adaptability", "question": "Tell me about a time when you had to adapt to an unfamiliar framework or codebase under a tight deadline."},
        {"id": 5, "category": "Behavioral", "question": "Describe a situation where you had a technical disagreement with a teammate. How did you handle it to reach a resolution?"}
    ]

    try:
        raw_text = generate_with_gemini(api_key, prompt)

        # Clean markdown code blocks if the model returned ```json ... ```
        cleaned_json = re.sub(r"^```json\s*", "", raw_text, flags=re.IGNORECASE)
        cleaned_json = re.sub(r"^```\s*", "", cleaned_json)
        cleaned_json = re.sub(r"```$", "", cleaned_json).strip()

        data = json.loads(cleaned_json, strict=False)

        # If data is a dict containing our expected keys
        if isinstance(data, dict):
            qs = data.get("questions", [])
            m_skills = data.get("matching_skills", [])
            s_gaps = data.get("skill_gaps", [])

            return {
                "matching_skills": m_skills if m_skills else fallback_matches,
                "skill_gaps": s_gaps if s_gaps else fallback_gaps,
                "questions": qs if len(qs) == 5 else default_questions
            }

        # If data is a list of questions directly (legacy format)
        if isinstance(data, list) and len(data) == 5:
            return {
                "matching_skills": fallback_matches,
                "skill_gaps": fallback_gaps,
                "questions": data
            }

    except Exception as e:
        print(f"[Notice] LLM question generation fallback used: {e}")

    # Fallback return
    return {
        "matching_skills": fallback_matches,
        "skill_gaps": fallback_gaps,
        "questions": default_questions
    }


def evaluate_answer(
    api_key: str,
    question: str,
    answer: str,
    category: str,
    resume_context: str,
    jd_context: str
) -> Dict[str, Any]:
    """
    Evaluates a candidate's answer against the question and role context using Gemini.
    Produces a numerical score (1-10) and 1-2 lines of constructive feedback.
    """
    prompt = f"""You are a professional hiring manager evaluating a candidate's interview answer.

Question ({category}): {question}
Candidate's Answer: {answer}

Evaluation criteria:
1. Relevance and clarity of the answer.
2. Demonstration of relevant problem-solving or domain knowledge.
3. Constructive feedback on what was strong and what was missing or could be improved.

Score the answer on a scale from 1.0 to 10.0 (where 10.0 is an exceptional, well-structured STAR-method response).

Return strictly a JSON object with:
- "score": number (between 1.0 and 10.0)
- "feedback": string (1-2 sentences highlighting key strength and specific improvement tip)

Do not add extra markdown or conversational text outside the JSON object.
"""
    try:
        raw_text = generate_with_gemini(api_key, prompt)
        cleaned_json = re.sub(r"^```json\s*", "", raw_text, flags=re.IGNORECASE)
        cleaned_json = re.sub(r"^```\s*", "", cleaned_json)
        cleaned_json = re.sub(r"```$", "", cleaned_json).strip()

        data = json.loads(cleaned_json)
        score = float(data.get("score", 7.0))
        feedback = str(data.get("feedback", "Good effort. Try incorporating specific metrics and results to strengthen your answer."))
        return {"score": min(max(score, 1.0), 10.0), "feedback": feedback}
    except Exception:
        # Fallback scoring if format varies
        return {
            "score": 7.0,
            "feedback": "Answer noted. To improve, structure your responses using the STAR method (Situation, Task, Action, Result) with tangible outcomes."
        }


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session_state():
    """Initializes Streamlit session state keys for the interview flow."""
    if "stage" not in st.session_state:
        st.session_state.stage = "setup"  # "setup" -> "interview" -> "completed"
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "current_q_idx" not in st.session_state:
        st.session_state.current_q_idx = 0
    if "last_played_q" not in st.session_state:
        st.session_state.last_played_q = -1
    if "current_audio_path" not in st.session_state:
        st.session_state.current_audio_path = None
    if "matching_skills" not in st.session_state:
        st.session_state.matching_skills = []
    if "skill_gaps" not in st.session_state:
        st.session_state.skill_gaps = []
    if "resume_name" not in st.session_state:
        st.session_state.resume_name = "Candidate Resume"
    if "jd_title" not in st.session_state:
        st.session_state.jd_title = "Target Job Role"
    if "current_eval" not in st.session_state:
        st.session_state.current_eval = None
    if "voice_transcription" not in st.session_state:
        st.session_state.voice_transcription = ""
    if "answer_submitted" not in st.session_state:
        st.session_state.answer_submitted = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(int(time.time()))
    if "speech_speed" not in st.session_state:
        st.session_state.speech_speed = 1.28


init_session_state()

# -----------------------------------------------------------------------------
# API Key Resolution
# -----------------------------------------------------------------------------
detected_key = get_gemini_api_key()
if detected_key:
    api_key = detected_key
else:
    api_key = st.session_state.get("gemini_api_key", "")


# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_interview, tab_past_sessions = st.tabs([
    "Mock Interview",
    "Past Sessions and Analytics"
])


# =============================================================================
# TAB 1: MOCK INTERVIEW
# =============================================================================
with tab_interview:
    col_header, col_reset = st.columns([4, 1])
    with col_header:
        st.header("AI Mock Interview Coach")
        st.caption("Upload your resume and a target job description to practice tailored, voice-enabled interview questions.")
    with col_reset:
        if st.session_state.stage != "setup":
            if st.button("Reset Interview", use_container_width=True):
                st.session_state.stage = "setup"
                st.session_state.questions = []
                st.session_state.current_q_idx = 0
                st.session_state.last_played_q = -1
                st.session_state.current_eval = None
                st.session_state.voice_transcription = ""
                st.session_state.answer_submitted = False
                st.rerun()

    # -------------------------------------------------------------------------
    # STAGE 1: SETUP & ANALYSIS
    # -------------------------------------------------------------------------
    if st.session_state.stage == "setup":
        st.subheader("Step 1: Upload Documents")

        if not detected_key:
            api_key_input = st.text_input(
                "Gemini API Key",
                value=st.session_state.get("gemini_api_key", ""),
                type="password",
                help="Your key is never logged or stored permanently."
            )
            if api_key_input:
                st.session_state.gemini_api_key = api_key_input
                api_key = api_key_input

        col_resume, col_jd = st.columns(2)

        with col_resume:
            st.markdown("#### Your Resume")
            resume_file = st.file_uploader(
                "Upload Resume (PDF format)",
                type=["pdf"],
                help="PDF will be parsed using pdfplumber and embedded locally."
            )

        with col_jd:
            st.markdown("#### Target Job Description")
            jd_input_type = st.radio(
                "Job Description Format",
                ["Paste Text", "Upload PDF"],
                horizontal=True
            )

            jd_text_input = ""
            jd_file = None
            if jd_input_type == "Paste Text":
                jd_text_input = st.text_area(
                    "Paste the job description here:",
                    height=200,
                    placeholder="Paste job title, responsibilities, required skills, qualifications..."
                )
            else:
                jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

        with st.expander("Voice Settings (Optional)", expanded=False):
            speech_speed = st.slider(
                "Interviewer Speech Pace",
                min_value=1.10,
                max_value=1.50,
                value=float(st.session_state.get("speech_speed", 1.28)),
                step=0.05,
                help="Controls how fast the AI interviewer speaks. 1.28x matches natural human interview conversation (~180 words/minute)."
            )
            st.session_state.speech_speed = speech_speed

        st.markdown("---")
        start_btn = st.button("Analyze Documents and Generate Questions", type="primary", use_container_width=True)

        if start_btn:
            if not api_key:
                st.error("Please provide a Gemini API Key in .env or the text field above to continue.")
            elif not resume_file:
                st.error("Please upload your resume in PDF format.")
            elif jd_input_type == "Paste Text" and not jd_text_input.strip():
                st.error("Please paste the job description text.")
            elif jd_input_type == "Upload PDF" and not jd_file:
                st.error("Please upload the job description PDF.")
            else:
                with st.spinner("Analyzing resume and job description using local RAG..."):
                    # 1. Parse documents
                    resume_bytes = resume_file.read()
                    resume_text = parsing.extract_text_from_pdf(resume_bytes)
                    st.session_state.resume_name = resume_file.name

                    if jd_input_type == "Paste Text":
                        jd_text = parsing.clean_text(jd_text_input)
                        st.session_state.jd_title = "Pasted Job Description"
                    else:
                        jd_bytes = jd_file.read()
                        jd_text = parsing.extract_text_from_pdf(jd_bytes)
                        st.session_state.jd_title = jd_file.name

                    if len(resume_text) < 50:
                        st.error("Could not extract readable text from the resume PDF. Please check the file.")
                        st.stop()
                    if len(jd_text) < 50:
                        st.error("Could not extract readable text from the job description. Please provide more text.")
                        st.stop()

                    # 2. Chunk text
                    resume_chunks = parsing.chunk_text(resume_text, chunk_size=350, overlap=40)
                    jd_chunks = parsing.chunk_text(jd_text, chunk_size=350, overlap=40)

                    # 3. Perform Basic RAG similarity analysis
                    rag_results = rag.analyze_matches_and_gaps(resume_chunks, jd_chunks, top_k=3)

                    # 4. Synthesize non-technical skill analysis AND 5 questions in ONE batched Gemini call
                    analysis_result = generate_interview_questions(
                        api_key,
                        resume_text,
                        jd_text,
                        rag_results["matching_skills"],
                        rag_results["skill_gaps"]
                    )

                    st.session_state.matching_skills = analysis_result.get("matching_skills", [])
                    st.session_state.skill_gaps = analysis_result.get("skill_gaps", [])
                    st.session_state.questions = analysis_result.get("questions", [])
                    st.session_state.current_q_idx = 0
                    st.session_state.last_played_q = -1
                    st.session_state.current_audio_path = None
                    st.session_state.session_id = str(int(time.time()))
                    st.session_state.stage = "interview"
                    st.session_state.current_eval = None
                    st.rerun()

    # -------------------------------------------------------------------------
    # STAGE 2: INTERACTIVE INTERVIEW
    # -------------------------------------------------------------------------
    elif st.session_state.stage == "interview":
        questions = st.session_state.questions
        q_idx = st.session_state.current_q_idx
        current_q = questions[q_idx]

        # Top Skill Analysis Accordion
        with st.expander("Skill Match Analysis: Your Strengths and Preparation Areas", expanded=False):
            st.caption(
                "This breakdown compares your resume with the job requirements in simple, plain English. "
                "Review your matching strengths to speak about them with confidence, and review your preparation areas "
                "so you can explain how you learn and adapt during the interview."
            )
            col_m, col_g = st.columns(2)
            with col_m:
                st.markdown("##### Strongest Matching Skills")
                if st.session_state.matching_skills:
                    for item in st.session_state.matching_skills:
                        skill_name = item.get("skill", "Matching Skill")
                        match_pct = item.get("match_score", 85)
                        req = item.get("job_requirement") or item.get("jd_chunk", "")
                        evidence = item.get("candidate_evidence") or item.get("best_resume_match", "")

                        st.markdown(f"**{skill_name}** - *{match_pct}% Match (High Alignment)*")
                        st.progress(match_pct / 100)
                        st.markdown(f"- **Employer Expectation:** {req}")
                        if evidence:
                            st.markdown(f"- **Your Background:** {evidence}")
                        st.markdown("")
                else:
                    st.info("General alignment with core role responsibilities.")

            with col_g:
                st.markdown("##### Topics to Prepare For (Skill Gaps)")
                if st.session_state.skill_gaps:
                    for item in st.session_state.skill_gaps:
                        skill_name = item.get("skill", "Skill Gap")
                        match_pct = item.get("match_score", 35)
                        req = item.get("job_requirement") or item.get("jd_chunk", "")
                        advice = item.get("advice") or "This requirement is less visible on your resume. Expect questions on how you learn and adapt."

                        st.markdown(f"**{skill_name}** - *Interview Focus Area ({match_pct}% Resume Coverage)*")
                        st.progress(match_pct / 100)
                        st.markdown(f"- **Employer Expectation:** {req}")
                        st.markdown(f"- **How to Answer:** {advice}")
                        st.markdown("")
                else:
                    st.info("No significant gaps identified. Expect standard behavioral and technical deep-dives.")

        # Progress bar
        progress_val = (q_idx + 1) / len(questions)
        st.progress(progress_val, text=f"Question {q_idx + 1} of {len(questions)}")

        # Question Header Box
        st.markdown(f"### Question {q_idx + 1}: *{current_q.get('category', 'Technical')}*")
        st.info(f"**\"{current_q.get('question')}\"**")

        # Ensure question audio filename is unique to this question, interview session, and speed setting
        speed_factor = float(st.session_state.get("speech_speed", 1.28))
        audio_filename = f"q_{q_idx + 1}_{st.session_state.session_id}_{int(speed_factor * 100)}.wav"
        audio_filepath = os.path.abspath(os.path.join(tts_stt.AUDIO_CACHE_DIR, audio_filename))

        # Synthesize question speech if not yet created for this question in this session
        if not os.path.exists(audio_filepath):
            success, saved_file = tts_stt.speak_text(
                current_q.get("question"),
                filename=audio_filename,
                speed=speed_factor
            )
            if success:
                st.session_state.current_audio_path = saved_file
            st.session_state.last_played_q = q_idx
        elif st.session_state.last_played_q != q_idx:
            st.session_state.current_audio_path = audio_filepath
            st.session_state.last_played_q = q_idx
            tts_stt.speak_text(
                current_q.get("question"),
                filename=audio_filename,
                speed=speed_factor
            )

        # In-browser audio player with autoplay + Replay button
        col_audio_player, col_replay = st.columns([3, 1])
        with col_audio_player:
            if os.path.exists(audio_filepath):
                with open(audio_filepath, "rb") as f:
                    q_audio_bytes = f.read()
                # Embed audio player for in-browser playback/review without simultaneous auto-play
                st.audio(q_audio_bytes, format="audio/wav", autoplay=False)
            else:
                st.caption("Audio generating...")

        with col_replay:
            if st.button("Replay Audio", key=f"replay_btn_{q_idx}"):
                tts_stt.speak_text(
                    current_q.get("question"),
                    filename=audio_filename,
                    speed=speed_factor
                )
                st.rerun()

        st.markdown("---")

        # Input Mode Selector
        input_mode = st.radio(
            "Select your answer mode:",
            ["Speak your answer", "Type your answer"],
            horizontal=True
        )

        final_answer_text = ""

        # VOICE INPUT PATH
        if input_mode == "Speak your answer":
            st.markdown("#### Voice Answer Input")
            st.info("**Click the microphone icon below to record. Speak your answer, then click stop.** Your speech will be automatically transcribed.")

            # Browser audio recorder with native Start / Stop / Live Timer / Waveform
            recorded_audio = st.audio_input(
                "Record your answer:",
                key=f"audio_record_{q_idx}_{st.session_state.session_id}"
            )

            # Auto-transcribe recorded browser audio
            if recorded_audio is not None:
                audio_bytes = recorded_audio.read()
                audio_hash = f"transcribed_{q_idx}_{len(audio_bytes)}"
                if st.session_state.get("last_transcribed_hash") != audio_hash:
                    with st.spinner("Transcribing your speech with SpeechRecognition..."):
                        ok, trans_text = tts_stt.transcribe_audio_bytes(audio_bytes)
                        if ok:
                            st.session_state[f"voice_edit_box_{q_idx}"] = trans_text
                            st.session_state.voice_transcription = trans_text
                            st.session_state["last_transcribed_hash"] = audio_hash
                            st.success("Transcribed successfully! Review or edit below:")
                            st.rerun()
                        else:
                            st.warning(f"{trans_text}")

            col_ans_header, col_ans_clear = st.columns([3, 1])
            with col_ans_header:
                st.markdown("**Your Transcribed Answer** *(feel free to review or edit before submitting):*")
            with col_ans_clear:
                if st.button("Clear Answer", key=f"clear_btn_{q_idx}"):
                    st.session_state[f"voice_edit_box_{q_idx}"] = ""
                    st.session_state.voice_transcription = ""
                    st.rerun()

            # Ensure session state key exists
            if f"voice_edit_box_{q_idx}" not in st.session_state:
                st.session_state[f"voice_edit_box_{q_idx}"] = st.session_state.voice_transcription

            final_answer_text = st.text_area(
                "Your Transcribed Answer",
                height=140,
                key=f"voice_edit_box_{q_idx}",
                label_visibility="collapsed"
            )

        # TEXT INPUT PATH
        else:
            final_answer_text = st.text_area(
                "Type your response to the interviewer:",
                height=160,
                placeholder="Structure your answer clearly, mentioning your experience, decisions, and results...",
                key=f"text_input_box_{q_idx}"
            )

        # SUBMIT AND EVALUATE
        col_submit, _ = st.columns([1, 2])
        with col_submit:
            submit_btn = st.button("Submit Answer for Evaluation", type="primary")

        if submit_btn:
            if not final_answer_text or not final_answer_text.strip():
                st.error("Please speak or type an answer before submitting.")
            else:
                with st.spinner("Evaluating your response with Gemini..."):
                    eval_result = evaluate_answer(
                        api_key,
                        question=current_q.get("question"),
                        answer=final_answer_text,
                        category=current_q.get("category", "General"),
                        resume_context=st.session_state.resume_name,
                        jd_context=st.session_state.jd_title
                    )

                    # Log to SQLite database
                    db.save_session(
                        question=current_q.get("question"),
                        answer=final_answer_text,
                        input_mode="Voice" if "Speak" in input_mode else "Text",
                        score=eval_result["score"],
                        feedback=eval_result["feedback"],
                        resume_name=st.session_state.resume_name,
                        jd_title=st.session_state.jd_title
                    )

                    st.session_state.current_eval = eval_result
                    st.session_state.answer_submitted = True

        # Display Evaluation if available for this question
        if st.session_state.current_eval:
            eval_data = st.session_state.current_eval
            st.markdown("#### Interviewer Evaluation")
            score_col, feedback_col = st.columns([1, 3])
            with score_col:
                score_val = eval_data["score"]
                st.metric("Score", f"{score_val} / 10")
            with feedback_col:
                st.info(f"**Feedback:** {eval_data['feedback']}")

            st.markdown("---")
            # Next Question or Finish Button
            if q_idx < len(questions) - 1:
                if st.button("Next Question", type="primary"):
                    st.session_state.current_q_idx += 1
                    st.session_state.current_eval = None
                    st.session_state.voice_transcription = ""
                    st.session_state.answer_submitted = False
                    st.rerun()
            else:
                if st.button("Finish Interview and View Summary", type="primary"):
                    st.session_state.stage = "completed"
                    st.rerun()

    # -------------------------------------------------------------------------
    # STAGE 3: INTERVIEW COMPLETED
    # -------------------------------------------------------------------------
    elif st.session_state.stage == "completed":
        st.balloons()
        st.success("Congratulations! You have completed the 5-question mock interview.")
        st.markdown("All your questions, answers, and scores have been logged to your local SQLite database.")

        st.subheader("What's Next?")
        st.markdown("- Switch to the **'Past Sessions and Analytics'** tab to inspect your performance trend.")
        st.markdown("- Practice again with another job description to strengthen your gap areas.")

        if st.button("Start Another Interview"):
            st.session_state.stage = "setup"
            st.session_state.questions = []
            st.session_state.current_q_idx = 0
            st.session_state.last_played_q = -1
            st.session_state.current_eval = None
            st.session_state.voice_transcription = ""
            st.rerun()


# =============================================================================
# TAB 2: PAST SESSIONS & PROGRESS DASHBOARD
# =============================================================================
with tab_past_sessions:
    st.header("Past Sessions and Progress Dashboard")
    st.caption("Track your interview preparation history, score progression, and input modes over time.")

    # Refresh button
    if st.button("Refresh Data"):
        st.rerun()

    # Aggregate SQLite statistics
    stats = db.get_session_stats()

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Questions Answered", stats["total_questions"])
    with m_col2:
        st.metric("Average Score", f"{stats['average_score']} / 10")
    with m_col3:
        st.metric("Highest Score", f"{stats['highest_score']} / 10")
    with m_col4:
        st.metric("Mode Breakdown", f"Voice: {stats['voice_count']} | Text: {stats['text_count']}")

    st.markdown("---")

    # Load records from SQLite
    df_sessions = db.get_all_sessions()

    if df_sessions.empty:
        st.info("No past interview sessions logged yet. Complete an interview to see your progress chart!")
    else:
        # Score trend line chart over time
        st.subheader("Score Progression Over Time")
        # Prepare chronologically sorted data for trend plotting
        chart_df = df_sessions.sort_values(by="id", ascending=True).copy()
        chart_df["Attempt #"] = range(1, len(chart_df) + 1)
        
        # Display Streamlit native line chart
        st.line_chart(
            chart_df,
            x="Attempt #",
            y="Score / 10",
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("Detailed Past Interview Logs")
        
        # Allow searching/filtering the table
        search_term = st.text_input("Search questions or answers:", placeholder="e.g. Python, Docker, Behavioral...")
        
        display_df = df_sessions.copy()
        if search_term:
            display_df = display_df[
                display_df["Question"].str.contains(search_term, case=False, na=False) |
                display_df["Answer"].str.contains(search_term, case=False, na=False)
            ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
