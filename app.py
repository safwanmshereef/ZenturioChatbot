"""
Zenturio Chatbot — Context-Aware AI Assistant
==============================================
Built for the ZenturioTech AI Intern Assignment.

Features:
  • Maintains full conversation history via Streamlit session state
  • Sliding-window token optimization (tiktoken-based)
  • Anti-hallucination & context-aware system prompt
  • Streaming responses via Google Gemini API (native SDK)
  • Auto-detects the best available Gemini model
  • Real-time token & message statistics
"""

import streamlit as st
import tiktoken
import google.generativeai as genai

# ────────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────────

ENCODING_NAME = "cl100k_base"   # BPE tokenizer for approximate token counting
MAX_CONTEXT_TOKENS = 3500       # Safe limit for demo purposes
RESERVED_FOR_RESPONSE = 800     # Tokens reserved for the model's reply
MAX_INPUT_TOKENS = MAX_CONTEXT_TOKENS - RESERVED_FOR_RESPONSE

SYSTEM_PROMPT = """You are Zenturio, an intelligent and helpful AI assistant created for ZenturioTech.

## Core Behavioral Rules

### 1. Context Awareness
- You MUST maintain full awareness of the entire conversation history provided to you.
- When the user uses pronouns like "it", "its", "they", "that", "this", etc., resolve them by referring back to the most recent relevant topic in the conversation.
- Example: If the user asks "Tell me about Python" and then asks "What are its advantages?", you MUST understand that "its" refers to Python.

### 2. No Repetition
- Track everything you have already told the user in this conversation.
- Do NOT repeat information you have already provided unless the user explicitly asks you to repeat or elaborate.
- If a follow-up question overlaps with a previous answer, acknowledge what was already said and provide NEW, additional information only.

### 3. Anti-Hallucination — Strict Honesty Policy
- If you are not confident in an answer, you MUST say: "I'm not fully certain about this, but here's what I know..."
- If you genuinely do not know something, say: "I don't have reliable information on that topic."
- NEVER fabricate facts, statistics, URLs, citations, dates, or names.
- Prefer being honest about uncertainty over providing a plausible-sounding but incorrect answer.

### 4. Clarification
- If the user's question is ambiguous, vague, or could have multiple interpretations, ASK for clarification before answering.
- Example: If the user asks "How do I set it up?", and multiple topics have been discussed, ask: "Are you referring to [Topic A] or [Topic B]?"

### 5. Response Quality
- Be concise but thorough. Organize responses with headings, bullet points, or numbered lists when appropriate.
- Provide practical, actionable information.
- Tailor your language complexity to match the user's apparent level of expertise.

You are a professional AI assistant. Follow these rules rigorously in every response."""


# ────────────────────────────────────────────────────────────────────
# MODEL AUTO-DETECTION
# ────────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    """Get the Gemini API key from Streamlit secrets or .env file."""
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ **Gemini API key not found!** Please set `GEMINI_API_KEY` in `.streamlit/secrets.toml` or a `.env` file.")
        st.stop()
    return api_key


@st.cache_resource
def connect_to_best_model(key):
    try:
        genai.configure(api_key=key)

        # Priority List: Tries Gemini 3.1/3.0 first, then 2.5, then 2.0, then 1.5
        candidates = [
            "gemini-3.1-pro",
            "gemini-3.1-flash",
            "gemini-3.0-pro",
            "gemini-3.0-flash",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]

        available_models = [m.name.replace("models/", "") for m in genai.list_models()]

        for c in candidates:
            # Check if candidate string exists in available models
            if any(c in m for m in available_models):
                # We found a match, now VERIFY it actually works
                try:
                    model = genai.GenerativeModel(c)
                    model.generate_content("test")
                    return c  # Success! Return this model
                except Exception:
                    continue  # Fails verification, try next candidate

        # If all candidates fail or None were found, try a generic fallback
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            model.generate_content("test")
            return "gemini-2.5-flash"
        except:
            return None
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────
# TOKEN COUNTING & CONTEXT WINDOW OPTIMIZATION
# ────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_tokenizer():
    """Load the tiktoken encoder once and cache it."""
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Count the number of tokens in a given text string."""
    enc = get_tokenizer()
    return len(enc.encode(text))


def count_message_tokens(messages: list[dict]) -> int:
    """
    Count total tokens across all messages.
    Uses tiktoken for approximate BPE token counting.
    Each message has ~4 overhead tokens (role, content markers).
    """
    total = 0
    for msg in messages:
        total += 4  # overhead per message
        total += count_tokens(msg.get("content", ""))
    total += 2  # reply priming
    return total


def optimize_context_window(messages: list[dict]) -> list[dict]:
    """
    Sliding-window token optimization.

    Strategy:
      1. Always preserve the system prompt (messages[0]).
      2. Keep the most recent user message (always include it).
      3. Drop the oldest user/assistant pairs first until
         total tokens fit within MAX_INPUT_TOKENS.
      4. If history is heavily truncated, inject a brief
         "[Earlier conversation truncated]" notice.
    """
    if not messages:
        return messages

    system_msg = messages[0]  # Always preserved
    conversation = messages[1:]  # Everything after system prompt

    # Fast path: if already within budget, return as-is
    total = count_message_tokens(messages)
    if total <= MAX_INPUT_TOKENS:
        return messages

    # Truncate: remove oldest messages until we fit
    truncated = False
    while len(conversation) > 1 and count_message_tokens([system_msg] + conversation) > MAX_INPUT_TOKENS:
        conversation.pop(0)
        truncated = True

    # Build the optimized list
    optimized = [system_msg]
    if truncated:
        optimized.append({
            "role": "system",
            "content": "[Note: Earlier parts of this conversation were truncated to stay within the context window. Continue responding based on the available context.]"
        })
    optimized.extend(conversation)

    return optimized


def convert_to_gemini_history(messages: list[dict]) -> list[dict]:
    """
    Convert our internal message format to Gemini's expected format.
    Skips system messages (handled via system_instruction).
    Maps 'assistant' role to 'model' role for Gemini.
    """
    history = []
    for msg in messages:
        if msg["role"] == "system":
            continue  # System prompt is passed via system_instruction
        role = "model" if msg["role"] == "assistant" else "user"
        history.append({"role": role, "parts": [msg["content"]]})
    return history


# ────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Zenturio Chatbot — ZenturioTech",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Main container */
    .stApp {
        background: linear-gradient(160deg, #0E1117 0%, #151922 40%, #1A1D29 100%);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        background: linear-gradient(135deg, rgba(108,99,255,0.12) 0%, rgba(130,87,255,0.08) 100%);
        border-radius: 16px;
        border: 1px solid rgba(108,99,255,0.2);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF, #A78BFA, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #9CA3AF;
        font-size: 0.9rem;
        font-weight: 400;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #131620 0%, #0E1117 100%);
        border-right: 1px solid rgba(108,99,255,0.15);
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(108,99,255,0.1) 0%, rgba(130,87,255,0.05) 100%);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108,99,255,0.15);
    }
    .stat-label {
        color: #9CA3AF;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        color: #F3F4F6;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        backdrop-filter: blur(10px);
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Chat input */
    .stChatInput > div {
        border-radius: 12px !important;
        border: 1px solid rgba(108,99,255,0.3) !important;
        background: rgba(26,29,41,0.8) !important;
    }
    .stChatInput > div:focus-within {
        border-color: #6C63FF !important;
        box-shadow: 0 0 0 3px rgba(108,99,255,0.2) !important;
    }

    /* Progress bar color */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6C63FF, #A78BFA) !important;
    }

    /* Badge */
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #6C63FF, #8B5CF6);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* Model badge */
    .model-badge {
        display: inline-block;
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-top: 0.5rem;
    }

    /* Feature list */
    .feature-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0;
        color: #D1D5DB;
        font-size: 0.8rem;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(108,99,255,0.5); }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# INITIALIZATION — API KEY & MODEL DETECTION
# ────────────────────────────────────────────────────────────────────

api_key = get_api_key()

# Auto-detect best model (cached — only runs once per session)
if "active_model" not in st.session_state:
    with st.spinner("🔍 Detecting best available Gemini model..."):
        best_model = connect_to_best_model(api_key)
    if best_model is None:
        st.error("❌ **Could not connect to any Gemini model.** Please check your API key and quota.")
        st.stop()
    st.session_state.active_model = best_model

active_model = st.session_state.active_model

# Configure genai globally
genai.configure(api_key=api_key)


import sqlite3
import uuid
from datetime import datetime

# ────────────────────────────────────────────────────────────────────
# DATABASE INITIALIZATION
# ────────────────────────────────────────────────────────────────────

DB_FILE = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id) ON DELETE CASCADE
        )
    ''')
    conn.commit()
    conn.close()

def get_all_sessions():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT session_id, name, created_at FROM chat_sessions ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()
    return [{"session_id": r[0], "name": r[1], "created_at": r[2]} for r in rows]

def create_new_session(name="New Chat") -> str:
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO chat_sessions (session_id, name) VALUES (?, ?)', (session_id, name))
    conn.commit()
    conn.close()
    save_message_to_db(session_id, "system", SYSTEM_PROMPT)
    return session_id

def rename_session(session_id: str, new_name: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('UPDATE chat_sessions SET name = ? WHERE session_id = ?', (new_name, session_id))
    conn.commit()
    conn.close()

def delete_session(session_id: str):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON")
    c = conn.cursor()
    c.execute('DELETE FROM chat_sessions WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()

def load_messages_from_db(session_id: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY id ASC', (session_id,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return [{"role": "system", "content": SYSTEM_PROMPT}]
    return [{"role": r[0], "content": r[1]} for r in rows]

def save_message_to_db(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)', (session_id, role, content))
    conn.commit()
    conn.close()

init_db()

# ────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ────────────────────────────────────────────────────────────────────

if "current_session_id" not in st.session_state:
    sessions = get_all_sessions()
    if sessions:
        st.session_state.current_session_id = sessions[0]["session_id"]
    else:
        st.session_state.current_session_id = create_new_session()

if "messages" not in st.session_state:
    st.session_state.messages = load_messages_from_db(st.session_state.current_session_id)

if "total_tokens_used" not in st.session_state:
    st.session_state.total_tokens_used = 0

if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0


# ────────────────────────────────────────────────────────────────────
# SIDEBAR — Stats & Info
# ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size: 2.5rem;">🤖</span>
        <h2 style="background: linear-gradient(135deg, #6C63FF, #A78BFA);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-weight: 700; margin: 0.5rem 0 0.25rem 0;">
            Zenturio
        </h2>
        <span class="badge">AI ASSISTANT</span>
        <br>
        <span class="model-badge">⚡ {active_model}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Live stats
    user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
    assistant_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
    current_tokens = count_message_tokens(st.session_state.messages)
    utilization = min(current_tokens / MAX_INPUT_TOKENS * 100, 100)

    st.markdown("#### 📊 Conversation Stats")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Messages", user_msgs + assistant_msgs)
    with col2:
        st.metric("API Calls Made", st.session_state.api_calls)
    
    st.metric("Current Context Tokens", f"{current_tokens:,}")

    # Context window utilization bar
    st.markdown("#### 🧠 Context Window")
    st.progress(min(utilization / 100, 1.0))
    st.caption(f"{current_tokens:,} / {MAX_INPUT_TOKENS:,} tokens ({utilization:.1f}%)")

    st.markdown("---")

    # Features
    st.markdown("#### ✨ Features")
    features = [
        ("🧠", "Context-aware responses"),
        ("🔄", "Sliding-window optimization"),
        ("🛡️", "Anti-hallucination prompts"),
        ("🚫", "No repeated answers"),
        ("⚡", "Streaming responses"),
        ("📊", "Real-time token tracking"),
    ]
    for icon, text in features:
        st.markdown(f"""<div class="feature-item">{icon} {text}</div>""", unsafe_allow_html=True)

    # Chat Manager
    st.markdown("#### 💬 Chat Sessions")
    
    # Fetch available sessions
    all_sessions = get_all_sessions()
    session_options = {s["session_id"]: s["name"] for s in all_sessions}

    def on_session_change():
        st.session_state.current_session_id = st.session_state.selectbox_session_id
        st.session_state.messages = load_messages_from_db(st.session_state.current_session_id)
        st.session_state.total_tokens_used = 0
        st.session_state.api_calls = 0

    option_keys = list(session_options.keys())
    active_index = option_keys.index(st.session_state.current_session_id) if st.session_state.current_session_id in option_keys else 0

    st.selectbox(
        "Active Chat",
        options=option_keys,
        index=active_index,
        format_func=lambda x: session_options.get(x, "Unknown"),
        key="selectbox_session_id",
        on_change=on_session_change,
        label_visibility="collapsed"
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("➕ New Space", use_container_width=True):
            new_id = create_new_session("New Chat")
            st.session_state.current_session_id = new_id
            st.session_state.messages = load_messages_from_db(new_id)
            st.session_state.total_tokens_used = 0
            st.session_state.api_calls = 0
            st.rerun()
    with colB:
        if st.button("🗑️ Delete", use_container_width=True):
            delete_session(st.session_state.current_session_id)
            updated_sessions = get_all_sessions()
            if updated_sessions:
                next_id = updated_sessions[0]["session_id"]
            else:
                next_id = create_new_session("New Chat")
            st.session_state.current_session_id = next_id
            st.session_state.messages = load_messages_from_db(next_id)
            st.session_state.total_tokens_used = 0
            st.session_state.api_calls = 0
            st.rerun()

    # Edit Chat Name
    with st.popover("✏️ Rename Chat", use_container_width=True):
        new_name = st.text_input("New Name", value=session_options.get(st.session_state.current_session_id, "New Chat"))
        if st.button("Save Name", use_container_width=True):
            rename_session(st.session_state.current_session_id, new_name)
            st.rerun()

    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0; color: #6B7280; font-size: 0.7rem;">
        Built for <strong>ZenturioTech</strong><br>
        Powered by {active_model} + tiktoken
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# MAIN CHAT INTERFACE
# ────────────────────────────────────────────────────────────────────

# Header
st.title("🤖 Zenturio Chatbot")
st.caption(f"Context-aware AI assistant with sliding-window token optimization · **{active_model}**")
st.divider()

# Render chat history (skip the system prompt)
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask Zenturio anything..."):
    # Display user message immediately
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Append to state and DB
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message_to_db(st.session_state.current_session_id, "user", prompt)

    # Optimize context window (sliding-window truncation)
    optimized_messages = optimize_context_window(st.session_state.messages)

    # Update state if truncation occurred
    if len(optimized_messages) != len(st.session_state.messages):
        st.session_state.messages = optimized_messages

    # Build Gemini model with system instruction
    model = genai.GenerativeModel(
        active_model,
        system_instruction=SYSTEM_PROMPT,
    )

    # Convert optimized messages to Gemini format (skip system messages)
    gemini_history = convert_to_gemini_history(optimized_messages)

    # Separate the last user message from history for send_message
    chat_history = gemini_history[:-1] if len(gemini_history) > 1 else []
    last_user_msg = gemini_history[-1]["parts"][0] if gemini_history else prompt

    # Create chat session with history and send the latest message
    chat = model.start_chat(history=chat_history)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                response = chat.send_message(last_user_msg, stream=True)

                # Stream the response using st.write_stream
                def stream_chunks():
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text

                full_response = st.write_stream(stream_chunks())
            except Exception as e:
                full_response = f"⚠️ An error occurred: {str(e)}"
                st.error(full_response)

    # Append assistant response to state and DB
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_message_to_db(st.session_state.current_session_id, "assistant", full_response)
    st.session_state.api_calls += 1

    # Track tokens
    st.session_state.total_tokens_used = count_message_tokens(st.session_state.messages)

    # Rerun to update sidebar stats
    st.rerun()
