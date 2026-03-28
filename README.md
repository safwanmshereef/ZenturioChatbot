# Zenturio Chatbot 🤖

A production-grade, context-aware AI assistant built for the **ZenturioTech AI Intern Assignment**. Powered by the native Google Gemini SDK, Streamlit, and advanced Prompt Engineering.

## ✨ Features and Capabilities

This application was engineered to precisely fulfill and exceed all assignment requirements:

- **Context-Aware Responses:** The bot maintains full conversation history, intelligently resolving pronouns (like "its") across multiple turns.
- **Sliding-Window Token Optimizer:** Uses `tiktoken` to accurately track BPE tokens. Once the designated context window limit is reached, it dynamically truncates the oldest messages while permanently preserving the System Prompt to prevent memory crashes.
- **Anti-Hallucination & Zero-Repetition:** A meticulously crafted 400+ token System Prompt enforces extreme honesty, prevents the bot from repeating previous answers, and explicitly instructs it to ask for clarification on vague queries.
- **Auto Model Detection Engine:** Gracefully scans your Gemini API Key's permissions and automatically connects via the newest, most capable model available (prioritizing `gemini-3.1-pro` → `gemini-3.0` → `gemini-2.5` → `gemini-2.0`).
- **Real-Time Analytics Dashboard:** Tracks active Session Tokens, Total API Calls, Context Window Utilization, and parsed message queues in a live, glassmorphism-styled sidebar.
- **Data Persistence & Multi-Chat Management:** Chat history is permanently saved using a built-in SQLite database (`chat_history.db`). Users can instantly start new chat sessions, effortlessly switch between them, rename their chats, or explicitly delete them entirely from the sidebar without ever losing history on an accidental browser refresh.

## 🚀 Live Demo

You can interact with the live deployed version of this chatbot on Streamlit Community Cloud:
> **[ZenturioTech Chatbot Demo](https://zenturiotechchatbot.streamlit.app)**

## 🛠️ Tech Stack

- **Frontend Framework:** Streamlit (Python)
- **LLM Engine:** Google Generative AI (Native Gemini Python SDK)
- **Tokenization:** Tiktoken (cl100k_base tokenizer)
- **Database:** SQLite3 (Native Python)
- **Environment Management:** Python-dotenv

## ⚙️ How to Run Locally

1. **Clone the repository:**

   ```bash
   git clone https://github.com/safwanmshereef/ZenturioChatbot.git
   cd ZenturioChatbot
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your API Key:**

   Create a `.env` file in the root directory and securely add your Gemini API Key:

   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

4. **Launch the application:**

   ```bash
   python -m streamlit run app.py
   ```

   The app will automatically open in your browser at `http://localhost:8501`.

## 📜 Assignment Checklist

- [x] Use LLM API (Google Gemini natively)
- [x] Maintain Chat history (SQLite Database Persistence & Multi-Chat Routing)
- [x] Context window handling (Safe token limits implemented)
- [x] Token optimization (tiktoken sliding-window truncation)
- [x] Context-aware prompt template (`SYSTEM_PROMPT` in `app.py`)
- [x] Instruction to avoid hallucination (Rule #3)
- [x] Instruction to ask clarification if unsure (Rule #4)
- [x] Multi-turn context retention & follow-up understanding confirmed

---
*Developed by Safwan M Shereef for Zenturio.*

