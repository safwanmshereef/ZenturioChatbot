# 🤖 Zenturio Chatbot — Intelligent Context-Aware Assistant

A production-grade chatbot built with **Streamlit** and the **OpenAI API**, featuring a custom sliding-window token optimizer and robust anti-hallucination prompt engineering.

Built for the **ZenturioTech AI Intern Assignment**.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Context-Aware Responses** | Maintains full conversation history so follow-up questions ("What are its advantages?") correctly resolve references |
| **Sliding-Window Token Optimization** | Custom token management using `tiktoken` — automatically truncates oldest exchanges when approaching the context limit while always preserving the system prompt |
| **Anti-Hallucination Prompts** | System prompt explicitly instructs the model to refuse fabrication, ask for clarification, and cite uncertainty |
| **No Repeated Answers** | Prompt engineering ensures the model tracks what it has already said and avoids redundancy |
| **Conversation Statistics** | Real-time sidebar showing token usage, message count, and context window utilization |
| **Streaming Responses** | Token-by-token streaming for a responsive, modern chat experience |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — Chat UI & deployment
- **OpenAI API** — LLM backend (GPT-4o-mini)
- **tiktoken** — Accurate BPE token counting
- **python-dotenv** — Local environment management

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/ZenturioChatbot.git
cd ZenturioChatbot
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file or use Streamlit secrets:

```bash
# Option A: .env file
echo OPENAI_API_KEY=sk-your-key-here > .env

# Option B: Streamlit secrets
mkdir -p .streamlit
echo 'OPENAI_API_KEY = "sk-your-key-here"' > .streamlit/secrets.toml
```

### 3. Run

```bash
streamlit run app.py
```

---

## 📦 Deployment (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the entrypoint
4. Add `OPENAI_API_KEY` under **Advanced Settings → Secrets**
5. Deploy!

---

## 🧪 Demo — Context Retention Test

```
User: Tell me about Python
Bot:  [Explains Python programming language]

User: What are its advantages?
Bot:  [Correctly understands "its" = Python, lists advantages without repeating the intro]
```

---

## 📐 Architecture

```
┌─────────────────────────────────────────┐
│           Streamlit Frontend            │
│  ┌──────────┐  ┌─────────────────────┐  │
│  │ Sidebar   │  │  Chat Interface     │  │
│  │ - Stats   │  │  - Message history  │  │
│  │ - Config  │  │  - Streaming output │  │
│  └──────────┘  └─────────────────────┘  │
├─────────────────────────────────────────┤
│         Token Optimization Layer        │
│  ┌──────────────────────────────────┐   │
│  │  Sliding Window Manager          │   │
│  │  - Count tokens (tiktoken)       │   │
│  │  - Truncate oldest exchanges     │   │
│  │  - Always preserve system prompt │   │
│  └──────────────────────────────────┘   │
├─────────────────────────────────────────┤
│            OpenAI API Layer             │
│  ┌──────────────────────────────────┐   │
│  │  GPT-4o-mini (streaming)         │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

---

## 📄 License

MIT — Free for educational and personal use.
