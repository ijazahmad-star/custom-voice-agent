# NOVA - AI Voice Agent with RAG (Supabase & pgvector)

NOVA is a high-performance, professional voice-to-voice agentic system designed for seamless, session-based interactions. It integrates advanced Speech-To-Text (STT), Retrieval-Augmented Generation (RAG) using Supabase, and high-fidelity Text-To-Speech (TTS) to provide a premium hands-free experience.

![NOVA UX](https://raw.githubusercontent.com/ijazahmad-star/custom-voice-agent/main/architecture.png) _Note: Replace with actual screenshot_

## 🌟 Key Features

- **🗣️ Natural Voice interaction**: Real-time voice-to-voice communication with low latency.
- **📚 Smart Knowledge Base (RAG)**: Upload PDF documents to your private knowledge base, indexed in **Supabase** via `pgvector`.
- **🧠 Agentic Intelligence**: Powered by LangGraph for structured tool-use and persistent memory.
- **🌊 Fluid UI**: A stunning Next.js frontend with glassmorphism, micro-animations, and real-time volume visualization.
- **⚡ Performance First**: Local model inference (Whisper, Kokoro) combined with Groq cloud inference for ultra-fast reasoning.

---

## 🛠️ Tech Stack

| Component        | Technology                              |
| :--------------- | :-------------------------------------- |
| **Frontend**     | Next.js 15, TailwindCSS, Lucide-React   |
| **Backend**      | FastAPI, Python 3.13                    |
| **Database**     | Supabase (PostgreSQL + `pgvector`)      |
| **RAG / Memory** | LangChain, LangGraph, Alembic           |
| **Models (STT)** | OpenAI Whisper (Local via Transformers) |
| **Models (LLM)** | Meta Llama 3/4 (via Groq Cloud)         |
| **Models (TTS)** | Kokoro-ONNX (High-fidelity local TTS)   |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** (Recommend 3.13 for best performance)
- **Node.js 18+**
- **Supabase Account**: Create a project and obtain your `URL` and `SERVICE_ROLE_KEY`.
- **Groq API Key**: Optional but recommended for faster LLM response.

### 1. Database Setup (Supabase)

NOVA uses **Alembic** to manage its database schema.

1.  Create a new Supabase project.
2.  Enable the `pgvector` extension (Alembic will handle this, but ensure your user has permissions).
3.  Add the `DATABASE_URL` (Direct Connection String) to your `.env` file.

### 2. Backend Setup

```bash
cd backend
uv venv
source .venv/bin/activate
uv sync
```

**Configure `.env`:**

```env
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_role_key
DATABASE_URL=postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres
groq_api_key=your_groq_key
# HF_TOKEN=optional_hf_token
```

**Run Migrations:**

```bash
uv run alembic upgrade head
```

**Start the Server:**

```bash
uv run uvicorn app:app --port 8000 --reload
```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` to interact with NOVA.

---

## 📖 Usage Guide

1.  **Connect**: Click the **"Connect to Nova"** button to start a session.
2.  **Speak**: NOVA detects your speech automatically. Just talk naturally.
3.  **Upload Knowledge**: Before a session, you can upload **PDFs**. These are chunked, embedded locally, and uploaded to Supabase.
4.  **Inquire**: Ask NOVA questions about the uploaded content. She will retrieve the relevant context and provide accurate answers.

---

## 🏗️ Architecture

NOVA follows a modular factory pattern:

- `core/agent.py`: Orchestrates STT -> LLM -> TTS.
- `core/knowledge.py`: Manages vector indexing in Supabase.
- `core/factory.py`: Constructs the LangGraph React agent with custom tools.
- `core/database/models.py`: Defines the `documents` schema for pgvector.

---
