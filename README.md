# 🤖 Enterprise RAG & SQL Copilot

A full-stack **Retrieval-Augmented Generation (RAG)** application that lets users upload documents, ask natural-language questions, and receive AI-powered answers grounded in their own data — with optional SQL generation for structured datasets.

---

## 📐 Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      Frontend (HTML/CSS/JS)                │
│  Upload documents · Chat interface · Analytics dashboard   │
└──────────────────────────┬─────────────────────────────────┘
                           │  REST / JSON
┌──────────────────────────▼─────────────────────────────────┐
│                    FastAPI Backend                         │
│  /api/upload · /api/chat · /api/analytics                  │
│                                                            │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ LangChain  │  │ Groq LLM     │  │ Embedding Model   │   │
│  │ Pipeline   │──│ (ChatGroq)   │  │ (configurable)    │   │
│  └────────────┘  └──────────────┘  └───────────────────┘   │
└──────────────────────────┬─────────────────────────────────┘
                           │  SQLAlchemy + pgvector
┌──────────────────────────▼─────────────────────────────────┐
│              PostgreSQL 16 + pgvector                      │
│  documents · document_chunks (vector) · chat_sessions ·    │
│  chat_messages                                             │
└────────────────────────────────────────────────────────────┘
```

---

## 🗂 Project Structure

```
RAG_Agent/
├── backend/
│   ├── main.py              # FastAPI app entry point & lifespan
│   ├── database.py          # SQLAlchemy engine & session factory
│   ├── models.py            # ORM models (Document, Chunk, Chat…)
│   ├── schemas.py           # Pydantic request / response schemas
│   ├── crud.py              # Database CRUD helpers
│   ├── routers/
│   │   ├── upload.py        # Document upload & ingestion
│   │   ├── chat.py          # RAG chat sessions & messages
│   │   └── analytics.py     # Usage statistics endpoint
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # Single-page UI
│   ├── style.css            # Dark-theme styling
│   └── app.js               # Vanilla JS API client
├── docker-compose.yml       # PostgreSQL + pgvector container
└── README.md                # ← You are here
```

---

## 🚀 Getting Started

### Prerequisites

| Tool              | Version   |
|-------------------|-----------|
| Python            | ≥ 3.11    |
| Docker & Compose  | Latest    |
| Groq API Key      | Required  |

### 1. Start the Database

```bash
docker compose up -d
```

This launches a **PostgreSQL 16** instance with the **pgvector** extension on port `5432`.

### 2. Configure Environment

Create a `.env` file in the `backend/` directory:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_copilot
GROQ_API_KEY=gsk_your_key_here
```

### 3. Install Python Dependencies

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API docs are available at **http://localhost:8000/docs** (Swagger UI).

### 5. Open the Frontend

Open `frontend/index.html` directly in a browser, or serve it with any static file server:

```bash
# Quick option with Python
cd frontend
python -m http.server 3000
```

Then visit **http://localhost:3000**.

---

## 🔌 API Endpoints

| Method | Path                                     | Description                     |
|--------|------------------------------------------|---------------------------------|
| GET    | `/`                                      | Health check                    |
| POST   | `/api/upload/`                           | Upload a document               |
| GET    | `/api/upload/`                           | List all documents              |
| GET    | `/api/upload/{doc_id}`                   | Get document details            |
| POST   | `/api/chat/sessions`                     | Create a chat session           |
| GET    | `/api/chat/sessions`                     | List all sessions               |
| GET    | `/api/chat/sessions/{id}`                | Get session with messages       |
| POST   | `/api/chat/sessions/{id}/messages`       | Send a message & get AI reply   |
| GET    | `/api/analytics/`                        | Usage statistics                |

---

## 🛠 Tech Stack

- **Backend**: FastAPI · SQLAlchemy · LangChain · Groq (LLM)
- **Database**: PostgreSQL 16 · pgvector (vector similarity search)
- **Frontend**: Vanilla HTML / CSS / JavaScript
- **Infra**: Docker Compose

---

## 📝 License

This project is for educational and enterprise-internal use. Add your preferred license as needed.

