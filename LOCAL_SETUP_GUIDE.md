# ThinkLife Local Development Setup Guide

This guide will help you set up and run ThinkxLife locally on your machine. The stack consists of a Python backend (FastAPI), a Next.js frontend, and Keycloak for authentication.

## Prerequisites Installation

### 1. Install Node.js (Required for Frontend)

#### Windows:
1. Visit [nodejs.org](https://nodejs.org/)
2. Download the LTS version (recommended)
3. Run the installer and follow the setup wizard
4. Restart your terminal/command prompt

#### macOS:
```bash
# Option 1: Direct download
# Visit https://nodejs.org/ and download the LTS version

# Option 2: Using Homebrew (if you have it)
brew install node
```

#### Linux (Ubuntu/Debian):
```bash
# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### 2. Install Python (Required for Backend)

#### Windows:
1. Visit [python.org](https://www.python.org/downloads/)
2. Download Python 3.9 or higher
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Run the installer

#### macOS:
```bash
# Using Homebrew
brew install python
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### 3. Install Docker (Required for Keycloak)

Keycloak runs in a Docker container for local development.

#### Windows/macOS:
1. Download and install **Docker Desktop** from [docker.com](https://www.docker.com/products/docker-desktop/).
2. Start Docker Desktop.

#### Linux:
Follow the official instructions to install Docker Engine and Docker Compose for your distribution.

### 4. Install Git (if not already installed)

#### Windows:
Download from [git-scm.com](https://git-scm.com/download/win)

#### macOS:
```bash
brew install git
```

#### Linux:
```bash
sudo apt install git
```

## Project Setup

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd ThinkLife
```

### Step 2: Backend Setup (Python)

#### Navigate to Backend Directory
```bash
cd backend
```

#### Create Python Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate
```

#### Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### Create Backend Environment File
Create a file named `.env` in the `backend/` directory with the following content:

```env
# --- Which LLM the chatbot (Zoe) uses -------------------------------
# Pick ONE provider: openai | anthropic | gemini | openrouter  (default: openai)
LLM_PROVIDER_TYPE=openai
# Optional. Leave unset to use the provider's default model (see table below).
# LLM_PROVIDER_MODEL=gpt-4o-mini

# --- API keys: set the key for the provider you selected ------------
OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=
# GEMINI_API_KEY=
# OPENROUTER_API_KEY=

# --- Which model builds/searches Zoe's knowledge base (embeddings) --
# auto (default) = OpenAI if OPENAI_API_KEY is set, otherwise a free local model.
# Or choose: local | openai | openrouter | gemini  (uses that provider's key above)
EMBEDDING_PROVIDER=auto
# EMBEDDING_MODEL=   # unset = provider default (see "Switching LLM Providers")

# OpenAI generation defaults (used when LLM_PROVIDER_TYPE=openai)
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=2000
OPENAI_TEMPERATURE=0.7

# Vector Database Configuration (OPTIONAL)
# CHROMA_DB_DIR=chroma_db

# Server Configuration
PORT=8000
DEBUG=True

# CORS Settings
CORS_ORIGINS=http://localhost:3000
```

> You only need the key for the provider you selected. Nothing else requires
> an OpenAI key (see "Switching LLM Providers" below).

### Step 3: Keycloak Setup (Docker)

We use Docker Compose to run Keycloak and its database locally.

#### Start Keycloak
From the **root** of the project (where `docker-compose.yml` is located):

```bash
docker-compose up -d
```

This will start Keycloak on `http://localhost:8080` and a PostgreSQL database.

*   **Admin Console**: http://localhost:8080
*   **Admin User**: `admin`
*   **Admin Password**: `admin`

Wait a minute for Keycloak to fully start up.

### Step 4: Frontend Setup (Node.js)

#### Navigate to Frontend Directory
```bash
cd frontend
```

#### Install Node.js Dependencies
```bash
npm install
```

#### Create Frontend Environment File
Create a file named `.env.local` in the `frontend/` directory with the following content:

```env
# Next.js Configuration
NEXT_PUBLIC_APP_URL=http://localhost:3000

# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# Keycloak Configuration
NEXT_PUBLIC_KEYCLOAK_URL=http://localhost:8080
NEXT_PUBLIC_KEYCLOAK_REALM=thinklife
NEXT_PUBLIC_KEYCLOAK_CLIENT_ID=thinklife-frontend
```

*Note: You may need to configure the Realm and Client in Keycloak manually if not pre-configured.*

## Running the Application

You need to run three things: Keycloak (via Docker), the Backend, and the Frontend.

### 1. Start Keycloak (if not running)
```bash
# From project root
docker-compose up -d
```

### 2. Start Backend
Open a new terminal:
```bash
cd backend
# Activate venv if needed: source .venv/bin/activate
python3 main.py
```

### 3. Start Frontend
Open another terminal:
```bash
cd frontend
npm run dev
```

## Switching LLM Providers

The chat model is chosen by two settings in `backend/.env`. Restart the backend after changing them.

| `LLM_PROVIDER_TYPE` | Key to set | Default model (when `LLM_PROVIDER_MODEL` is unset) | Other allowed models |
|---|---|---|---|
| `openai` (default) | `OPENAI_API_KEY` | value of `OPENAI_MODEL` (`gpt-4o-mini`) | `gpt-4o`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo` |
| `anthropic` | `ANTHROPIC_API_KEY` | see `brain/providers/provider_registry.py` | listed there |
| `gemini` | `GEMINI_API_KEY` | see `brain/providers/provider_registry.py` | listed there |
| `openrouter` | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` | `anthropic/claude-3.5-sonnet`, `meta-llama/llama-3.1-70b-instruct`, `google/gemini-flash-1.5` |

Example: use OpenRouter with Claude 3.5 Sonnet.

```env
LLM_PROVIDER_TYPE=openrouter
LLM_PROVIDER_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_API_KEY=sk-or-...
```

Notes:
- Get an OpenRouter key at https://openrouter.ai/keys. Model names there look like `vendor/model`.
- The model must belong to the chosen provider. A mismatch (for example `gpt-4o-mini` with `openrouter`) logs a "Startup configuration warning" and requests will fail. In production (`ENVIRONMENT=production`) the backend refuses to start.
- A missing key for the selected provider gives the same warning. Check the backend log at startup.
- To allow a new model, add it to `PROVIDER_CONFIGS` in `backend/brain/providers/provider_registry.py`.
- **Knowledge-base embeddings are chosen separately** with `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`:

  | `EMBEDDING_PROVIDER` | Key needed | Default model |
  |---|---|---|
  | `auto` (default) | none | `openai` if `OPENAI_API_KEY` is set, otherwise `local` |
  | `local` | none | MiniLM-L6-v2, downloaded once (~80 MB, to `~/.cache/chroma`) and run on your machine |
  | `openai` | `OPENAI_API_KEY` | `text-embedding-3-small` |
  | `openrouter` | `OPENROUTER_API_KEY` | `openai/text-embedding-3-small` (also e.g. `qwen/qwen3-embedding-0.6b`) |
  | `gemini` | `GEMINI_API_KEY` | `gemini-embedding-001` |

  **An index only works with the model that built it.** After changing `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL`, rebuild it from `backend/` with `python -m agents.zoe.utils.knowledge_index_builder` (needs the source files `data/context.txt`, `data/knowledge_base.json`, `data/empathetic_dialogues_train.csv`; they are not in the repo, and the shipped `chroma.sqlite3` is empty). If the chosen embedding provider has no key, knowledge search is switched off and Zoe still answers.
- **Still OpenAI-only:** text-to-speech (switched off without `OPENAI_API_KEY`), the LLM-judge evaluators in `backend/evaluators/` and the separate `faq-bot-rag/` app.
- The frontend needs no change; it only talks to the backend.

## Accessing the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Keycloak Admin**: http://localhost:8080

## Troubleshooting

### Port Conflicts
If ports 3000, 8000, or 8080 are in use:
- **Frontend**: `npm run dev -- -p 3001`
- **Backend**: Update `PORT` in `.env`
- **Keycloak**: Update ports in `docker-compose.yml`

### Keycloak Connection Issues
- Ensure Docker is running.
- Check logs: `docker-compose logs -f keycloak`
- Ensure `NEXT_PUBLIC_KEYCLOAK_URL` in frontend `.env.local` matches the Docker setup.

### Chatbot Not Answering / Wrong Provider
- Look for `Startup configuration warning` in the backend log: it names a missing key or an invalid `LLM_PROVIDER_TYPE` / `LLM_PROVIDER_MODEL`.
- Check `GET /health/ready` for per-provider status and circuit-breaker state.

### Python Issues
- Ensure you are using the virtual environment (`.venv`).
- Upgrade pip: `pip install --upgrade pip`
