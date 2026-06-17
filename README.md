# Invexsai — Semantic Agent Discovery Platform

> **Website:** [invexsai.com](https://invexsai.com)
> > **Course:** Advanced Text Processing
> > > **Description:** Semantic search and discovery for AI agents using NLP and vector embeddings
> > >
> > > ## Overview
> > >
> > > Invexsai is a full-stack platform for **semantic agent discovery** — enabling intelligent search and retrieval of AI agents based on natural language descriptions, capability embeddings, and semantic similarity. Built for the Advanced Text Processing course, it combines modern NLP techniques with a production-grade microservice architecture to make AI agent fleets discoverable and queryable via meaning, not just keywords.
> > >
> > > ## Architecture
> > >
> > > ```
> > > ┌─────────────────────────────────────────────────────────┐
> > > │                  Frontend (React / Vite)                │
> > > │          Semantic search UI · Agent explorer            │
> > > └──────────────────────┬──────────────────────────────────┘
> > >                        │ HTTP / REST
> > > ┌──────────────────────▼──────────────────────────────────┐
> > > │               Backend (Go / Gin REST API)               │
> > > │  /v1/agents/register   /v1/agents/search (semantic)     │
> > > │  /v1/agents/heartbeat  /v1/fleet                        │
> > > │             Auth: X-API-Key middleware                  │
> > > └──────┬──────────────────────────────────────┬───────────┘
> > >        │ pgx                                  │ embeddings
> > > ┌──────▼───────┐                  ┌───────────▼───────────┐
> > > │  PostgreSQL  │                  │   Semantic / NLP Layer │
> > > │  agents      │                  │   Sentence Transformers│
> > > │  heartbeats  │                  │   Vector similarity    │
> > > │  cost_events │                  │   Agent capability map │
> > > │  api_keys    │                  └───────────────────────┘
> > > └──────────────┘
> > >
> > > ┌─────────────────────────────────────────────────────────┐
> > > │              Python SDK  (pip install invexsai)         │
> > > │   Async HTTP client · LangChain · AutoGen · CrewAI     │
> > > └─────────────────────────────────────────────────────────┘
> > > ```
> > >
> > > ## Key Features
> > >
> > > - **Semantic Search** — discover agents by describing what you need in plain English; embeddings match by capability rather than exact name
> > > - - **Agent Registration & Heartbeat** — register agents with capability descriptions; monitor liveness in real time
> > >   - - **Cost Tracking** — log per-LLM-call cost events and visualize fleet spend
> > >     - - **NLP Capability Mapping** — the `semantic/` module embeds agent descriptions using Sentence Transformers and computes similarity scores for ranking
> > >       - - **Multi-Framework SDK** — first-class integrations for LangChain, AutoGen, and CrewAI agents
> > >        
> > >         - ## Repository Structure
> > >        
> > >         - ```
> > >           Semantic-Agent-Discovery/
> > >           ├── .github/workflows/       # CI pipeline (build, test, deploy)
> > >           ├── backend/                 # Go REST API
> > >           │   ├── cmd/server/main.go   # Entry point
> > >           │   ├── internal/api/        # Handlers, middleware, router
> > >           │   ├── internal/db/         # PostgreSQL pool + migrations
> > >           │   ├── internal/services/   # Business logic
> > >           │   └── internal/types/      # Shared type definitions
> > >           ├── frontend/                # React + TypeScript dashboard
> > >           │   └── src/
> > >           │       ├── components/      # AgentCard, SearchBar, FleetTable
> > >           │       ├── hooks/           # useAgentSearch, useFleet
> > >           │       └── api/             # REST client
> > >           ├── sdk/                     # Python SDK
> > >           │   └── invexsai/
> > >           │       ├── client.py        # Async HTTP client
> > >           │       ├── heartbeat.py     # Heartbeat scheduler
> > >           │       ├── pricing.py       # Cost calculation
> > >           │       └── handlers/        # LangChain & AutoGen integrations
> > >           ├── semantic/                # NLP / semantic layer
> > >           │   ├── embeddings.py        # Sentence Transformer embedding pipeline
> > >           │   ├── similarity.py        # Cosine similarity agent ranking
> > >           │   └── capability_map.py    # Agent capability graph
> > >           ├── demo/                    # Example agent scripts
> > >           ├── infra/                   # Docker Compose + GCP Cloud Build
> > >           └── .env.example             # Environment variable template
> > >           ```
> > >
> > > ## Tech Stack
> > >
> > > | Layer | Technology |
> > > |-------|-----------|
> > > | Backend API | Go 1.23, Gin, pgx v5 |
> > > | Database | PostgreSQL 15 |
> > > | NLP / Embeddings | Python, Sentence Transformers, scikit-learn |
> > > | Frontend | React 18, TypeScript, Vite, Tailwind CSS |
> > > | Python SDK | Python 3.9+, httpx, langchain-core |
> > > | Framework integrations | LangChain, AutoGen, CrewAI |
> > > | Infrastructure | Docker Compose (local), Google Cloud Build (prod) |
> > > | Auth | API key via X-API-Key header |
> > >
> > > ## Quick Start
> > >
> > > ### Prerequisites
> > >
> > > - Docker & Docker Compose
> > > - - Go 1.23+
> > >   - - Node.js 18+
> > >     - - Python 3.9+
> > >      
> > >       - ### 1. Clone & configure
> > >      
> > >       - ```bash
> > >         git clone https://github.com/MrSocial0079/Semantic-Agent-Discovery.git
> > >         cd Semantic-Agent-Discovery
> > >         cp .env.example .env
> > >         # Fill in your values
> > >         ```
> > >
> > > ### 2. Start the stack
> > >
> > > ```bash
> > > cd infra
> > > docker-compose up --build
> > > # Backend: http://localhost:8080
> > > # Frontend: http://localhost:5173
> > > ```
> > >
> > > ### 3. Install the Python SDK
> > >
> > > ```bash
> > > pip install invexsai
> > > # With framework extras:
> > > pip install "invexsai[langchain]"
> > > pip install "invexsai[autogen]"
> > > ```
> > >
> > > ## Semantic Search Usage
> > >
> > > ```python
> > > from invexsai import InvexsaiClient
> > >
> > > client = InvexsaiClient(base_url="http://localhost:8080", api_key="your_key")
> > >
> > > # Register an agent with a natural language capability description
> > > agent = await client.register_agent(
> > >     name="summarization-agent",
> > >     framework="langchain",
> > >     version="1.0.0",
> > >     description="Summarizes long documents and extracts key insights using GPT-4o"
> > > )
> > >
> > > # Semantic search: find agents by describing what you need
> > > results = await client.search_agents(
> > >     query="I need an agent that can summarize research papers",
> > >     top_k=5
> > > )
> > > ```
> > >
> > > ## NLP Pipeline (`semantic/`)
> > >
> > > The semantic discovery layer uses **Sentence Transformers** to embed agent capability descriptions into dense vectors, then ranks candidates by cosine similarity to a query embedding:
> > >
> > > 1. Agent registers with a natural language `description` field
> > > 2. 2. Backend calls the semantic service to embed the description
> > >    3. 3. Embeddings are stored alongside agent metadata
> > >       4. 4. Search queries are embedded at query time and ranked by similarity score
> > >         
> > >          5. ## API Reference
> > >         
> > >          6. All `/v1/*` routes require: `X-API-Key: <your-api-key>`
> > >         
> > >          7. | Method | Endpoint | Description |
> > > |--------|----------|-------------|
> > > | GET | `/health` | Health check |
> > > | POST | `/v1/agents/register` | Register a new agent |
> > > | GET | `/v1/agents/search?q=...` | Semantic agent search |
> > > | POST | `/v1/agents/heartbeat` | Send agent liveness signal |
> > > | POST | `/v1/agents/cost` | Log LLM cost event |
> > > | GET | `/v1/fleet` | Get full fleet status |
> > >
> > > ## Learn More
> > >
> > > Visit [invexsai.com](https://invexsai.com) for documentation, demos, and the hosted platform.
> > >
> > > ## License
> > >
> > > MIT — see LICENSE for details.
