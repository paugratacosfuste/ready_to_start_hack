# 🚀 Ready to START Hack — Barcelona 2026

Pre-configured hackathon environment with AI skills, ML pipelines, and pitch deck templates.

## Quick Start

```bash
# 1. Clone
git clone git@github.com:paugratacosfuste/ready_to_start_hack.git
cd ready_to_start_hack

# 2. Run setup (installs Python venv + all packages + Node deps)
chmod +x setup.sh
./setup.sh

# 3. Add your API keys
nano .env

# 4. Verify backend works
source .venv/bin/activate
cd backend && python main.py
# → Open http://localhost:8000/docs
```

## For Your Teammates

Everyone on the team runs:
```bash
git clone git@github.com:paugratacosfuste/ready_to_start_hack.git
cd ready_to_start_hack
./setup.sh
# Edit .env with API keys
```

That's it. Environment is identical for everyone.

## What's Inside

```
ready_to_start_hack/
├── setup.sh                 ← One-command setup (run this first!)
├── CLAUDE.md                ← Claude Code auto-loads this as project context
├── .env.example             ← API keys template
├── requirements.txt         ← Python deps (AI, ML, API, data, viz)
├── package.json             ← Node deps (pitch deck generator)
│
├── skills/                  ← 9 Claude Code skill files
│   ├── hackathon-mode.md    ← MVP mindset + decision framework
│   ├── rapid-prototyping.md ← Stack selection + scaffolding (10 min)
│   ├── ai-integration.md   ← LLM patterns (OpenAI, RAG, agents, streaming)
│   ├── ml-pipeline.md      ← ML model patterns + auto-ML guide
│   ├── data-generation.md  ← Synthetic data (N26/fintech focused)
│   ├── debug-fast.md       ← 3AM emergency debugging
│   ├── ui-polish.md        ← Last-hour visual upgrades
│   ├── deploy-and-demo.md  ← Deployment + demo checklist
│   └── pitch-prep.md       ← Winning pitch structure
│
├── templates/
│   ├── ml_pipeline_auto.py       ← Set 2 vars → auto trains + evaluates models
│   ├── ml_pipeline_original.py   ← Detailed manual ML template
│   └── pitch_deck_template.js    ← Edit CONFIG → `npm run pitch-deck`
│
├── backend/
│   └── main.py              ← FastAPI skeleton (CORS, health, AI stubs)
│
├── frontend/                ← Scaffold with Claude Code during hackathon
├── data/                    ← Demo data goes here
└── docs/
    └── ROADMAP.md           ← 24-hour battle plan with checkpoints
```

## During the Hackathon

### With Claude Code:
```bash
source .venv/bin/activate
claude
# → "Read all skills in skills/. We're doing the [Microsoft/N26] case.
#    The problem is [X]. Scaffold the project."
```

### Key Commands:
```bash
# Backend
cd backend && python main.py

# ML Pipeline
python templates/ml_pipeline_auto.py

# Pitch Deck
npm run pitch-deck

# Frontend (after scaffolding)
cd frontend && npm run dev
```

## Case Partners
- **Microsoft** — AI, cloud, enterprise challenges
- **N26 Bank** — Fintech, digital banking, personal finance

## Pre-installed Python Packages
AI: openai, anthropic, langchain, chromadb
ML: scikit-learn, xgboost, pandas, numpy
Viz: matplotlib, plotly, seaborn
API: fastapi, uvicorn, pydantic
Data: faker, beautifulsoup4, pypdf
