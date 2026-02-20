# Project: START Hack Barcelona 2026

## Context
This is a 24-hour hackathon project. Case partners: **Microsoft** and **N26 Bank**.
Team: 4 vibe coders using Claude Code + Claude.ai.

## Skills
Before starting ANY task, read the relevant skill files in `skills/`:
- `skills/hackathon-mode.md` — **ALWAYS read first.** MVP mindset, decision framework.
- `skills/rapid-prototyping.md` — Stack selection and scaffolding.
- `skills/ai-integration.md` — LLM API patterns (OpenAI, Anthropic, RAG, agents).
- `skills/ml-pipeline.md` — ML model patterns + auto-ML template usage.
- `skills/data-generation.md` — Synthetic data generators (N26/fintech focused).
- `skills/debug-fast.md` — Emergency debugging patterns.
- `skills/ui-polish.md` — Last-hour visual upgrades.
- `skills/deploy-and-demo.md` — Deployment and demo preparation.
- `skills/pitch-prep.md` — Pitch structure and storytelling.

## Environment
- Python venv at `.venv/` — activate with `source .venv/bin/activate`
- All Python deps in `requirements.txt` (already installed)
- Node deps in `package.json` (pptxgenjs for pitch deck)
- Backend skeleton at `backend/main.py` (FastAPI, CORS pre-configured)
- API keys in `.env` (copy from `.env.example` if missing)

## Key Files
- `templates/ml_pipeline_auto.py` — Auto-ML: set DATA_SOURCE + TARGET, run, done.
- `templates/pitch_deck_template.js` — Edit CONFIG, run `npm run pitch-deck`.
- `docs/ROADMAP.md` — 24-hour battle plan.

## Rules
1. **Ship > Perfect.** Every decision optimizes for "can we demo this?"
2. **Don't over-engineer.** No auth, no CI/CD, no tests, no TypeScript strict mode.
3. **Mock what's hard.** If something takes >30 min and isn't in the demo, fake it.
4. **One happy path.** Build only what you'll show on stage.
5. **Ask before building.** Confirm features are for the demo flow before implementing.
