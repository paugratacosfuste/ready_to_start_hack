#!/bin/bash
# ============================================
# 🚀 HACKATHON ENVIRONMENT SETUP
# Run: chmod +x setup.sh && ./setup.sh
# ============================================

set -e  # Exit on error

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  🚀 START Hack — Environment Setup           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Step 1: Check prerequisites ──────────────────

echo -e "${BOLD}[1/6] Checking prerequisites...${NC}"

# Check Python
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "  ${GREEN}✅ Python $PY_VERSION${NC}"
else
    echo -e "  ${RED}❌ Python 3 not found. Install: brew install python3${NC}"
    exit 1
fi

# Check Node
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "  ${GREEN}✅ Node $NODE_VERSION${NC}"
else
    echo -e "  ${RED}❌ Node.js not found. Install: brew install node${NC}"
    exit 1
fi

# Check npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "  ${GREEN}✅ npm $NPM_VERSION${NC}"
else
    echo -e "  ${RED}❌ npm not found${NC}"
    exit 1
fi

# Check git
if command -v git &> /dev/null; then
    echo -e "  ${GREEN}✅ git$(git --version | awk '{print " "$3}')${NC}"
else
    echo -e "  ${RED}❌ git not found${NC}"
    exit 1
fi

# ── Step 2: Python virtual environment ───────────

echo ""
echo -e "${BOLD}[2/6] Setting up Python virtual environment...${NC}"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "  ${GREEN}✅ Created .venv${NC}"
else
    echo -e "  ${YELLOW}⏭️  .venv already exists, skipping${NC}"
fi

source .venv/bin/activate
echo -e "  ${GREEN}✅ Activated .venv${NC}"

# ── Step 3: Install Python dependencies ──────────

echo ""
echo -e "${BOLD}[3/6] Installing Python packages...${NC}"
echo -e "  ${YELLOW}(This may take 2-3 minutes)${NC}"

pip install --upgrade pip -q
pip install -r requirements.txt -q

echo -e "  ${GREEN}✅ Python packages installed${NC}"

# Verify key packages
echo ""
echo -e "  Verifying key packages:"
python3 -c "import openai; print(f'    ✅ openai {openai.__version__}')" 2>/dev/null || echo -e "    ${RED}❌ openai${NC}"
python3 -c "import anthropic; print(f'    ✅ anthropic {anthropic.__version__}')" 2>/dev/null || echo -e "    ${RED}❌ anthropic${NC}"
python3 -c "import fastapi; print(f'    ✅ fastapi {fastapi.__version__}')" 2>/dev/null || echo -e "    ${RED}❌ fastapi${NC}"
python3 -c "import sklearn; print(f'    ✅ scikit-learn {sklearn.__version__}')" 2>/dev/null || echo -e "    ${RED}❌ scikit-learn${NC}"
python3 -c "import pandas; print(f'    ✅ pandas {pandas.__version__}')" 2>/dev/null || echo -e "    ${RED}❌ pandas${NC}"
python3 -c "import xgboost; print(f'    ✅ xgboost {xgboost.__version__}')" 2>/dev/null || echo -e "    ${RED}❌ xgboost${NC}"
python3 -c "import langchain; print(f'    ✅ langchain {langchain.__version__}')" 2>/dev/null || echo -e "    ${RED}❌ langchain${NC}"
python3 -c "import chromadb; print(f'    ✅ chromadb {chromadb.__version__}')" 2>/dev/null || echo -e "    ${RED}❌ chromadb${NC}"
python3 -c "import plotly; print(f'    ✅ plotly {plotly.__version__}')" 2>/dev/null || echo -e "    ${RED}❌ plotly${NC}"
python3 -c "import faker; print(f'    ✅ faker {faker.VERSION}')" 2>/dev/null || echo -e "    ${RED}❌ faker${NC}"

# ── Step 4: Install Node dependencies ────────────

echo ""
echo -e "${BOLD}[4/6] Installing Node packages...${NC}"

npm install -q 2>/dev/null
echo -e "  ${GREEN}✅ Root Node packages installed (pptxgenjs for pitch deck)${NC}"

# ── Step 5: Setup .env ───────────────────────────

echo ""
echo -e "${BOLD}[5/6] Setting up environment variables...${NC}"

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "  ${GREEN}✅ Created .env from .env.example${NC}"
    echo -e "  ${YELLOW}⚠️  IMPORTANT: Edit .env and add your API keys!${NC}"
else
    echo -e "  ${YELLOW}⏭️  .env already exists, skipping${NC}"
fi

# ── Step 6: Verify everything works ──────────────

echo ""
echo -e "${BOLD}[6/6] Running smoke tests...${NC}"

# Test backend can start
python3 -c "
from fastapi import FastAPI
from dotenv import load_dotenv
print('    ✅ FastAPI backend imports OK')
" 2>/dev/null || echo -e "    ${RED}❌ FastAPI backend failed${NC}"

# Test ML pipeline
python3 -c "
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
print('    ✅ ML pipeline imports OK')
" 2>/dev/null || echo -e "    ${RED}❌ ML pipeline failed${NC}"

# Test AI imports
python3 -c "
from openai import OpenAI
from anthropic import Anthropic
print('    ✅ AI client imports OK')
" 2>/dev/null || echo -e "    ${RED}❌ AI imports failed${NC}"

# Test pitch deck generation
node -e "const p = require('pptxgenjs'); console.log('    ✅ pptxgenjs OK');" 2>/dev/null || echo -e "    ${RED}❌ pptxgenjs failed${NC}"

# ── Done ─────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ${GREEN}✅ Setup complete!${NC}                            ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo ""
echo "  1. Add your API keys:"
echo -e "     ${YELLOW}nano .env${NC}"
echo ""
echo "  2. Test the backend:"
echo -e "     ${YELLOW}source .venv/bin/activate${NC}"
echo -e "     ${YELLOW}cd backend && python main.py${NC}"
echo -e "     → Open http://localhost:8000/docs"
echo ""
echo "  3. Generate a pitch deck:"
echo -e "     ${YELLOW}npm run pitch-deck${NC}"
echo ""
echo "  4. Tomorrow, scaffold your project with Claude Code:"
echo -e "     ${YELLOW}claude${NC}"
echo -e "     → \"Read all skills in skills/. We're doing the [case]. Scaffold the project.\""
echo ""
echo -e "${BOLD}Repo structure:${NC}"
echo "  skills/       → Claude Code skill files (9 skills)"
echo "  templates/    → ML pipeline + pitch deck templates"
echo "  backend/      → FastAPI skeleton (ready to extend)"
echo "  frontend/     → (scaffold with Claude Code tomorrow)"
echo "  data/         → Demo data goes here"
echo "  docs/         → 24-hour roadmap"
echo ""
echo -e "  ${GREEN}Good luck tomorrow! 🚀${NC}"
echo ""
