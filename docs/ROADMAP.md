# 🗺️ 24-Hour Hackathon Roadmap — 4-Person Vibe-Coding Team

> **Philosophy:** Ship a working demo + killer pitch. Nothing else matters.
> **Tool stack:** Claude Code (builds) + Claude.ai (strategy/pitch) + Lovable/v0 (UI if needed)

---

## Team Roles (Assign immediately)

| Role | Person | Responsibility |
|------|--------|---------------|
| **Tech Lead** | _____ | Architecture decisions, Claude Code driver, integration |
| **Builder 2** | _____ | Second Claude Code instance, parallel feature development |
| **Data / ML** | _____ | Data pipeline, model training, API endpoints |
| **Pitch Lead** | _____ | Pitch deck, demo script, user story, design polish |

> **Rule:** Pitch Lead codes for the first 16 hours, then switches to pitch-only at Hour 16.

---

## Phase 0: Case Drop (Hour 0 — 30 min)

**Everyone together. No coding yet.**

- [ ] Read the case brief 2-3 times as a team
- [ ] Identify: What is the CORE PROBLEM? (one sentence)
- [ ] Identify: Who is the USER? (one persona)
- [ ] Identify: What would a DEMO look like? (describe the "wow moment")
- [ ] Decide: Which case partner (Microsoft or N26)?
- [ ] Check: What APIs/tools/data do case partners provide?

**Decision checkpoint:** Can you describe your solution in one sentence?
> Template: "We build [WHAT] that helps [WHO] to [BENEFIT] by [HOW]."

---

## Phase 1: Architecture & Setup (Hour 0.5–1.5 — 1 hour)

**Tech Lead decides stack. Everyone sets up.**

- [ ] Choose stack (use skill file `rapid-prototyping.md` with Claude Code)
- [ ] Initialize repo (one person) → share via GitHub
- [ ] Set up project structure with Claude Code
- [ ] Define 3-4 API endpoints / data flows on paper or whiteboard
- [ ] Assign parallel workstreams:
  - **Tech Lead:** Backend + API skeleton
  - **Builder 2:** Frontend scaffold
  - **Data/ML:** Data exploration + pipeline setup
  - **Pitch Lead:** Start wireframes + user flow

**Decision checkpoint:** Everyone knows what they're building for the next 4 hours.

---

## Phase 2: Core Build (Hour 1.5–10 — 8.5 hours)

**Heads down. Parallel work. Sync every 2 hours.**

### Sync Points (5 min max each):
- **Hour 4:** "Does the basic flow work end-to-end?" → If no, simplify.
- **Hour 6:** "Can we show SOMETHING to a person?" → Minimum viable demo.
- **Hour 8:** "What features do we CUT?" → Ruthless scoping.

### Priority Order (build in this sequence):
1. **Data in → Model/Logic → Result out** (the core loop)
2. **One screen that shows the result** (ugly is fine)
3. **Connect frontend to backend** (end-to-end flow)
4. **Second feature only if core works perfectly**

### Anti-patterns to avoid:
- ❌ Spending >1 hour on auth/login (fake it or skip it)
- ❌ Building an admin panel
- ❌ Perfecting CSS before logic works
- ❌ Using a database when a JSON file works
- ❌ Building features nobody will see in the demo

---

## Phase 3: Integration & Polish (Hour 10–16 — 6 hours)

**Everything connects. Demo flow takes shape.**

- [ ] All components talking to each other
- [ ] Happy path works perfectly (THE demo path)
- [ ] Add 2-3 "wow" touches:
  - Real-time updates / streaming responses
  - Nice data visualization
  - Smooth transitions / loading states
- [ ] Hardcode demo data if real data is flaky
- [ ] Test the exact demo flow 3+ times

### Hour 12 checkpoint: Feature freeze discussion
> "If we stopped coding right now, could we demo?"
> If NO → drop features and make core work.
> If YES → polish what you have.

---

## Phase 4: Pitch Prep (Hour 16–20 — 4 hours)

**Pitch Lead goes full-time on pitch. Others polish + support.**

### Pitch Lead:
- [ ] Write pitch script (see pitch template)
- [ ] Build slides using pitch deck template
- [ ] Create demo flow script (exact clicks, exact data)
- [ ] Prepare backup: screenshots/video in case live demo fails

### Tech Team:
- [ ] Fix last bugs in demo flow
- [ ] Ensure deployment works (Vercel/Railway/local)
- [ ] Record backup demo video (screen recording)
- [ ] Prepare "planted questions" — impressive answers for Q&A

---

## Phase 5: Rehearsal & Sleep (Hour 20–22 — 2 hours)

- [ ] **Full dry run #1** — time it (usually 3-5 min pitch)
- [ ] **Fix issues** from dry run
- [ ] **Full dry run #2** — in front of someone outside the team if possible
- [ ] Decide who presents what section
- [ ] **Sleep if possible** — even 1-2 hours helps

---

## Phase 6: Final Demo (Hour 22–24)

- [ ] Test WiFi / projector connection 30 min before
- [ ] Have demo running before you go on stage
- [ ] Keep backup screenshots ready
- [ ] Smile. Speak slowly. Show confidence.
- [ ] End with a clear ASK or VISION statement

---

## 🚨 Emergency Protocols

### "Nothing works at Hour 12"
→ Pivot to a simpler version. Use mock data. Focus on the CONCEPT + pitch.

### "We can't agree on the approach"
→ Tech Lead makes the call. Move on. Disagreement is more expensive than a suboptimal choice.

### "API/data isn't available"
→ Mock it. Generate synthetic data. Judges care about the solution, not the data source.

### "Live demo crashes on stage"
→ Switch to backup video/screenshots. Say "Let me show you what this looks like" — judges understand.

---

## ⏰ Quick Time Reference

| Hour | Phase | Key Question |
|------|-------|-------------|
| 0-0.5 | Case analysis | What's the one-sentence solution? |
| 0.5-1.5 | Setup | Is everyone unblocked? |
| 4 | Build check | Does basic flow work? |
| 8 | Scope check | What do we CUT? |
| 12 | Feature freeze | Can we demo right now? |
| 16 | Pitch mode | Pitch Lead stops coding |
| 20 | Rehearsal | Dry run #1 |
| 22 | Final prep | Backup demo ready? |
| 24 | 🎤 SHOWTIME | |
