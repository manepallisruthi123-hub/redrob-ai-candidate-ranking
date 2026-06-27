# Intelligent Candidate Discovery
### INDIA RUNS Hackathon · Data & AI Challenge
**Team:** Tech Titans &nbsp;|&nbsp; **Team Leader:** Manepalli Sruthi

---

## Problem Statement

Read a job description and actually understand what the role needs — not just pull out words. Look at the full picture — career history, skills, behavioral signals, platform activity — and figure out who genuinely fits. Deliver a shortlist that a recruiter can trust.

---

## Our Approach

Most ATS tools score candidates on keyword overlap. We built a **semantic-first ranking pipeline** that computes true meaning-based similarity between the JD and every candidate's profile — paired with verified skill depth and real platform behavior signals.

### Key design decision
An early version of our pipeline ran semantic scoring only on the top 1,000 candidates pre-filtered by keyword skill score. This silently excluded strong candidates who described their experience differently — *"search infrastructure"* instead of *"vector search"*, for example. The final version batch-encodes **the entire surviving pool** using sentence-transformers, so semantic fit is computed for every candidate before anyone is ranked out.

---

## Pipeline Overview

```
candidates.jsonl
      │
      ▼
┌─────────────────────────────────────┐
│  Stage 1: Stream & Clean            │
│  • Honeypot detection               │
│  • IT service-giant filter          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 2: Score (full pool)         │
│  • Verified skill competency        │
│  • Batched semantic JD similarity   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 3: Normalize & Blend         │
│  • Min-max both scores              │
│  • 0.4 × skill + 0.6 × semantic    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 4: Behavioral Multiplier     │
│  • Recency, response rate,          │
│    interview completion, GitHub     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Stage 5: Rank & Explain            │
│  • Top 100, score descending        │
│  • Per-candidate reasoning string   │
│  • submission.csv                   │
└─────────────────────────────────────┘
```

---

## Scoring Formula

```
final score = ( 0.4 × skill_score + 0.6 × semantic_score ) × behavioral_multiplier
```

Both `skill_score` and `semantic_score` are **min-max normalized across the full candidate pool** before blending — so the 0.4 / 0.6 weights in the code are the weights that actually drive the ranking, not just labels.

---

## Stage Details

### Stage 1 — Data Integrity

| Filter | Logic |
|---|---|
| **Honeypot detection** | Removes any profile where a claimed skill duration exceeds total years of experience (+ 2 month tolerance) |
| **IT service-giant filter** | Removes candidates whose *entire* career history is at outsourcing firms (TCS, Infosys, Wipro, Cognizant, Accenture, Tech Mahindra, Mindtree). Mixed backgrounds pass. |

### Stage 2 — Skill Competency Score

Matched against 15 core signals for the role:

`NLP` · `Embeddings` · `Vector Search` · `Fine-Tuning LLMs` · `MLOps` · `Information Retrieval` · `Dense Retrieval` · `Hybrid Search` · `Pinecone` · `Weaviate` · `Qdrant` · `Milvus` · `FAISS` · `OpenSearch` · `Elasticsearch`

Each matched skill is scored on:
- **Base:** proficiency tier (beginner → expert), 1–4
- **Trust:** tenure (months) + peer endorsements
- **Verify:** blended 60/40 with official platform assessment score where available
- **Depth bonus:** ×1.25 when 3+ core skills matched

### Stage 2b — Behavioral Multiplier

| Signal | Modifier |
|---|---|
| Open to work flag | ×1.15 |
| Active within 30 days | ×1.10 |
| Inactive 90–180 days | ×0.75 |
| Inactive 180+ days | ×0.50 |
| Recruiter response rate ≥ 85% | ×1.10 |
| Recruiter response rate < 40% | ×0.70 |
| Interview completion ≥ 90% | ×1.05 |
| Interview completion < 60% | ×0.80 |
| GitHub activity score ≥ 75 | ×1.15 |

Multipliers stack — a recently active, highly responsive candidate compounds both boosts.

### Stage 3 — Semantic Similarity

Model: **`all-MiniLM-L6-v2`** (sentence-transformers)

- JD encoded once into a reusable vector
- All candidate profile narratives batch-encoded (batch size 128)
- Cosine similarity computed between JD vector and each candidate vector
- No keyword pre-filtering — every surviving candidate is semantically evaluated

### Stage 5 — Explainability

Every ranked candidate ships with a reasoning string built only from that candidate's real numbers — semantic score, response rate, interview completion, and recency — not a templated compliment.

Example:
> *"AI Engineer with 16.9 years of history showing a strong 0.81 semantic fit for the JD. Demonstrates moderate recruiter engagement (72%), excellent interview attendance history (91%), and a passive candidate footprint."*

---

## Repository Structure

```
redrob-ai-candidate-ranking/
├── rank_candidates.py       # Full pipeline — run this
├── validate_submission.py   # Format & ranking validator
├── submission.csv           # Final ranked output (top 100)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Setup & Usage

```bash
pip install -r requirements.txt
```

Place `candidates.jsonl` in the same directory, then:

```bash
python3 rank_candidates.py
```

This writes `submission.csv`. To validate the output format independently:

```bash
python3 validate_submission.py submission.csv
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Streaming JSONL pipeline |
| sentence-transformers (`all-MiniLM-L6-v2`) | Batched semantic embeddings — CPU-friendly, no GPU needed |
| NumPy | Vectorized normalization and score blending |
| csv / json | Streaming ingestion and output |

---

## Submission

| Asset | Status |
|---|---|
| `rank_candidates.py` | ✅ Complete |
| `submission.csv` | ✅ 100 rows, self-validated |
| `validate_submission.py` | ✅ Passes 100/100 |
| Deck (PDF) | ✅ Submitted |
