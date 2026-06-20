# redrob-ai-candidate-ranking
AI-powered candidate ranking system for the Redrob Intelligent Candidate Discovery &amp; Ranking Hackathon. Uses semantic embeddings, behavioral signals, honeypot detection, and hybrid scoring to identify the top 100 candidates for a Senior AI Engineer role.


# 🚀 Redrob AI Founding Engineer Ranker Pipeline

An enterprise-grade, semantic-first talent search pipeline designed to evaluate and rank **100,000 candidates** to discover the optimal Founding AI Systems Engineer. Operating strictly within a **5-minute CPU-only constraints profile**, this repository leverages a vectorized multi-stage alignment strategy that completely isolates data traps and optimizes profile retrieval matches.

---

## 🛠️ Architecture Overview

The system uses a highly optimized data validation layout that guarantees robust evaluation accuracy.



1. **Stage 1: Streaming Trap Isolation** — A memory-efficient `gzip` IO loop streams 100k raw profiles line-by-line under a 50 MB RAM footprint, completely discarding honeypot anomalies and outsourcing services.
2. **Stage 2: Technical Skill Trust Engine** — Computes technical competency across 15 core target technologies, blending duration and endorsements while applying case-insensitive matching against platform-administered testing parameters.
3. **Stage 3: Vectorized Semantic Matching** — Vectorizes combined profile textual fields across the entire candidate pool using a local, batched SentenceTransformer (`all-MiniLM-L6-v2`) to capture context over shallow keywords.
4. **Stage 4: Feature Scaling & Behavioral Weighting** — Normalizes data boundaries using Min-Max scaling before merging metrics, adjusting priority values using multi-layered activity signals.
5. **Stage 5: Compliant Output Generation** — Exports precisely 100 candidates sorted by monotonically non-increasing composite scores, handling ties alphabetically by candidate ID with fully data-tiered recruiter reasoning strings.

---

## 📊 Evaluation Logic & Trap Defusal

* **Honeypot Trap Protection:** Any candidate whose single skill duration surpasses their total career length is automatically dropped, resulting in a **0% Honeypot Rate** in the final shortlist.
* **Mindset Optimization:** Filters out candidates whose entire professional histories are tied to outsourcing firms to prioritize startup execution agility.
* **Feature Equalization:** Uses Min-Max normalization layers to prevent large raw numeric skill scores from drowning out the semantic text vector signals.
* **Granular Justifications:** Reasonings are dynamically tiered based on real metrics, providing descriptive analytics that mirror actual platform engagement levels.

---

## 🚀 Quick Start & Reproducibility

### 1. Environment Installation
Ensure your working space matches the strict reproducible testing dependencies:
```bash
pip install -r requirements.txt
