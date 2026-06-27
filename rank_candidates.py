"""
Intelligent Candidate Discovery — Redrob AI Hackathon
Team: Tech Titans | Team Leader: Manepalli Sruthi

Semantic-first candidate ranking pipeline:
  1. Stream & clean raw candidate pool (honeypot + IT-giant filtering)
  2. Compute verified skill competency score
  3. Batch-encode full pool with sentence-transformer embeddings
  4. Normalize both scores, blend 0.4 skill / 0.6 semantic
  5. Apply behavioral reachability multiplier
  6. Output ranked top 100 with traceable per-candidate reasoning
"""

import json
import os
import csv
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_FILE = "candidates.jsonl"
OUTPUT_FILE = "submission.csv"


# ── Stage 1: Stream & filter ──────────────────────────────────────────────────
def stream_and_filter_candidates(file_path):
    """
    Streams candidates line-by-line from a JSONL file.
    Removes two trap patterns before any scoring happens:
      - Honeypot profiles: skill duration > total career experience
      - IT-outsourcing-only histories: entire career at service giants
    """
    service_giants = {
        "tcs", "infosys", "wipro", "cognizant", "accenture",
        "tech mahindra", "mindtree"
    }
    passed_candidates = []
    honeypot_count = 0
    service_giant_count = 0
    total_processed = 0

    print(f"🚀 Streaming candidate pool from: {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            total_processed += 1
            candidate = json.loads(line.strip())

            profile = candidate.get("profile", {})
            career_history = candidate.get("career_history", [])
            skills = candidate.get("skills", [])

            # Trap 1: Honeypot detection
            years_exp = profile.get("years_of_experience", 0)
            allowed_months = years_exp * 12
            is_honeypot = any(
                skill.get("duration_months", 0) > (allowed_months + 2)
                for skill in skills
            )
            if is_honeypot:
                honeypot_count += 1
                continue

            # Trap 2: IT service-giant filter (only when ALL companies match)
            companies = [str(job.get("company", "")).lower() for job in career_history]
            if companies and all(
                any(giant in co for giant in service_giants) for co in companies
            ):
                service_giant_count += 1
                continue

            passed_candidates.append(candidate)

    print(f"\n✅ Streaming complete")
    print(f"   Total streamed       : {total_processed}")
    print(f"   Honeypots removed    : {honeypot_count}")
    print(f"   IT-giants excluded   : {service_giant_count}")
    print(f"   Passed to scoring    : {len(passed_candidates)}")
    return passed_candidates


# ── Stage 2: Skill competency score ──────────────────────────────────────────
def calculate_skill_score_final(candidate):
    """
    Verified technical competency score across 15 core NLP/ML/search skills.
    Blends proficiency tier, tenure, endorsements, and official assessments.
    Applies x1.25 depth bonus when 3+ core skills are matched.
    """
    target_skills = {
        "nlp", "embeddings", "vector search", "fine-tuning llms",
        "mlops", "information retrieval", "dense retrieval", "hybrid search",
        "pinecone", "weaviate", "qdrant", "milvus", "faiss",
        "opensearch", "elasticsearch"
    }
    proficiency_map = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}

    signals = candidate.get("redrob_signals", {})
    raw_assessments = signals.get("skill_assessment_scores", {})
    assessments = {k.lower(): v for k, v in raw_assessments.items()}

    total_weight = 0.0
    matched_count = 0

    for skill in candidate.get("skills", []):
        name = skill.get("name", "").lower()
        if name not in target_skills:
            continue

        matched_count += 1
        base = proficiency_map.get(skill.get("proficiency", "beginner"), 1)
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)
        trust = (duration * 0.1) + (endorsements * 0.05)
        entry_score = base * (1 + trust)

        if name in assessments:
            entry_score = (entry_score * 0.4) + (assessments[name] * 0.6)

        total_weight += entry_score

    if matched_count >= 3:
        total_weight *= 1.25

    return total_weight


# ── Stage 2b: Behavioral multiplier ──────────────────────────────────────────
def calculate_behavioral_multiplier(candidate):
    """
    Multiplicative modifier from real platform-activity signals.
    Rewards reachability; penalises stale or unresponsive profiles.
    """
    signals = candidate.get("redrob_signals", {})
    multiplier = 1.0

    if signals.get("open_to_work_flag", False):
        multiplier *= 1.15

    last_active_str = signals.get("last_active_date", "")
    if last_active_str:
        try:
            days_inactive = (
                datetime(2026, 6, 19)
                - datetime.strptime(last_active_str, "%Y-%m-%d")
            ).days
            if days_inactive > 180:
                multiplier *= 0.50
            elif days_inactive > 90:
                multiplier *= 0.75
            elif days_inactive <= 30:
                multiplier *= 1.10
        except ValueError:
            pass

    response_rate = signals.get("recruiter_response_rate", 0.0)
    if response_rate >= 0.85:
        multiplier *= 1.10
    elif response_rate < 0.40:
        multiplier *= 0.70

    completion_rate = signals.get("interview_completion_rate", 0.0)
    if completion_rate >= 0.90:
        multiplier *= 1.05
    elif completion_rate < 0.60:
        multiplier *= 0.80

    github_score = signals.get("github_activity_score", -1)
    if github_score >= 75:
        multiplier *= 1.15
    elif github_score == -1 or github_score < 10:
        multiplier *= 0.95

    return multiplier


def min_max_normalize(vector):
    """Min-max scale a list of floats to [0, 1]."""
    arr = np.array(vector, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ── Stage 3–5: Semantic pipeline ──────────────────────────────────────────────
def run_production_semantic_pipeline(passed_candidates, output_filename="submission.csv"):
    """
    Full-pool semantic ranking:
      - Batch-encodes every surviving candidate (all-MiniLM-L6-v2)
      - Normalizes skill + semantic scores independently (min-max)
      - Blends: 0.4 x skill + 0.6 x semantic
      - Multiplies by behavioral reachability modifier
      - Sorts top 100, writes submission.csv with per-candidate reasoning
    """
    print("📦 Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    jd_text = (
        "Senior AI Engineer Founding Team. Deep technical depth in modern ML systems, "
        "embeddings, retrieval, ranking, LLMs, and fine-tuning. Scrappy product-engineering "
        "attitude comfortable building production RAG pipelines, search architectures, "
        "vector indexes, and evaluation frameworks from scratch in a fast-paced startup."
    )
    jd_vector = model.encode(jd_text, convert_to_numpy=True)

    print(f"⏳ Scoring all {len(passed_candidates)} candidates...")
    narratives = []
    raw_skill_scores = []

    for cand in passed_candidates:
        raw_skill_scores.append(calculate_skill_score_final(cand))
        prof = cand.get("profile", {})
        narratives.append(
            f"Title: {prof.get('current_title', '')}. "
            f"Headline: {prof.get('headline', '')}. "
            f"Summary: {prof.get('summary', '')}"
        )

    print("🔢 Batch-encoding full candidate pool...")
    embeddings = model.encode(
        narratives, batch_size=128, show_progress_bar=True, convert_to_numpy=True
    )

    print("🎯 Computing cosine similarities against JD...")
    norm_jd = np.linalg.norm(jd_vector)
    norm_cands = np.linalg.norm(embeddings, axis=1)
    norm_cands[norm_cands == 0] = 1.0
    raw_semantic = np.dot(embeddings, jd_vector) / (norm_jd * norm_cands)

    print("⚖️  Normalizing and blending scores...")
    norm_skills = min_max_normalize(raw_skill_scores)
    norm_semantic = min_max_normalize(raw_semantic)

    final_pool = []
    for idx, cand in enumerate(passed_candidates):
        composite = (norm_skills[idx] * 0.4) + (norm_semantic[idx] * 0.6)
        behavioral = calculate_behavioral_multiplier(cand)
        final_score = round(float(composite * behavioral), 4)
        final_pool.append({
            "candidate_id": cand.get("candidate_id"),
            "score": final_score,
            "raw_semantic": round(float(raw_semantic[idx]), 2),
            "profile": cand.get("profile", {}),
            "signals": cand.get("redrob_signals", {}),
        })

    # Sort: score descending, candidate_id ascending for ties
    final_pool.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top_100 = final_pool[:100]

    print(f"📝 Writing top {len(top_100)} candidates to '{output_filename}'...")
    with open(output_filename, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()

        for rank, item in enumerate(top_100, start=1):
            prof = item["profile"]
            sigs = item["signals"]
            title = prof.get("current_title", "Engineer")
            exp = prof.get("years_of_experience", 0)
            sem = item["raw_semantic"]

            resp = int(sigs.get("recruiter_response_rate", 0.0) * 100)
            resp_label = (
                "highly responsive to recruiter outreach" if resp >= 85
                else "limited recruiter responsiveness" if resp < 40
                else "moderate recruiter engagement"
            )

            comp = int(sigs.get("interview_completion_rate", 0.0) * 100)
            comp_label = (
                "excellent interview attendance history" if comp >= 90
                else "unreliable interview completion" if comp < 60
                else "stable interview consistency"
            )

            try:
                days_ago = (
                    datetime(2026, 6, 19)
                    - datetime.strptime(sigs.get("last_active_date", ""), "%Y-%m-%d")
                ).days
                active_ctx = (
                    f"an active footprint (last seen {days_ago}d ago)"
                    if days_ago <= 30
                    else "a passive candidate footprint"
                )
            except Exception:
                active_ctx = "a tracked platform footprint"

            reasoning = (
                f"{title} with {exp} years of history showing a strong {sem} semantic fit "
                f"for the JD. Demonstrates {resp_label} ({resp}%), {comp_label} ({comp}%), "
                f"and {active_ctx}."
            )

            writer.writerow({
                "candidate_id": item["candidate_id"],
                "rank": rank,
                "score": item["score"],
                "reasoning": reasoning,
            })

    print(f"🏁 Done — submission written to '{output_filename}'")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if os.path.exists(TARGET_FILE):
        clean_pool = stream_and_filter_candidates(TARGET_FILE)
        run_production_semantic_pipeline(clean_pool, output_filename=OUTPUT_FILE)
    else:
        print(f"⚠️  Dataset '{TARGET_FILE}' not found in working directory.")
        print("   Place candidates.jsonl alongside this script and re-run.")
