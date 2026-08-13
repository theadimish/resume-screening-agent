"""
scorer.py
---------
Computes a relevance score for each resume against a job description
using TF-IDF vectorization + cosine similarity (a standard NLP
similarity method), then blends in a small bonus for explicit skill
overlap so the ranking rewards resumes that literally mention the
JD's required skills, not just generically similar wording.

Design choice: TF-IDF + cosine similarity was chosen over calling an
LLM for scoring because it is:
  1. Deterministic and explainable (same input -> same score, always).
  2. Free to run, with no API key / network dependency (foolproof setup).
  3. A well-established "NLP similarity method" per the brief.
An LLM-based re-ranking step could be layered on top later — see
README "Tradeoffs" section for how.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_parser import extract_skills


def compute_similarity_scores(jd_text: str, resume_texts: list) -> list:
    """Return cosine similarity (0-1) between the JD and each resume,
    using a shared TF-IDF vector space fit across JD + all resumes."""
    corpus = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # capture 2-word skill phrases like "machine learning"
        max_features=5000,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(jd_vector, resume_vectors)[0]
    return similarities.tolist()


def score_candidates(jd_text: str, parsed_resumes: list) -> list:
    """
    Score and rank a list of parsed resume dicts against a job description.

    Final score (0-100) = 70% TF-IDF cosine similarity
                         + 30% explicit required-skill overlap ratio

    Returns the list sorted by score, descending, with 'score',
    'matched_skills', 'missing_skills', and 'reasoning' fields added.
    """
    jd_skills = set(extract_skills(jd_text))
    resume_texts = [r["raw_text"] for r in parsed_resumes]

    similarities = compute_similarity_scores(jd_text, resume_texts)

    results = []
    for resume, sim in zip(parsed_resumes, similarities):
        resume_skills = set(resume["skills"])
        matched = sorted(jd_skills & resume_skills)
        missing = sorted(jd_skills - resume_skills)

        skill_overlap_ratio = (len(matched) / len(jd_skills)) if jd_skills else 0.0

        final_score = (0.70 * sim * 100) + (0.30 * skill_overlap_ratio * 100)
        final_score = round(final_score, 2)

        reasoning = _build_reasoning(resume, sim, matched, missing)

        results.append({
            **resume,
            "score": final_score,
            "text_similarity_pct": round(sim * 100, 2),
            "skill_overlap_pct": round(skill_overlap_ratio * 100, 2),
            "matched_skills": matched,
            "missing_skills": missing,
            "reasoning": reasoning,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    for idx, r in enumerate(results, start=1):
        r["rank"] = idx

    return results


def _build_reasoning(resume, sim, matched, missing) -> str:
    parts = []
    parts.append(
        f"Overall text/content similarity to the JD is {sim * 100:.1f}%."
    )
    if matched:
        parts.append(f"Matches {len(matched)} required skill(s): {', '.join(matched)}.")
    else:
        parts.append("No directly matching required skills were found in the resume text.")
    if missing:
        parts.append(f"Missing: {', '.join(missing)}.")
    if resume.get("years_experience"):
        parts.append(f"States {resume['years_experience']:.1f} years of experience.")
    if resume.get("education"):
        parts.append(f"Education mentions: {', '.join(resume['education'])}.")
    return " ".join(parts)
