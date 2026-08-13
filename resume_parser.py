"""
resume_parser.py
-----------------
Handles reading resumes in PDF / DOCX / TXT format and pulling out
lightweight structured fields (skills, education, years of experience)
using keyword and regex heuristics.

This is intentionally dependency-light: no LLM call is required to
extract these fields, which keeps the whole agent runnable offline
with zero API keys. If you DO configure an LLM API key (see README),
scorer.py can optionally use it to generate a natural-language
"reasoning" sentence for each candidate instead of the templated one.
"""

import os
import re
from pathlib import Path

from pypdf import PdfReader
import docx


# ---------------------------------------------------------------------------
# 1. Raw text extraction per file type
# ---------------------------------------------------------------------------

def extract_text(filepath: str) -> str:
    """Return plain text content of a resume file (.pdf, .docx, .txt)."""
    ext = Path(filepath).suffix.lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".pdf":
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == ".docx":
        document = docx.Document(filepath)
        return "\n".join(p.text for p in document.paragraphs)

    raise ValueError(f"Unsupported resume file type: {ext}")


# ---------------------------------------------------------------------------
# 2. Lightweight structured-field extraction
# ---------------------------------------------------------------------------

# A reasonably broad tech/business skills vocabulary. Extend this list
# freely -- matching is case-insensitive and word-boundary aware.
SKILL_VOCAB = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
    "git", "linux", "rest api", "graphql", "microservices",
    "data analysis", "data engineering", "data science", "etl",
    "excel", "power bi", "tableau", "spark", "hadoop", "airflow",
    "project management", "agile", "scrum", "jira",
    "communication", "leadership", "problem solving", "teamwork",
    "html", "css", "sass", "webpack", "figma", "ui/ux",
]

DEGREE_PATTERNS = [
    r"\bph\.?d\.?\b", r"\bm\.?tech\.?\b", r"\bb\.?tech\.?\b",
    r"\bm\.?sc\.?\b", r"\bb\.?sc\.?\b", r"\bmba\b", r"\bmca\b", r"\bbca\b",
    r"\bbachelor(?:'s)?\b", r"\bmaster(?:'s)?\b", r"\bdiploma\b",
]

YEARS_EXP_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s*(?:of)?\s*experience",
    re.IGNORECASE,
)


def extract_skills(text: str) -> list:
    text_lower = text.lower()
    found = []
    for skill in SKILL_VOCAB:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def extract_education(text: str) -> list:
    text_lower = text.lower()
    found = []
    for pattern in DEGREE_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            found.append(m.group(0).upper().replace(".", ""))
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for item in found:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def extract_years_experience(text: str) -> float:
    matches = YEARS_EXP_PATTERN.findall(text)
    if matches:
        return max(float(m) for m in matches)
    return 0.0


def extract_name(text: str, fallback: str) -> str:
    """Best-effort: assume the candidate's name is the first non-empty
    line of the resume (very common convention). Falls back to filename."""
    for line in text.splitlines():
        line = line.strip()
        if line and len(line.split()) <= 5 and not any(ch.isdigit() for ch in line):
            return line
    return fallback


def parse_resume(filepath: str) -> dict:
    """Parse a single resume file into a structured dict."""
    text = extract_text(filepath)
    return {
        "file": os.path.basename(filepath),
        "name": extract_name(text, fallback=Path(filepath).stem),
        "raw_text": text,
        "skills": extract_skills(text),
        "education": extract_education(text),
        "years_experience": extract_years_experience(text),
    }


def load_resumes_from_dir(resumes_dir: str) -> list:
    """Parse every supported resume file in a directory."""
    supported_ext = {".pdf", ".docx", ".txt"}
    parsed = []
    for fname in sorted(os.listdir(resumes_dir)):
        fpath = os.path.join(resumes_dir, fname)
        if Path(fname).suffix.lower() in supported_ext and os.path.isfile(fpath):
            try:
                parsed.append(parse_resume(fpath))
            except Exception as e:
                print(f"  [warn] Skipping '{fname}' — could not parse ({e})")
    return parsed
