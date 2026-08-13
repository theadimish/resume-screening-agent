# Resume Screening Agent

An AI agent that ranks a folder of resumes against a job description and
outputs a scored, explained shortlist — built for the Rooman AI Challenge
(24-Hour AI Agent Challenge).

> **My agent takes** a job description + a folder of resumes **and produces**
> a ranked, scored shortlist with a written reason for every candidate's score.

---

## What it does

1. Parses resumes in **PDF, DOCX, or TXT** format and extracts skills,
   education, and years of experience.
2. Computes a **relevance score** between each resume and the job
   description using **TF-IDF vectorization + cosine similarity** (NLP
   similarity method), blended with an explicit required-skill overlap
   check.
3. Outputs a **ranked, ordered list** (CSV or JSON) with a plain-English
   reasoning sentence per candidate — what matched, what's missing, years
   of experience, and education.
4. Handles **10+ resumes in a single run** (tested with 5 sample resumes
   in this repo; scales linearly with resume count).

---

## 1. Install

```bash
git clone <your-repo-url>
cd resume-screening-agent
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No API key is required — the agent runs 100% offline using scikit-learn's
TF-IDF + cosine similarity, so setup is foolproof.

---

## 2. Run it

```bash
python main.py --jd sample_data/job_description.txt \
                --resumes sample_data/resumes \
                --output output/ranked_output.csv
```

Options:

| Flag         | Required | Description                                             |
|--------------|----------|-----------------------------------------------------------|
| `--jd`       | yes      | Path to the job description (`.txt`, `.pdf`, or `.docx`) |
| `--resumes`  | yes      | Path to a folder of resumes (`.txt`, `.pdf`, or `.docx`) |
| `--output`   | no       | Output path — `.csv` (default) or `.json`                |
| `--top`      | no       | Only output the top N candidates                         |

Example with your own data:

```bash
python main.py --jd my_job.pdf --resumes ./candidates --output output/shortlist.json --top 5
```

The script also prints a ranked summary table straight to the terminal.

---

## 3. Sample inputs/outputs included in this repo

- `sample_data/job_description.txt` — a Backend Python Developer (ML focus) JD
- `sample_data/resumes/` — 5 sample resumes with varying levels of fit
  (2 strong Python/ML matches, 1 partial-fit full-stack dev, 1 ML intern,
  1 unrelated marketing resume as a negative-case test)
- `output/ranked_output.csv` and `output/ranked_output.json` — the agent's
  actual output when run against the sample data above

Sample run result (see `output/ranked_output.csv` for full detail):

| Rank | Name         | Score | Why                                              |
|------|--------------|-------|---------------------------------------------------|
| 1    | Priya Sharma | 44.0  | 14/15 required skills matched, 3 yrs experience   |
| 2    | Arjun Mehta  | 40.7  | 12/15 required skills matched, 4 yrs experience   |
| 3    | Karan Singh  | 31.2  | 8/15 matched, strong ML intern background         |
| 4    | Sneha Kapoor | 11.4  | Mostly frontend skills, minimal overlap           |
| 5    | Rahul Verma  | 1.3   | Marketing background, essentially no overlap      |

---

## Design choices & scoring method

**Similarity method:** TF-IDF (unigrams + bigrams) fit jointly over the JD
and all resumes, then cosine similarity between the JD vector and each
resume vector. TF-IDF was chosen over an LLM call for the core scoring
step because it is:

- **Deterministic** — identical input always gives the identical score,
  which matters for a fair, auditable screening process.
- **Free and offline** — no API key, no network dependency, no per-run
  cost; setup stays foolproof for anyone cloning the repo.
- **Explainable** — cosine similarity + explicit skill overlap makes it
  easy to say *why* a resume scored the way it did (see `reasoning` field).

**Final score formula:**
```
score = 0.70 × (TF-IDF cosine similarity × 100)
      + 0.30 × (matched required-skills / total required-skills × 100)
```
The 70/30 split rewards holistic content similarity while still giving
real weight to hitting the JD's explicit required skills, so a resume
can't game the score with generic filler text.

**Field extraction:** Skills/education/experience are extracted with a
curated keyword vocabulary + regex (see `resume_parser.py`). This keeps
the agent fast and dependency-light. It will miss skills phrased in
unusual ways (e.g. "Postgres" vs "PostgreSQL" if not in the vocab) —
easy to fix by extending `SKILL_VOCAB` in `resume_parser.py`.

---

## Tradeoffs & what I'd improve with more time

- **Skill vocabulary is a fixed list.** A production version would use
  a proper NER/skills-extraction model (or an LLM call) instead of
  keyword matching, so it generalizes to skills not in the hardcoded list.
- **No semantic embeddings.** TF-IDF is purely lexical — it won't know
  that "Postgres" and "PostgreSQL" are the same thing, or that "led a
  team" implies leadership skill. Swapping in sentence-embedding
  similarity (e.g. `sentence-transformers`) would catch more of this,
  at the cost of needing a downloaded model and more setup complexity.
- **Name extraction is a heuristic** (first short line without digits).
  Works for the sample resumes but could misfire on resumes with a
  header/logo line before the name.
- **Optional LLM layer:** the codebase is structured so an LLM call could
  be added inside `scorer.py`'s `_build_reasoning()` to generate a more
  natural write-up per candidate, without touching the core ranking logic.
- With more time I would add: batch PDF OCR fallback for scanned resumes,
  a small web UI, and unit tests for the parser's regex patterns.

---

## Project structure

```
resume-screening-agent/
├── main.py                  # CLI entry point
├── resume_parser.py         # File reading + skill/education extraction
├── scorer.py                 # TF-IDF similarity + scoring logic
├── requirements.txt
├── sample_data/
│   ├── job_description.txt
│   └── resumes/              # 5 sample resumes (.txt)
├── output/
│   ├── ranked_output.csv     # Generated sample output
│   └── ranked_output.json    # Generated sample output
└── README.md
```

---

## Ground rules compliance

- Built individually; design decisions and integration are my own.
- Runnable end-to-end via the commands in "Run it" above — a stranger
  can clone this repo and reproduce the exact sample output.
- CLI only (no UI) — per the brief, a clean CLI that clearly demonstrates
  the agent is sufficient.

## Notes

Tested successfully on Windows with Python 3.9 — ran end-to-end without issues.
