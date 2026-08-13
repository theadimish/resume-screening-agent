#!/usr/bin/env python3
"""
main.py
-------
CLI entry point for the Resume Screening Agent.

Usage:
    python main.py --jd sample_data/job_description.txt \
                    --resumes sample_data/resumes \
                    --output output/ranked_output.csv \
                    --top 10

Run `python main.py --help` for all options.
"""

import argparse
import csv
import json
import os
import sys

from resume_parser import extract_text, load_resumes_from_dir
from scorer import score_candidates


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank a folder of resumes against a job description."
    )
    parser.add_argument("--jd", required=True, help="Path to the job description file (.txt/.pdf/.docx)")
    parser.add_argument("--resumes", required=True, help="Path to a folder containing resume files")
    parser.add_argument("--output", default="output/ranked_output.csv", help="Where to write ranked output (.csv or .json)")
    parser.add_argument("--top", type=int, default=None, help="Only show/save the top N candidates")
    return parser.parse_args()


def write_csv(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "rank", "name", "file", "score", "text_similarity_pct",
        "skill_overlap_pct", "years_experience", "education",
        "matched_skills", "missing_skills", "reasoning",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            row["education"] = "; ".join(r.get("education", []))
            row["matched_skills"] = "; ".join(r.get("matched_skills", []))
            row["missing_skills"] = "; ".join(r.get("missing_skills", []))
            writer.writerow(row)


def write_json(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    slim = [
        {k: v for k, v in r.items() if k != "raw_text"}
        for r in results
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2)


def print_summary(results):
    print("\n" + "=" * 78)
    print(f"{'Rank':<5}{'Score':<8}{'Name':<28}{'File'}")
    print("=" * 78)
    for r in results:
        print(f"{r['rank']:<5}{r['score']:<8}{r['name'][:26]:<28}{r['file']}")
    print("=" * 78 + "\n")


def main():
    args = parse_args()

    if not os.path.isfile(args.jd):
        sys.exit(f"Error: job description file not found: {args.jd}")
    if not os.path.isdir(args.resumes):
        sys.exit(f"Error: resumes folder not found: {args.resumes}")

    print(f"Loading job description from: {args.jd}")
    jd_text = extract_text(args.jd)

    print(f"Parsing resumes from: {args.resumes}")
    parsed_resumes = load_resumes_from_dir(args.resumes)
    if not parsed_resumes:
        sys.exit("Error: no supported resume files (.pdf/.docx/.txt) found in that folder.")
    print(f"  Parsed {len(parsed_resumes)} resume(s).")

    print("Scoring candidates against the job description (TF-IDF + cosine similarity)...")
    results = score_candidates(jd_text, parsed_resumes)

    if args.top:
        results = results[: args.top]

    print_summary(results)

    if args.output.lower().endswith(".json"):
        write_json(results, args.output)
    else:
        write_csv(results, args.output)
    print(f"Full ranked results with reasoning written to: {args.output}\n")


if __name__ == "__main__":
    main()
