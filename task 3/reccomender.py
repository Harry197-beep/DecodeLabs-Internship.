"""
DecodeLabs - Project 3: AI Recommendation Logic
Batch 2026

Capstone: Tech Stack Recommender

Dataset: mikeasilva/data-scientist-skills (raw_skills.csv) - 453 real
job postings scraped from Dice.com, each tagged with a list of skills
extracted from the posting text.

This maps a user's own skills to the most similar job POSTINGS (not
generic role titles) - a more faithful recommendation engine, since
it's matching against real listings rather than a hand-made lookup.

Pipeline (IPO):
  INPUT   -> User provides 3+ skills; postings loaded from raw_skills.csv
  PROCESS -> TF-IDF vectorization + Cosine Similarity scoring
  OUTPUT  -> Sorted, filtered Top-N list of the closest-matching postings

Why TF-IDF instead of simple binary overlap:
Some skills (e.g. "python") appear in nearly every posting, while others
("solr", "spark") are rare and far more distinguishing. TF-IDF
downweights the common ones and upweights the specific ones, so matches
reflect genuine specialization rather than raw overlap count.

Why Cosine Similarity instead of Euclidean distance:
Euclidean distance is sensitive to vector magnitude - a posting with a
long skill list would look "far" from a user profile purely because it
has more words, not because it's actually less relevant. Cosine
similarity measures the angle between vectors, so it only cares about
the *direction* (skill emphasis), not the *length* of the skill list.
"""

import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "raw_skills.csv"
MIN_SKILLS_REQUIRED = 3
TOP_N = 3


def load_job_data(path: str) -> pd.DataFrame:
    """
    Load job postings. The 'raw_skills' column is stored as a
    stringified Python list (e.g. "['python', ' sql']"), so it needs
    ast.literal_eval rather than plain string parsing.
    """
    df = pd.read_csv(path)
    df["skills_list"] = df["raw_skills"].apply(ast.literal_eval)

    # Clean whitespace and dedupe skills within each posting, then
    # join back into a plain string for TF-IDF vectorization.
    df["skills_text"] = df["skills_list"].apply(
        lambda skills: " ".join(sorted(set(s.strip() for s in skills)))
    )
    return df


def get_user_input() -> list[str]:
    """
    Step 1: Ingestion.
    Collect at least MIN_SKILLS_REQUIRED skills/interests from the user.
    """
    print("=== Tech Stack Recommender ===")
    print(f"Enter at least {MIN_SKILLS_REQUIRED} skills or interests, one at a time.")
    print("Type 'done' once you've entered enough.\n")

    skills = []
    while True:
        skill = input(f"Skill #{len(skills) + 1} (or 'done'): ").strip()

        if skill.lower() == "done":
            if len(skills) < MIN_SKILLS_REQUIRED:
                print(f"Need at least {MIN_SKILLS_REQUIRED} skills. Keep going.")
                continue
            break

        if skill:
            skills.append(skill)

    return skills


def build_similarity_scores(user_skills: list[str], job_df: pd.DataFrame):
    """
    Step 2: Scoring.
    Vectorize the user's profile and every posting's skill set into the
    same TF-IDF vocabulary space, then score with cosine similarity.
    """
    user_profile_text = " ".join(user_skills)

    # Shared vocabulary: postings' skill text + the user's profile text.
    corpus = job_df["skills_text"].tolist() + [user_profile_text]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    posting_vectors = tfidf_matrix[:-1]   # all rows except the last
    user_vector = tfidf_matrix[-1]        # the last row is the user profile

    scores = cosine_similarity(user_vector, posting_vectors).flatten()
    return scores


def rank_and_filter(job_df: pd.DataFrame, scores, top_n: int = TOP_N):
    """
    Steps 3 & 4: Sorting and Filtering.
    Attach scores, sort descending, cut to the Top-N list.
    """
    results = job_df.copy()
    results["match_score"] = scores
    results = results.sort_values("match_score", ascending=False)
    return results.head(top_n)


def display_recommendations(top_matches: pd.DataFrame):
    print("\n=== Top Recommended Job Postings ===")
    for rank, (_, row) in enumerate(top_matches.iterrows(), start=1):
        percent = row["match_score"] * 100
        skills_preview = ", ".join(row["skills_list"][:6])
        print(f"{rank}. Posting #{row['id']}  —  {percent:.1f}% match")
        print(f"   Skills: {skills_preview}")
        print(f"   Link: {row['url']}\n")


def main():
    job_df = load_job_data(DATA_PATH)
    print(f"Loaded {len(job_df)} job postings.\n")

    user_skills = get_user_input()

    scores = build_similarity_scores(user_skills, job_df)
    top_matches = rank_and_filter(job_df, scores)

    display_recommendations(top_matches)


if __name__ == "__main__":
    main()
