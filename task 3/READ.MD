# decodelabs-tech-stack-recommender
project 3 submission

# Project 3: AI Recommendation Logic — Tech Stack Recommender
**DecodeLabs Internship — Batch 2026**

## Overview

A content-based recommendation engine that matches a user's skills to
the most similar real job postings, using TF-IDF vectorization and
Cosine Similarity. This is the "personalization phase" project — instead
of classifying data into fixed categories (Project 2), the system now
ranks a large pool of items by relevance to a specific user profile.

## Dataset

**[mikeasilva/data-scientist-skills](https://github.com/mikeasilva/data-scientist-skills)**
— 453 real job postings scraped from Dice.com, each tagged with a list
of skills extracted from the posting text.

| Column | Description |
|---|---|
| `id` | Posting identifier |
| `raw_skills` | Stringified Python list of skills, e.g. `"['python', ' sql']"` |
| `url` | Link to the original job listing |

Since `raw_skills` is stored as text that *looks* like a Python list
(not an actual list), it's parsed with `ast.literal_eval` rather than
plain string splitting. A few postings also repeat their skill list
twice within the same row — these are deduped before vectorizing so
they don't artificially inflate their own term frequency.

## Why Content-Based Filtering

Recommendation systems generally split into two approaches:

- **Collaborative filtering** — driven by community behavior ("users
  who picked this also picked that"). Needs a large history of user
  interactions to work at all.
- **Content-based filtering** — driven by item attributes, mapped
  directly to what the user says they want. Works immediately, with
  no historical data required.

This project uses content-based filtering exclusively, since there's no
user interaction history to draw from — just skills and postings.

## Pipeline (IPO Framework)

### INPUT — Ingestion
```python
skills = []
while True:
    skill = input(f"Skill #{len(skills) + 1} (or 'done'): ")
    ...
```
Collects a minimum of 3 skills or interests from the user before
proceeding — enough data density for the similarity math to be
meaningful.

### PROCESS — Vectorization & Scoring
```python
corpus = job_df["skills_text"].tolist() + [user_profile_text]
tfidf_matrix = TfidfVectorizer().fit_transform(corpus)
scores = cosine_similarity(user_vector, posting_vectors)
```

**Why TF-IDF instead of binary overlap ("has this skill: yes/no"):**
Some skills (e.g. `python`) appear in nearly every posting, while
others (`solr`, `spark`) are rare and far more distinguishing. TF-IDF
downweights the common terms and upweights the specific ones, so a
match reflects genuine specialization rather than raw keyword overlap.

**Why Cosine Similarity instead of Euclidean distance:**
Euclidean distance is sensitive to vector magnitude — a posting with a
long skill list would look "far" from a short user profile purely
because it has more words, not because it's actually less relevant.
Cosine similarity measures the *angle* between vectors, so it only
cares about the direction (skill emphasis), not the length of the list.

Crucially, the user's profile and every posting are vectorized into the
**same shared vocabulary** (built from the combined corpus) — this is
what makes the two comparable at all.

### OUTPUT — Sorting & Filtering
```python
results = results.sort_values("match_score", ascending=False)
return results.head(TOP_N)
```
Scores are sorted descending and truncated to the Top-3, so the user
gets a short, ranked shortlist instead of being shown all 453 postings.

## Example Run

```
Skill #1: AI
Skill #2: LLM integration
Skill #3: genai

=== Top Recommended Job Postings ===
1. Posting #147 — 25.5% match
   Skills: r, python, machine learning, artificial intelligence, ai
2. Posting #129 — 20.4% match
   Skills: data scientist, machine learning, nlp, deep learning, python
3. Posting #101 — 20.4% match
   Skills: data scientist, machine learning, nlp, deep learning, python
```

**Note on this result:** none of the 453 postings literally contain
"LLM integration" or "genai" — the dataset predates those terms. The
model correctly fell back to the closest conceptual neighbors it could
find ("artificial intelligence", "machine learning") rather than
returning nothing or a false high score. The relatively low match
percentages (20–25%) are an honest reflection of that gap, not a bug —
a good example of the system **failing gracefully** at the edge of
what its data can represent (the "Cold Start" problem covered in the
project brief, in miniature).

## Known Limitation

This dataset was scraped specifically from "Data Scientist" search
results, so nearly every posting leans toward data/ML roles regardless
of what skills are entered. A user searching for, say, frontend or
mobile development skills will still get data-science-adjacent
postings back — the closest matches available, but not a perfect
domain fit. This is a property of the dataset's scope, not the
matching logic itself.

## How to Run

```bash
pip install pandas scikit-learn
python reccomender.py
```
Requires `raw_skills.csv` in the same folder.

## Key Concepts Demonstrated

- Content-based filtering vs. collaborative filtering
- TF-IDF feature extraction and why it beats binary overlap
- Cosine similarity vs. Euclidean distance for text-based matching
- IPO pipeline: ingestion → scoring → sorting → filtering
- Handling real, messy, non-tabular data (stringified lists, duplicate entries)
- Graceful degradation when input has no close match in the dataset (cold start)
