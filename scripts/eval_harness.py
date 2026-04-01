"""
Evaluation Harness — Session 1 Starter

This is a SKELETON. During Session 1, we'll build each function
from scratch to create a complete eval pipeline.

Functions to implement:
  1. check_retrieval_hit() — is the expected source in the top-K results?
  2. calculate_mrr() — how high is the first relevant chunk ranked?
  3. judge_faithfulness() — is the answer grounded in the context? (LLM-as-judge)
  4. judge_correctness() — does the answer match the expected answer? (LLM-as-judge)
  5. run_eval() — orchestrate everything and produce a scorecard

Run: python scripts/eval_harness.py
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

SCRIPT_DIR = os.path.dirname(__file__)


# =========================================================================
# GOLDEN DATASET
# =========================================================================
# TODO: We'll build this together in Session 1.
# Start with 5 hand-written question-answer-context triples.
# Format:
# {
#     "id": "q01",
#     "query": "What is the standard return window?",
#     "expected_answer": "30 calendar days from delivery date.",
#     "expected_source": "01_return_policy.md",
#     "difficulty": "easy",
#     "category": "returns"
# }
# =========================================================================


def load_golden_dataset():
    """Load the golden dataset from JSON file."""
    path = os.path.join(SCRIPT_DIR, "golden_dataset.json")
    if not os.path.exists(path):
        print("No golden_dataset.json found. Create one first!")
        return []
    with open(path) as f:
        return json.loads(f.read())


# =========================================================================
# RETRIEVAL METRICS
# =========================================================================

def check_retrieval_hit(retrieved_chunks, expected_source):
    """
    Is the expected source document in the retrieved chunks?
    Returns True/False.

    TODO: Implement this in Session 1.
    """
    return any(chunk["doc_name"] == expected_source for chunk in retrieved_chunks)


def calculate_mrr(retrieved_chunks, expected_source):
    """
    Mean Reciprocal Rank — how high is the first relevant chunk?
    If relevant chunk is at position 1: MRR = 1.0
    If at position 3: MRR = 0.33
    If not found: MRR = 0.0

    TODO: Implement this in Session 1.
    """
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk["doc_name"] == expected_source:
            return 1.0 / rank
    return 0.0


# =========================================================================
# GENERATION METRICS (LLM-as-Judge)
# =========================================================================

def judge_faithfulness(query, answer, context):
    """
    Is the answer grounded in the retrieved context?
    Uses GPT-4o-mini as a judge with a structured rubric.
    Returns: {"score": 1-5, "reason": "explanation"}

    TODO: Implement this in Session 1.
    """
    prompt = f"""
You are judging whether an answer is faithful to the provided context.

Score using this rubric:
- 5 = every claim is explicitly supported by context
- 3 = some claims are unsupported or vague
- 1 = fabricated or hallucinated information

Return JSON only in this exact format:
{{"score": 5, "reason": "brief explanation"}}

Query: {query}
Answer: {answer}
Context: {context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a strict evaluation judge. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    elif content.startswith("```"):
        content = content[len("```"):].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    try:
        result = json.loads(content)
        return {
            "score": int(result["score"]),
            "reason": str(result["reason"]),
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {
            "score": 1,
            "reason": "Failed to parse judge response as valid JSON.",
        }


def judge_correctness(query, answer, expected_answer):
    """
    Does the answer match the expected answer?
    Uses GPT-4o-mini as a judge.
    Returns: {"score": 1-5, "reason": "explanation"}

    TODO: Implement this in Session 1.
    """
    prompt = f"""
You are judging whether an answer is correct compared with the expected answer.

Score using this rubric:
- 5 = answer matches expected_answer exactly or is semantically equivalent
- 3 = partially correct or incomplete
- 1 = incorrect or irrelevant

Return JSON only in this exact format:
{{"score": 5, "reason": "brief explanation"}}

Query: {query}
Answer: {answer}
Expected Answer: {expected_answer}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a strict evaluation judge. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    elif content.startswith("```"):
        content = content[len("```"):].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    try:
        result = json.loads(content)
        return {
            "score": int(result["score"]),
            "reason": str(result["reason"]),
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {
            "score": 1,
            "reason": "Failed to parse judge response as valid JSON.",
        }


# =========================================================================
# EVAL RUNNER
# =========================================================================

def run_eval(save_baseline=False):
    """
    Run the full evaluation:
    1. Load golden dataset
    2. Run each query through the RAG pipeline
    3. Score retrieval (hit rate, MRR)
    4. Score generation (faithfulness, correctness)
    5. Print scorecard

    TODO: Implement this in Session 1.
    """
    from rag import ask

    dataset = load_golden_dataset()
    results = []
    total = len(dataset)

    for index, entry in enumerate(dataset, start=1):
        query = entry["query"]
        category = entry["category"]
        expected_answer = entry["expected_answer"]
        expected_source = entry["expected_source"]
        item_id = entry["id"]

        print(f"[{index}/{total}] Processing {item_id} | category={category} | save_baseline={save_baseline}")

        response = ask(query)
        answer = response["answer"]
        retrieved_chunks = response["retrieved_chunks"]
        context = response["context"]
        trace_id = response.get("trace_id")

        retrieval_hit = check_retrieval_hit(retrieved_chunks, expected_source)
        mrr = calculate_mrr(retrieved_chunks, expected_source)
        faithfulness = judge_faithfulness(query, answer, context)
        correctness = judge_correctness(query, answer, expected_answer)

        results.append({
            "id": item_id,
            "query": query,
            "category": category,
            "expected_source": expected_source,
            "expected_answer": expected_answer,
            "answer": answer,
            "retrieval_hit": retrieval_hit,
            "mrr": mrr,
            "faithfulness_score": faithfulness["score"],
            "faithfulness_reason": faithfulness["reason"],
            "correctness_score": correctness["score"],
            "correctness_reason": correctness["reason"],
            "trace_id": trace_id,
        })

        print(f"Completed {item_id}")

    output_path = os.path.join(SCRIPT_DIR, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    total_queries = len(results)
    hit_rate = sum(result["retrieval_hit"] for result in results) / total_queries if total_queries else 0.0
    mean_mrr = sum(result["mrr"] for result in results) / total_queries if total_queries else 0.0
    mean_faithfulness = (
        sum(result["faithfulness_score"] for result in results) / total_queries if total_queries else 0.0
    )
    mean_correctness = (
        sum(result["correctness_score"] for result in results) / total_queries if total_queries else 0.0
    )
    stratified_summary = run_stratified_eval(results)

    baseline_summary = {
        "total_queries": total_queries,
        "hit_rate": hit_rate,
        "mean_mrr": mean_mrr,
        "mean_faithfulness": mean_faithfulness,
        "mean_correctness": mean_correctness,
        "per_category": stratified_summary["per_category"],
        "worst_categories": stratified_summary["worst_categories"],
    }

    if save_baseline:
        baseline_path = os.path.join(SCRIPT_DIR, "baseline_scores.json")
        with open(baseline_path, "w") as f:
            json.dump(baseline_summary, f, indent=2)

    print("Evaluation Summary")
    print(f"Total queries: {total_queries}")
    print(f"Hit rate: {hit_rate:.2f}")
    print(f"Mean MRR: {mean_mrr:.2f}")
    print(f"Mean faithfulness: {mean_faithfulness:.2f}")
    print(f"Mean correctness: {mean_correctness:.2f}")


def run_stratified_eval(results):
    """Compute and print per-category evaluation metrics."""
    grouped_results = {}

    for result in results:
        category = result["category"]
        grouped_results.setdefault(category, []).append(result)

    per_category = {}
    for category, items in grouped_results.items():
        count = len(items)
        hit_rate = sum(item["retrieval_hit"] for item in items) / count if count else 0.0
        mean_faithfulness = sum(item["faithfulness_score"] for item in items) / count if count else 0.0
        mean_correctness = sum(item["correctness_score"] for item in items) / count if count else 0.0

        per_category[category] = {
            "hit_rate": hit_rate,
            "mean_faithfulness": mean_faithfulness,
            "mean_correctness": mean_correctness,
            "count": count,
        }

    print("Stratified Evaluation Summary")
    print(f"{'Category':<20} {'Count':>5} {'Hit Rate':>10} {'Faithfulness':>14} {'Correctness':>14}")
    for category in sorted(per_category):
        metrics = per_category[category]
        print(
            f"{category:<20} "
            f"{metrics['count']:>5} "
            f"{metrics['hit_rate']:>10.2f} "
            f"{metrics['mean_faithfulness']:>14.2f} "
            f"{metrics['mean_correctness']:>14.2f}"
        )

    worst_categories = sorted(
        per_category.items(),
        key=lambda item: item[1]["mean_correctness"],
    )[:3]
    worst_categories = [category for category, _ in worst_categories]

    print("\nWorst Categories")
    for category in worst_categories:
        print(f"- {category}")

    return {
        "per_category": per_category,
        "worst_categories": worst_categories,
    }


if __name__ == "__main__":
    # print("Eval harness skeleton loaded.")
    # print("Functions to implement: check_retrieval_hit, calculate_mrr,")
    # print("judge_faithfulness, judge_correctness, run_eval")
    # print("\nWe'll build these together in Session 1.")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-baseline", action="store_true")
    args = parser.parse_args()

    print(f"Starting eval harness (save_baseline={args.save_baseline})")
    run_eval(save_baseline=args.save_baseline)
