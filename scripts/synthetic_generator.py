import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

CANONICAL_CATEGORIES = {
    "returns",
    "shipping",
    "payments",
    "warranty",
    "membership",
    "orders",
    "products",
    "account",
    "rewards",
    "promotions",
    "sustainability",
    "business",
    "support",
    "troubleshooting",
    "pricing",
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpus")

PERSONA_PROMPTS = {
    "standard": """
You are generating synthetic evaluation questions for Project A.

Your task is to create realistic customer-style questions grounded strictly in the provided document.

Rules:
- Choose category only from this canonical set: {canonical_categories}
- Do not invent new category names.
- Do not invent facts not supported by the document.
- Generate realistic customer-style questions.
- Vary difficulty across easy, medium, and hard.
- Keep expected answers concise and grounded in the document.
- Choose a category from the canonical set that matches the document topic.
- Avoid duplicates within the generated batch.
- Return JSON only.

Generate exactly {count} items from this document.

Document name: {doc_name}
Document text:
{doc_text}

Return a JSON array only, using this schema for each item:
{{
  "query": "...",
  "expected_answer": "...",
  "difficulty": "easy | medium | hard",
  "category": "..."
}}
""".strip(),
    "frustrated": """
You are generating synthetic evaluation questions for Project A.

The question writer persona is a frustrated customer. Questions should sound impatient, annoyed, or urgent, while still being answerable strictly from the provided document.

Rules:
- Choose category only from this canonical set: {canonical_categories}
- Do not invent new category names.
- Do not invent facts not supported by the document.
- Generate realistic customer-style questions.
- Vary difficulty across easy, medium, and hard.
- Keep expected answers concise and grounded in the document.
- Choose a category from the canonical set that matches the document topic.
- Avoid duplicates within the generated batch.
- Return JSON only.

Generate exactly {count} items from this document.

Document name: {doc_name}
Document text:
{doc_text}

Return a JSON array only, using this schema for each item:
{{
  "query": "...",
  "expected_answer": "...",
  "difficulty": "easy | medium | hard",
  "category": "..."
}}
""".strip(),
    "mismatch": """
You are generating synthetic evaluation questions for Project A.

The question writer persona is a customer whose wording does not neatly match the wording in the document. Use natural paraphrases and indirect phrasing, but keep every question answerable strictly from the document.

Rules:
- Choose category only from this canonical set: {canonical_categories}
- Do not invent new category names.
- Do not invent facts not supported by the document.
- Generate realistic customer-style questions.
- Vary difficulty across easy, medium, and hard.
- Keep expected answers concise and grounded in the document.
- Choose a category from the canonical set that matches the document topic.
- Avoid duplicates within the generated batch.
- Return JSON only.

Generate exactly {count} items from this document.

Document name: {doc_name}
Document text:
{doc_text}

Return a JSON array only, using this schema for each item:
{{
  "query": "...",
  "expected_answer": "...",
  "difficulty": "easy | medium | hard",
  "category": "..."
}}
""".strip(),
}


def load_doc_text(doc_name):
    """Load a single markdown document from the corpus directory."""
    path = os.path.join(CORPUS_DIR, doc_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document not found: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_output_path(output_dir, doc_name, persona):
    """Build a timestamped output path for generated synthetic questions."""
    absolute_output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(absolute_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    doc_stem = os.path.splitext(os.path.basename(doc_name))[0]
    filename = f"synthetic_questions_processed_{timestamp}_{doc_stem}_{persona}.json"
    return os.path.join(absolute_output_dir, filename)


def clean_json_response(content):
    """Strip markdown fences from a JSON-only model response."""
    content = (content or "").strip()
    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    elif content.startswith("```"):
        content = content[len("```"):].strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    return content


def infer_category_from_doc_name(doc_name):
    """Infer a canonical fallback category from the document name."""
    name = os.path.basename(doc_name).lower()

    if "return" in name:
        return "returns"
    if "shipping" in name:
        return "shipping"
    if "payment" in name or "wallet" in name:
        return "payments"
    if "warranty" in name:
        return "warranty"
    if "membership" in name:
        return "membership"
    if "order" in name:
        return "orders"
    if "catalog" in name or "ecosystem" in name:
        return "products"
    if "account" in name:
        return "account"
    if "referral" in name:
        return "rewards"
    if "promotion" in name:
        return "promotions"
    if "sustainability" in name:
        return "sustainability"
    if "business" in name or "corporate" in name:
        return "business"
    if "support" in name or "ticket" in name or "slack" in name:
        return "support"
    if "troubleshooting" in name:
        return "troubleshooting"
    if "pricing" in name:
        return "pricing"

    return "support"


def normalize_questions(items, doc_name, persona):
    """Validate and normalize generated questions."""
    normalized = []
    fallback_category = infer_category_from_doc_name(doc_name)
    doc_stem = os.path.splitext(os.path.basename(doc_name))[0]

    for item in items:
        if not isinstance(item, dict):
            continue

        query = item.get("query")
        expected_answer = item.get("expected_answer")
        difficulty = item.get("difficulty")
        category = item.get("category")

        if not isinstance(query, str) or not query.strip():
            continue
        if not isinstance(expected_answer, str) or not expected_answer.strip():
            continue
        if not isinstance(difficulty, str) or not difficulty.strip():
            continue
        if not isinstance(category, str):
            category = ""

        normalized_difficulty = difficulty.strip().lower()
        if normalized_difficulty not in {"easy", "medium", "hard"}:
            continue

        normalized_category = category.strip().lower()
        if normalized_category not in CANONICAL_CATEGORIES:
            normalized_category = fallback_category

        normalized.append({
            "query": query.strip(),
            "expected_answer": expected_answer.strip(),
            "difficulty": normalized_difficulty,
            "category": normalized_category,
            "expected_source": doc_name,
            "persona": persona,
        })

    for index, item in enumerate(normalized, start=1):
        item["id"] = f"{doc_stem}_{persona}_{index:03d}"

    return normalized


def generate_questions(doc_name, persona="standard", count=5, output_dir="synthetic_data"):
    """Generate grounded synthetic question entries for one corpus document."""
    if persona not in PERSONA_PROMPTS:
        raise ValueError(f"Unsupported persona: {persona}")

    doc_text = load_doc_text(doc_name)
    prompt = PERSONA_PROMPTS[persona].format(
        canonical_categories=", ".join(sorted(CANONICAL_CATEGORIES)),
        doc_name=doc_name,
        doc_text=doc_text,
        count=count,
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.8,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "synthetic_questions",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "expected_answer": {"type": "string"},
                                    "difficulty": {"type": "string"},
                                    "category": {"type": "string"},
                                },
                                "required": ["query", "expected_answer", "difficulty", "category"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["questions"],
                    "additionalProperties": False,
                },
            },
        },
        messages=[
            {
                "role": "system",
                "content": "You generate grounded synthetic evaluation data. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = clean_json_response(response.choices[0].message.content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = []

    if isinstance(parsed, dict):
        parsed = parsed.get("questions", [])
    elif not isinstance(parsed, list):
        parsed = []

    questions = normalize_questions(parsed, doc_name, persona)

    output_path = build_output_path(output_dir, doc_name, persona)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(json.dumps(questions, indent=2, ensure_ascii=False))
    print(output_path)

    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", required=True)
    parser.add_argument(
        "--persona",
        default="standard",
        choices=["standard", "frustrated", "mismatch"],
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output-dir", default="synthetic_data")
    args = parser.parse_args()

    generate_questions(
        doc_name=args.doc,
        persona=args.persona,
        count=args.count,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
