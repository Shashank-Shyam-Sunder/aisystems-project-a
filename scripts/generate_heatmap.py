import argparse
import csv
import json
import os

import plotly.graph_objects as go


SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_INPUT_PATH = os.path.join(SCRIPT_DIR, "eval_results.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "failure_reports")
DIFFICULTY_ORDER = ["easy", "medium", "hard"]
REQUIRED_FIELDS = {"category", "difficulty", "correctness_score"}


def load_results(input_path):
    """Load eval results from JSON."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Eval results JSON must contain a list of result objects.")

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Result at position {index} is not a JSON object.")
        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            raise ValueError(
                f"Result at position {index} is missing required fields: {', '.join(sorted(missing))}"
            )

    return data


def build_pivot(results):
    """Aggregate mean correctness percentage by category and difficulty."""
    grouped = {}

    for item in results:
        category = str(item["category"]).strip()
        difficulty = str(item["difficulty"]).strip().lower()
        correctness_score = item["correctness_score"]

        if difficulty not in DIFFICULTY_ORDER:
            continue

        try:
            correctness_score = float(correctness_score)
        except (TypeError, ValueError):
            continue

        grouped.setdefault(category, {}).setdefault(difficulty, []).append(correctness_score)

    categories = sorted(grouped.keys())
    pivot = []

    for category in categories:
        row = {"category": category}
        for difficulty in DIFFICULTY_ORDER:
            scores = grouped[category].get(difficulty)
            if scores:
                mean_correctness = sum(scores) / len(scores)
                row[difficulty] = (mean_correctness / 5.0) * 100.0
            else:
                row[difficulty] = None
        pivot.append(row)

    return pivot


def save_pivot_csv(pivot, output_path):
    """Save the aggregated pivot table as CSV."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category"] + DIFFICULTY_ORDER)
        for row in pivot:
            writer.writerow([row["category"]] + [row[difficulty] for difficulty in DIFFICULTY_ORDER])


def build_heatmap(pivot):
    """Build a Plotly heat map figure."""
    categories = [row["category"] for row in pivot]
    z_values = [[row[difficulty] for difficulty in DIFFICULTY_ORDER] for row in pivot]
    text_values = [
        [f"{value:.1f}%" if value is not None else "" for value in row_values]
        for row_values in z_values
    ]

    colorscale = [
        [0.00, "#c0392b"],
        [0.59, "#c0392b"],
        [0.60, "#f1c40f"],
        [0.79, "#f1c40f"],
        [0.80, "#27ae60"],
        [1.00, "#27ae60"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=DIFFICULTY_ORDER,
            y=categories,
            text=text_values,
            texttemplate="%{text}",
            textfont={"color": "black"},
            colorscale=colorscale,
            zmin=0,
            zmax=100,
            hovertemplate="Category: %{y}<br>Difficulty: %{x}<br>Correctness: %{z:.1f}%<extra></extra>",
            colorbar={"title": "Correctness %"},
        )
    )

    fig.update_layout(
        title="Category × Difficulty Correctness Heat Map",
        xaxis_title="Difficulty",
        yaxis_title="Category",
    )

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    args = parser.parse_args()

    results = load_results(args.input)
    pivot = build_pivot(results)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    html_path = os.path.join(OUTPUT_DIR, "category_difficulty_heatmap.html")
    csv_path = os.path.join(OUTPUT_DIR, "category_difficulty_heatmap.csv")

    save_pivot_csv(pivot, csv_path)
    fig = build_heatmap(pivot)
    fig.write_html(html_path)

    print(html_path)
    print(csv_path)


if __name__ == "__main__":
    main()
