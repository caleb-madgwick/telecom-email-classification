"""Compare models on the email dataset: accuracy, precision, recall, latency and cost.

Usage:
    python evaluator.py                       # original 60 emails x 2 models = 120 API calls
    python evaluator.py --dataset hard        # harder 40 emails x 2 models = 80 API calls
    python evaluator.py --per-category 2      # cheap test run: 6 emails x 2 models = 12 calls
    python evaluator.py --provider anthropic  # run one provider only
    python evaluator.py --from-results        # re-report saved results, no API calls
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from classifier import (
    CATEGORIES,
    EMAILS_PATH,
    MODELS,
    MissingAPIKeyError,
    classify_email,
    create_client,
    load_emails,
)

PROJECT_DIR = Path(__file__).parent

# Each dataset has its own results file, so a run on one can never overwrite
# the saved results of another.
DATASETS = {
    "original": (EMAILS_PATH, PROJECT_DIR / "results" / "results.json"),
    "hard": (
        PROJECT_DIR / "data" / "hard_emails.json",
        PROJECT_DIR / "results" / "hard_results.json",
    ),
}

# Standard prices in USD per 1M tokens: (input, output), as of 2026-10-25.
# Prices change, so check these before relying on the cost figures:
#   https://developers.openai.com/api/docs/pricing
#   https://platform.claude.com/docs/en/about-claude/pricing
PRICES_PER_MTOK = {
    "gpt-6-luna": (0.10, 0.50),
    "claude-haiku-5-5": (0.10, 0.50),
}


def select_sample(emails, per_category):
    """Take the first `per_category` emails from each category for a cheap test run."""
    sample = []
    for category in CATEGORIES:
        matching = [e for e in emails if e["expected_category"] == category]
        sample.extend(matching[:per_category])
    return sample


def run_evaluation(emails, providers):
    """Classify every email with every provider and return the result records."""
    # Create all clients up front so a missing key fails before any money is spent.
    clients = {provider: create_client(provider) for provider in providers}

    results = []
    for provider, client in clients.items():
        for i, email in enumerate(emails, start=1):
            result = classify_email(email, provider, client)
            outcome = result["predicted"] or f"ERROR - {result['error']}"
            print(f"[{provider} {i}/{len(emails)}] email {email['id']}: {outcome}")
            results.append(result)
    return results


def accuracy(results, predicted="predicted", expected="expected"):
    """Fraction of emails answered correctly. Errors count as incorrect.

    `predicted` and `expected` name the result fields to compare, so the same
    function scores both the category and the escalation decision.
    """
    correct = sum(r[predicted] == r[expected] for r in results)
    return correct / len(results)


def precision_recall(results, positive, predicted="predicted", expected="expected"):
    """Precision and recall, treating `positive` as the class we're looking for.

    precision: of the emails predicted as `positive`, how many really were?
    recall:    of the emails that really are `positive`, how many did we find?

    Either value is None when it can't be calculated, e.g. recall when no
    email in the sample is actually `positive`.
    """
    tp = sum(r[predicted] == positive and r[expected] == positive for r in results)
    fp = sum(r[predicted] == positive and r[expected] != positive for r in results)
    fn = sum(r[predicted] != positive and r[expected] == positive for r in results)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return precision, recall


def mean_or_none(values):
    """Mean of the values that could be calculated, or None if there are none."""
    known = [v for v in values if v is not None]
    return statistics.mean(known) if known else None


def estimated_cost(result):
    """Cost in USD of one API call, from the token counts the provider reported."""
    input_price, output_price = PRICES_PER_MTOK[result["model"]]
    return (
        result["input_tokens"] * input_price + result["output_tokens"] * output_price
    ) / 1_000_000


def summarise(results):
    """Calculate every metric for one model's results."""
    per_class = {c: precision_recall(results, c) for c in CATEGORIES}
    escalation_fields = {"predicted": "predicted_escalation", "expected": "expected_escalation"}
    latencies = [r["latency_s"] for r in results if r["latency_s"] is not None]
    total_cost = sum(estimated_cost(r) for r in results)
    return {
        "emails": len(results),
        "errors": sum(r["error"] is not None for r in results),
        "accuracy": accuracy(results),
        "per_class": per_class,
        "macro_precision": mean_or_none(p for p, _ in per_class.values()),
        "macro_recall": mean_or_none(r for _, r in per_class.values()),
        "escalation_accuracy": accuracy(results, **escalation_fields),
        "escalation_precision_recall": precision_recall(results, True, **escalation_fields),
        "mean_latency_s": statistics.mean(latencies) if latencies else None,
        "median_latency_s": statistics.median(latencies) if latencies else None,
        "cost_per_1000": total_cost / len(results) * 1000,
    }


def find_disagreements(results):
    """Emails where the models gave different answers, as {email_id: results}.

    An answer is the (category, escalation) pair, so this includes emails
    where one model was right and the other wrong.
    """
    by_email = {}
    for r in results:
        by_email.setdefault(r["email_id"], []).append(r)
    return {
        email_id: email_results
        for email_id, email_results in by_email.items()
        if len({(r["predicted"], r["predicted_escalation"]) for r in email_results}) > 1
    }


def format_seconds(value):
    return "n/a" if value is None else f"{value:.2f}"


def format_percent(value):
    return "n/a" if value is None else f"{value:.1%}"


def print_report(results):
    """Print a Markdown comparison table (ready to paste into the README)."""
    models = list(dict.fromkeys(r["model"] for r in results))
    summaries = {m: summarise([r for r in results if r["model"] == m]) for m in models}

    def header(label):
        print(f"\n| {label} | " + " | ".join(models) + " |")
        print("|---|" + "---:|" * len(models))

    def row(label, format_value):
        values = [format_value(summaries[m]) for m in models]
        print(f"| {label} | " + " | ".join(values) + " |")

    header("Metric")
    row("Emails classified", lambda s: str(s["emails"]))
    row("Errors", lambda s: str(s["errors"]))
    row("Accuracy", lambda s: f"{s['accuracy']:.1%}")
    row("Macro precision", lambda s: format_percent(s["macro_precision"]))
    row("Macro recall", lambda s: format_percent(s["macro_recall"]))
    row("Escalation accuracy", lambda s: f"{s['escalation_accuracy']:.1%}")
    row("Escalation precision", lambda s: format_percent(s["escalation_precision_recall"][0]))
    row("Escalation recall", lambda s: format_percent(s["escalation_precision_recall"][1]))
    row("Mean latency (s)", lambda s: format_seconds(s["mean_latency_s"]))
    row("Median latency (s)", lambda s: format_seconds(s["median_latency_s"]))
    row("Est. cost per 1,000 emails", lambda s: f"${s['cost_per_1000']:.4f}")

    header("Precision / recall")
    for category in CATEGORIES:
        row(category, lambda s: " / ".join(map(format_percent, s["per_class"][category])))

    mistakes = [
        r
        for r in results
        if r["predicted"] != r["expected"]
        or r["predicted_escalation"] != r["expected_escalation"]
    ]
    print(f"\nMistakes or failures: {len(mistakes)}")
    for r in mistakes:
        expected = f"{r['expected']} (escalate={r['expected_escalation']})"
        got = f"{r['predicted'] or 'ERROR'} (escalate={r['predicted_escalation']})"
        detail = r["reasoning"] or r["error"]
        print(f"- [{r['model']}] email {r['email_id']}: expected {expected}, got {got}. {detail}")

    disagreements = find_disagreements(results)
    print(f"\nModels disagreed on: {len(disagreements)}")
    for email_id, email_results in disagreements.items():
        answers = "; ".join(
            f"{r['model']} {r['predicted'] or 'ERROR'} (escalate={r['predicted_escalation']})"
            for r in email_results
        )
        first = email_results[0]
        expected = f"{first['expected']} (escalate={first['expected_escalation']})"
        print(f"- email {email_id}: {answers}. Expected {expected}")


def main():
    parser = argparse.ArgumentParser(description="Compare LLM email classifiers.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="original",
        help="which email set to use (default: original)",
    )
    parser.add_argument(
        "--provider", choices=sorted(MODELS), help="run one provider only (default: both)"
    )
    parser.add_argument(
        "--per-category",
        type=int,
        metavar="N",
        help="only classify the first N emails of each category (cheap test run)",
    )
    parser.add_argument(
        "--from-results",
        action="store_true",
        help="skip API calls and report on the saved results",
    )
    args = parser.parse_args()
    emails_path, results_path = DATASETS[args.dataset]
    results_file = results_path.relative_to(PROJECT_DIR)

    if args.from_results:
        if not results_path.exists():
            sys.exit(
                f"Error: {results_file} not found. "
                f"Run python evaluator.py --dataset {args.dataset} first."
            )
        results = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        emails = load_emails(emails_path)
        if args.per_category:
            emails = select_sample(emails, args.per_category)
        providers = [args.provider] if args.provider else list(MODELS)

        try:
            results = run_evaluation(emails, providers)
        except MissingAPIKeyError as e:
            sys.exit(f"Error: {e}")

        # git doesn't track empty folders, so results/ may not exist yet.
        results_path.parent.mkdir(exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved {len(results)} results to {results_file}")

    print_report(results)


if __name__ == "__main__":
    main()
