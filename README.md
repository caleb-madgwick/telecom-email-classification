# Telecom Email Classification & Evaluation

An independent LLM evaluation project. All customers, emails and the telecom provider are synthetic.

## Overview

Customer emails arrive as unstructured text and need to reach the right team. This project sends 60 synthetic emails to one OpenAI model and one Anthropic model. Each model sorts them into `billing`, `technical_support` or `general_enquiry`. It also decides whether a person should review the email before anything happens. The project measures which model is more accurate, faster and cheaper.

```json
{"category": "billing", "escalate_to_human": true, "reasoning": "Customer disputes $240 in roaming charges, over the $100 review threshold."}
```

## Architecture

![Architecture diagram](docs/architecture.svg)

Both models receive the same prompt, emails and output schema. Every response is checked locally before it counts. Results are saved, so metrics can be recalculated without new API calls.

## Project Structure

```text
classifier.py        prompt, schema, API calls and response validation
evaluator.py         runs both models and calculates metrics
test_classifier.py   offline tests with mocked API clients
data/emails.json     60 synthetic emails with expected labels
results/             saved per-email results from each run
docs/                architecture diagram
```

## Key Design Decisions

- **One prompt, one schema:** the model is the only thing that differs between the two runs. Trade-off: neither prompt is tuned to its model.
- **Two layers of output checking:** each provider enforces the JSON schema, then `validate_response()` checks it again. Invalid output is recorded as an error and counted as wrong, never guessed.
- **Explicit tie-break rule:** a multi-topic email is labelled by what the customer wants done: money back is `billing`, something fixed is `technical_support`. Trade-off: some real emails need more than one label.
- **Written escalation rules:** the model escalates an email if any of five rules applies:
  - financial hardship
  - legal, credit or privacy risk
  - a complaint about staff
  - a customer about to leave
  - a dispute over $100

  Trade-off: written rules are predictable and testable, but they miss situations they don't name.
- **Fail on configuration, continue on blips:** a wrong key or model name stops the run immediately. A timeout or bad response is logged and the run continues.
- **Minimal dependencies:** the official SDKs, `python-dotenv` and `pytest`. Metrics are calculated by hand rather than with scikit-learn.

## Model Choice & Cost

I chose **gpt-6-luna** and **claude-haiku-5-5** for three reasons:

- Classification is a simple task, so it doesn't need a frontier model.
- Each is its provider's small, fast tier.
- Both list at $0.10 / $0.50 per million input / output tokens, so price doesn't favour either.

**Estimated cost:** about **$0.08 per 1,000 emails**. This assumes roughly 500 input and 50 output tokens per email.

**Measured cost:** TODO, from the full run.

## Run the Evaluation

Python 3.10+. Copy `.env.example` to `.env` and add both API keys.

```bash
pip install -r requirements.txt
pytest                                 # 29 offline tests, no API calls
python evaluator.py --per-category 2   # quick run, 12 calls
python evaluator.py                    # full run, 120 calls
python evaluator.py --from-results     # re-report saved results for free
```

## Eval Results

TODO: paste the table printed by `python evaluator.py`.

**What the results tell you:**
- How each model performs on clear and deliberately ambiguous emails using the same prompt.
- How reliably each model flags the 8 emails that need a person.
- Real latency and token cost.

For escalation, **recall matters most**: a missed escalation reaches a customer unchecked, while a false alarm only costs a reviewer's time.

**What they don't tell you:**
- With 60 emails, one mistake is about 1.7 percentage points, so small gaps are noise.
- Only 8 emails should escalate, so each missed escalation moves recall by 12.5 points.
- The emails are synthetic and cleaner than real customer mail.
- Latency was measured from one location at one time.

## Where It's Weakest

TODO: confirm against the misclassified emails from the full run.

Expected weak spots:
- **Mixed-intent emails:** for example, an outage plus a request for a credit (emails 16, 17, 19, 35, 36, 39, 55, 58, 60).
- **Boundary between general enquiries and billing:** questions about plans or cancellation fees.
- **Escalation that needs arithmetic or interpretation:**
  - Email 20 implies over $100 overcharged without stating a total.
  - Email 19 signals hardship without saying "hardship".
- **Labels from one person:** every category and escalation label is my judgement under the written rules.

## Before Real Customers (design only)

- **Real data:** evaluate on a few hundred real, anonymised emails labelled by support staff, with agreed accuracy targets per category.
- **Confidence and review:** send ambiguous or failed classifications to a human queue instead of routing them automatically.
- **Monitoring:** track accuracy, error rate, latency and cost over time, and re-run the evaluation before any model or prompt change.
- **Privacy:** remove or mask personal details before sending emails to a third-party API, and confirm each provider's data retention terms.
- **Reliability:** add rate limiting and a fallback to the second provider if one is unavailable.
