# Telecom Email Classification & Evaluation

An independent LLM evaluation project. All customers, emails and the telecom provider are synthetic.

## Overview

Customer emails arrive as unstructured text and need to reach the right team. This project sends synthetic emails to one OpenAI model and one Anthropic model. Each model sorts every email into `billing`, `technical_support` or `general_enquiry`, and decides whether a person should review it before anything happens.

The models are compared on accuracy, escalation, speed and cost across two datasets:
- the original 60-email benchmark
- 40 harder emails added afterwards

```json
{"category": "billing", "escalate_to_human": true, "reasoning": "Customer disputes $240 in roaming charges, over the $100 review threshold."}
```

## Architecture

![Architecture diagram](docs/architecture.svg)

Both models receive the same prompt, emails and output schema. Every response is checked locally before it counts. Results are saved, so metrics can be recalculated without new API calls.

## Project Structure

```text
classifier.py           prompt, schema, API calls and response validation
evaluator.py            runs both models and calculates metrics
test_classifier.py      offline tests with mocked API clients
data/emails.json        original 60 emails with expected labels
data/hard_emails.json   40 harder emails, each with a note explaining its label
results/                saved per-email results for each dataset
docs/                   architecture diagram
```

## Key Design Decisions

- **One prompt, one schema:** the model is the only thing that differs between runs. Trade-off: neither prompt is tuned to its model.
- **Two layers of output checking:** each provider enforces the JSON schema, then `validate_response()` checks it again. Invalid output is recorded as an error and counted as wrong, never guessed.
- **Explicit tie-break rule:** a multi-topic email is labelled by what the customer wants done: money back is `billing`, something fixed is `technical_support`. Trade-off: some real emails need more than one label.
- **Written escalation rules:** the model escalates an email if any of five rules applies:
  - financial hardship
  - legal, credit or privacy risk
  - a complaint about staff
  - a customer about to leave
  - a dispute over $100

  Trade-off: written rules are predictable and testable, but they miss situations they don't name.
- **Labels before results:** the harder emails and their labels were reviewed and committed to git before any model saw them, so the labels couldn't be adjusted to fit the results.
- **Fail on configuration, continue on blips:** a wrong key or model name stops the run immediately. A timeout or bad response is logged and the run continues.
- **Minimal dependencies:** the official SDKs, `python-dotenv` and `pytest`. Metrics are calculated by hand rather than with scikit-learn.

## Model Choice & Cost

I chose **gpt-6-luna** and **claude-haiku-5-5** for three reasons:

- Classification is a simple task, so it doesn't need a frontier model.
- Each is its provider's small, fast tier.
- Both list at $0.10 / $0.50 per million input / output tokens, so price doesn't favour either.

**Measured cost:**

| Model | Input tokens / email | Output tokens / email | Cost per 1,000 (original / harder) |
|---|---:|---:|---:|
| gpt-6-luna | ~496 | ~63 | $0.080 / $0.083 |
| claude-haiku-5-5 | ~945 | ~94 | $0.141 / $0.141 |

Equal list prices did not mean equal cost. Claude used about twice the input tokens, probably because Anthropic adds the schema to the prompt and its tokenizer counts the same text differently. It also wrote longer reasoning than the prompt asked for. At 1 million emails a month, that's about $80 against $141. Both runs together cost $0.022.

**For this use case I'd choose claude-haiku-5-5.**
- It escalated all 21 emails that needed a person. GPT missed 2.
- It had the faster median response time in both runs.
- For an escalation decision, a missed case costs more than about $61 per million emails.

This rests on a small number of mistakes on synthetic data. The pattern was consistent across both datasets, but it isn't proven.

## Run the Evaluation

Python 3.10+. Copy `.env.example` to `.env` and add both API keys.

```bash
pip install -r requirements.txt
pytest                                       # 31 offline tests, no API calls
python evaluator.py --per-category 2         # quick run, 12 calls
python evaluator.py                          # original 60 emails, 120 calls
python evaluator.py --dataset hard           # harder 40 emails, 80 calls
python evaluator.py --dataset hard --from-results   # re-report saved results for free
```

## Eval Results

Each dataset was run once per model, with 0 invalid responses.

### Original benchmark: 60 emails

The original 60 emails include 9 deliberately ambiguous ones, and 8 that should escalate.

| Metric | gpt-6-luna | claude-haiku-5-5 |
|---|---:|---:|
| Category accuracy | 100.0% | 100.0% |
| Escalation precision | 100.0% | 88.9% |
| **Escalation recall** | **87.5%** | **100.0%** |
| Median latency (s) | 1.63 | 1.55 |
| Cost per 1,000 emails | $0.080 | $0.141 |

### Harder set: 40 emails

13 of these emails should escalate. The set includes:
- indirect wording
- emails with more than one request
- forwarded threads, slang, all caps and non-native English
- money amounts either side of $100
- three prompt-injection attempts

| Metric | gpt-6-luna | claude-haiku-5-5 |
|---|---:|---:|
| Category accuracy | 100.0% | 100.0% |
| Escalation precision | 92.3% | 92.9% |
| **Escalation recall** | **92.3%** | **100.0%** |
| Median latency (s) | 1.74 | 1.39 |
| Cost per 1,000 emails | $0.083 | $0.141 |

### Every mistake

| Email | Set | Case | gpt-6-luna | claude-haiku-5-5 |
|---|---|---|---|---|
| 20 | Original | $38/month since June, so over $100 is implied but never stated | Missed escalation | Correct |
| 13 | Original | $4.99/week premium SMS charges | Correct | Escalated unnecessarily ("may exceed $100 over time") |
| 63 | Harder | Two charges totalling $100.50 | Missed escalation, although its own reasoning says the total "exceeds the $100 escalation threshold" | Correct |
| 93 | Harder | Bereaved customer closing a parent's account, where none of the five rules applies | Escalated | Escalated |

**What the results tell you:**
- **Categories are easy for both models.** Both scored 100% on both sets, including the harder emails. All the differences are in escalation.
- **The models fail in opposite directions, consistently.** GPT reads the rules strictly and missed 2 of 21 escalations. Claude is more cautious: it missed none but escalated 2 unnecessarily. Recall matters most here: a missed escalation reaches a customer unchecked, while a false alarm only costs a reviewer's time.
- **Both models resisted prompt injection.** Both ignored all three attempts to override the instructions from inside an email.
- **Claude was slightly faster** in both runs, by 0.1 to 0.35 seconds at the median.

**What they don't tell you:**
- **Which model is better at categories.** Even the harder set didn't separate them.
- **Whether the escalation gap is real.** It's 2 misses against 0. The direction was the same in both runs, but the numbers are small.
- **Real-world performance.** The emails are synthetic and cleaner than real customer mail. They were drafted with help from a Claude model (the same family as one of the models tested) and reviewed and labelled by me, which could favour Claude.
- **Run-to-run variation.** Each dataset was run once, with default sampling, from one location.

## Where It's Weakest

- **A decision can contradict its own reasoning.** On email 63, GPT answered `escalate_to_human: false` while its reasoning said the threshold was exceeded. The schema asks for the decision *before* the reasoning, so the model commits to an answer before working it out. Validation can't catch this, because both fields are individually valid. Asking for the reasoning first is the obvious next experiment.
- **The rules don't cover everything that needs a person.** Both models escalated the bereavement email (93) even though no rule applies. That's arguably the right instinct, but it means behaviour isn't fully governed by the written rules.
- **The $100 rule is ambiguous.** It doesn't say whether it means the amount charged so far or what the total could grow to. That caused both mistakes on the original set (emails 13 and 20). The harder set deliberately avoided this ambiguity, and the rule was left unchanged for a later prompt revision.
- **Claude ignores length instructions.** It writes longer explanations than "one short sentence", which adds cost at scale.
- **Category weak spots are unknown.** No category mistakes in 100 emails means these datasets haven't found any, not that there are none.
- **Labels come from one person.** Every label is my judgement under the written rules.

## Before Real Customers (design only)

- **Real data:** evaluate on a few hundred real, anonymised emails labelled by support staff, with agreed accuracy targets per category.
- **Confidence and review:** send ambiguous or failed classifications to a human queue instead of routing them automatically.
- **Monitoring:** track accuracy, error rate, latency and cost over time, and re-run the evaluation before any model or prompt change.
- **Privacy:** remove or mask personal details before sending emails to a third-party API, and confirm each provider's data retention terms.
- **Reliability:** add rate limiting and a fallback to the second provider if one is unavailable.
