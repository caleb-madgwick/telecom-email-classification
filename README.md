# Telecom Email Classification & Evaluation

Comparing two LLMs at classifying and escalating customer emails for a fictional telecom provider. All data is synthetic.

## Overview

Each model reads a customer email and returns structured JSON:

```json
{
  "category": "billing",
  "escalate_to_human": true,
  "reasoning": "Customer disputes $240 in roaming charges, over the $100 threshold."
}
```

- **Categories:** `billing`, `technical_support` or `general_enquiry`
- **Escalation:** whether a person should review the email first, under five written rules (hardship, legal or privacy risk, complaints about staff, customers about to leave, disputes over $100)
- **Datasets:** 60 original emails, plus 40 harder emails whose labels were committed before any model saw them

![Architecture diagram](docs/architecture.svg)

## Approach & Trade-offs

- **Same prompt, schema and emails for both models**, so the model is the only variable. *Trade-off:* neither prompt is tuned to its model.
- **Two layers of output checking:** each provider enforces the JSON schema, then the code checks the output again. Invalid output counts as an error, never a guess. *Trade-off:* an answer can be valid but still wrong (see email 63 below).
- **Written rules instead of model judgement.** A tie-break rule decides multi-topic emails, and five rules decide escalation. *Trade-off:* rules are predictable and testable, but miss situations they don't name.
- **Small, low-cost models** (each provider's fast tier), because classification shouldn't need a frontier model.
- **Plain Python and the official SDKs**, with no framework. Results are saved, so metrics can be recalculated without paying for new API calls.

## Model Choice & Cost

**I'd choose Claude Haiku 5.5.**
- It caught every email that needed a person.
- It was faster in both runs.
- For escalation, a missed case costs more than the price difference.

**Estimated production cost:**

| Model | Per 1,000 emails | Per million emails |
|---|---:|---:|
| Claude Haiku 5.5 | **$0.14** | $140 |
| GPT-6 Luna | $0.08 | $80 |

These figures use the token counts measured in both runs, at list prices. Real emails are usually longer than these, which would raise the cost. Batch or caching discounts would lower it.

## Results

| | GPT-6 Luna | Claude Haiku 5.5 |
|---|---:|---:|
| Category accuracy (original / harder) | 100% / 100% | 100% / 100% |
| Escalations caught (out of 21) | 19 | **21** |
| Unnecessary escalations | **1** | 2 |
| Median latency | 1.6–1.7 s | **1.4–1.6 s** |
| Invalid responses | 0 | 0 |

Each dataset was run once per model. Total API spend was $0.02.

**What the results show:**
- **Categorisation was easy; escalation was where the models differed.** Both scored 100% on categories, even with indirect wording, multiple requests and slang.
- **The models fail in opposite directions.** GPT applies the rules literally and missed 2 escalations. Claude is more cautious and over-escalated twice. A missed escalation is the worse mistake.
- **Equal list prices don't mean equal cost.** Claude used about twice the input tokens and wrote longer explanations, so it cost about 1.75× as much per email.
- **Both models resisted prompt injection.** All three embedded attempts to override the instructions were ignored.

**What they don't tell you:**
- **Which model is better at categories.** Both scored 100%.
- **Whether the escalation gap is real.** It's 2 misses against 0. With 21 escalation cases, one email shifts recall by about 5 points. The pattern was the same in both datasets, but that isn't proof.
- **How the models would do on real email.** The emails are synthetic and cleaner than real customer mail. They were drafted with help from a Claude model, which could favour Claude Haiku. All labels come from one person, and each dataset was run once.

## Where It's Weakest

- **A decision can contradict its own reasoning.** On email 63, GPT answered "don't escalate" while its own reasoning said the $100 threshold was exceeded. The schema asks for the decision before the reasoning, so the model commits before it thinks.
- **The $100 rule is ambiguous.** It doesn't say whether it means the amount charged so far or what the total could grow to. That caused both mistakes on the original set (emails 13 and 20).
- **The rules don't cover everything that needs a person.** On the bereavement email (93), both models escalated even though no rule applied.
- **Category weak spots are unknown.** No category mistakes in 100 emails means none were found, not that there are none.

## Before Real Customers

- **Fix the known weaknesses first:** ask for the reasoning before the decision, rewrite the $100 rule, and test both changes on new emails.
- **Evaluate on real data:** a few hundred anonymised emails labelled by support staff, with an agreed target for escalation recall close to 100%.
- **Start in shadow mode:** the model suggests a category and escalation, staff make the final decision, and every disagreement is reviewed before the model routes anything itself.
- **Protect privacy:** mask personal details before sending emails to a third-party API, and confirm each provider's data retention terms.
- **Monitor it:** track missed escalations, error rate, latency and cost, and re-run the evaluation before any model or prompt change.

## Run It

```bash
pip install -r requirements.txt
cp .env.example .env                        # add OPENAI_API_KEY and ANTHROPIC_API_KEY
pytest                                      # 31 offline tests, no API calls
python evaluator.py                         # original 60 emails
python evaluator.py --dataset hard          # harder 40 emails
python evaluator.py --dataset hard --from-results   # re-print saved results for free
```

`classifier.py` makes the API calls and validates responses. `evaluator.py` runs the comparison and calculates the metrics. Datasets are in `data/` and saved results in `results/`.
