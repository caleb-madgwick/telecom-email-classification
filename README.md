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

- **Categories:** `billing`, `technical_support` or `general_enquiry`.
- **Escalation:** whether a person should review the email first. Five written rules decide this: financial hardship; legal, regulatory or credit risk (including privacy requests and formal complaints); complaints about staff; customers about to leave because of poor service; and disputes over $100.
- **Datasets:** 60 original emails, plus 40 harder emails whose labels were committed before any model saw them.

![Architecture diagram](docs/architecture.svg)

## Approach & Trade-offs

- **Same prompt, schema and emails for both models**, so the model is the only variable. *Trade-off:* neither prompt is tuned to its model.
- **Two layers of output checking.** Each provider's structured-output feature constrains the response to a JSON schema. The code then checks it again: valid JSON, an allowed category, a true/false escalation decision and non-empty reasoning. Invalid output is recorded as an error, never guessed. *Trade-off:* an answer can be valid but still wrong (see email 63).
- **Written rules instead of model judgement.** A tie-break rule decides emails that cover more than one topic, and five rules decide escalation. *Trade-off:* rules are predictable and testable, but they miss situations they don't name.
- **Small, low-cost models**, because classification shouldn't need a frontier model.
- **Plain Python and the official SDKs**, with no framework. Results are saved, so metrics can be recalculated without new API calls.

## Model Choice & Cost

**Provisional choice: Claude Haiku 5.5, if a missed escalation costs more than an unnecessary review.**
- **Why Claude:** it missed none of the 21 emails that needed a person; GPT-6 Luna missed 2.
- **Why the evidence is thin:** 21 emails needing escalation is too few for a firm conclusion.
- **When GPT would win:** if reviewer time is the tighter constraint, GPT-6 Luna's higher precision and ~40% lower cost would matter more.

**Estimated production cost:**

| Model | Tokens per email (input / output) | Per 1,000 emails | Per million emails |
|---|---:|---:|---:|
| Claude Haiku 5.5 | 945 / 93 | **$0.14** | $140 |
| GPT-6 Luna | 497 / 63 | $0.08 | $80 |

- **How it's calculated:** measured token counts × list prices as of October 2026. Both models list at $0.10 per million input tokens and $0.50 per million output tokens.
- **Why Claude costs more anyway:** it used about 1.9× the input tokens and wrote longer explanations.
- **What would change it:** real emails are usually longer, which would raise the cost; batch or caching discounts would lower it.

## Results

All 100 emails, one run per model:

| Metric | GPT-6 Luna | Claude Haiku 5.5 |
|---|---:|---:|
| Category accuracy | 100% (100/100) | 100% (100/100) |
| Escalation recall | 90.5% (19/21) | **100% (21/21)** |
| Escalation precision | **95.0% (19/20)** | 91.3% (21/23) |
| Missed escalations (false negatives) | 2 | **0** |
| Unnecessary escalations (false positives) | **1** | 2 |
| Invalid responses | 0 | 0 |
| Median latency | 1.66 s | **1.48 s** |
| Est. cost per 1,000 emails | **$0.08** | $0.14 |

- **GPT missed escalations** on emails 20 and 63.
- **Unnecessary escalations:** Claude on emails 13 and 93; GPT on email 93.
- **Email 93 is a gap in the rules.** It's a bereavement case that no rule covers, and both models escalated it.
- **Dataset sizes:** the original set has 8 emails that should escalate, and the harder set has 13. Total API spend was $0.02.

**What the results show:**
- **Escalation was where the models differed.** Both classified every email correctly, including indirect wording, emails with more than one request, and slang.
- **The models' mistakes lean in opposite directions.** GPT applied the rules more literally and missed cases. Claude was more cautious and over-escalated.
- **Both ignored the injection attempts.** Both followed their instructions despite the three attempts embedded in emails 68, 84 and 98. Three cases don't show general resistance to prompt injection.

**What they don't tell you:**
- **Which model is better at categories.** Both scored 100%.
- **Whether the escalation gap is real.** With 21 emails that should escalate, one email moves recall by about 5 points.
- **Real-world performance.** The emails are synthetic and cleaner than real customer mail.
- **Unbiased performance.** The emails were drafted with help from a Claude model, which could favour Claude Haiku. All labels come from one person.
- **Variation between runs.** Each dataset was run once, so this wasn't measured.

## Where It's Weakest

**Valid JSON can still contain an inconsistent decision.** On email 63 (two disputed charges totalling $100.50), GPT-6 Luna returned:

```json
{
  "category": "billing",
  "escalate_to_human": false,
  "reasoning": "The customer wants $100.50 refunded for disputed call charges, so this is billing and exceeds the $100 escalation threshold."
}
```

- **Why it matters:** the reasoning says the threshold is exceeded, but the decision says don't escalate. Both fields are individually valid, so validation can't catch it.
- **A possible cause:** the schema asks for the decision before the reasoning. This is untested.

**Other weak spots:**
- **The $100 rule is ambiguous.** It doesn't say whether it means the amount charged so far or what the total could grow to. That ambiguity caused both mistakes on the original set (emails 13 and 20).
- **The rules don't cover everything that needs a person.** The bereavement email (93) is an example.
- **Category weak spots are unknown.** No category mistakes in 100 emails means none were found, not that there are none.

## Before Real Customers

- **Clarify the $100 rule:** count the amount charged so far, and state how to total recurring charges.
- **Add targeted regression cases.** Re-test the known failures (emails 13, 20, 63 and 93) and amounts either side of $100. Test any prompt change on fresh emails, not the ones used to design it.
- **Treat field order as an experiment.** Putting the reasoning before the decision might reduce contradictions. Adopt it only if it does on new emails.
- **Evaluate on real data:** a few hundred anonymised emails labelled by support staff, with an agreed target for escalation recall.
- **Start in shadow mode:** the model suggests, staff decide, and disagreements are reviewed before anything is routed automatically.
- **Protect privacy and monitor it:**
  - mask personal details before calling a third-party API
  - confirm each provider's data retention terms
  - track missed escalations, errors, latency and cost

## Run It

```bash
pip install -r requirements.txt
cp .env.example .env                        # add OPENAI_API_KEY and ANTHROPIC_API_KEY
pytest                                      # 31 offline tests, no API calls
python evaluator.py --dataset hard --from-results   # re-print saved results for free
python evaluator.py                         # re-run original 60 emails (overwrites saved results)
python evaluator.py --dataset hard          # re-run harder 40 emails (overwrites saved results)
```

`classifier.py` makes the API calls and validates responses. `evaluator.py` runs the comparison and calculates the metrics. Datasets are in `data/` and saved results in `results/`.
