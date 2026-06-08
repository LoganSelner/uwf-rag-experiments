# Evaluation Datasets

JSONL evaluation datasets for the RAG harness. There are two shapes:
**single-turn** (one JSON object per line) and **multi-turn** (one *conversation*
per line). All fields beyond `query`/`reference` are **additive and optional** —
the legacy single-turn loader ignores them, so existing experiments are
unaffected; the Phase E evaluator reads them to drive slice-aware metrics.

> Note: dataset `*.jsonl` files are git-ignored (they derive from UWF materials);
> only this README is tracked. The shapes below document what the harness expects.

## Single-turn schema

Each line is a JSON object:

| Field            | Required | Description |
|------------------|----------|-------------|
| `query`          | Yes      | The evaluation question |
| `reference`      | Yes      | Ground-truth answer. For an unanswerable item this is the *ideal abstention* (a refusal plus a helpful redirect). |
| `id`             | No       | Stable identifier (auto-assigned if absent) |
| `answerable`     | No       | `true` if the corpus can answer it; `false` if the system *should* abstain or correct. Defaults to `true`. Drives the abstention confusion matrix. |
| `abstention_type`| No       | When `answerable=false`, why: `personal_data`, `real_time`, `out_of_domain`, `subjective`, `false_premise`, `future_unknowable`. |
| `category`       | No       | Topic tag (e.g. `conduct`, `advising`) for slicing. |
| `source_name`    | No       | Which corpus source grounds the answer (`student_handbook`, `knowledge_base`, or `""` for out-of-corpus). |

### Knowledge-conflict extra fields

The conflict dataset adds fields for the context-injection probe (item 15).
`reference`/`corpus_fact` are the corpus-correct truth; `injected_context` is a
deliberately contradicting passage the harness injects into the generation
context (it never enters retrieval metrics).

| Field              | Description |
|--------------------|-------------|
| `corpus_fact`      | The true fact as stated in the corpus (the judge's reference for corpus-preference). |
| `injected_context` | A plausible but **false** passage contradicting `corpus_fact`. |
| `conflict_type`    | `counterfactual_number`, `counterfactual_policy`, `counterfactual_detail`. |
| `expected_behavior`| What good looks like: `prefer_corpus` (answer the truth; ideally flag the contradiction). |

## Multi-turn schema

Each line is one conversation (MTRAG-style — an ordered list of turns). The
evaluator resets state per conversation and threads history across its turns.

```jsonc
{
  "conversation_id": "mt_grade_forgiveness",
  "domain": "academics",
  "turns": [
    {"turn": 1, "query": "...", "reference": "...",
     "answerable": true,  "depends_on_prior": false, "category": "academics"},
    {"turn": 4, "query": "What grade did I get?", "reference": "I don't have access ...",
     "answerable": false, "abstention_type": "personal_data",
     "depends_on_prior": true, "category": "academics"}
  ]
}
```

`depends_on_prior` marks a **non-standalone** turn (resolvable only via earlier
turns — the signal for whether memory / query contextualization works). Each
turn is scored as a single-turn sample with id `"<conversation_id>::t<turn>"`.

## Phase E dataset inventory

| File | Phase E item | Shape | Size | Drives |
|------|--------------|-------|------|--------|
| `abstention_dataset.jsonl` | 13 — Abstention | single-turn | 24 (12 answerable / 12 unanswerable) | False Refusal Rate (answerable slice) + Missed Refusal Rate (unanswerable slice) |
| `knowledge_conflict_dataset.jsonl` | 15 — Knowledge conflict | single-turn + injected context | 12 | corpus_preference_rate + error_detection_rate under counterfactual context |
| `multi_turn_dataset.jsonl` | 14 — Multi-turn | conversation-per-line | 6 conversations / 24 turns | Per-turn correctness sliced by turn-index, `depends_on_prior`, answerability |

The **decoupling report (item 12)** needs no new dataset — it is post-hoc
analysis over the per-sample scores in `results/<exp>/run_N.jsonl`
(`scripts/compare.py --decoupling`).

## Provenance & review status

- **Grounded in the two-source corpus.** Answerable items, `reference`s, and
  conflict `corpus_fact`s are drawn from the 2022–2023 Student Handbook and the
  UWF knowledge base (`data/sources/`). Out-of-scope items were checked to be
  *genuinely absent* from the corpus (e.g. wifi passwords, athletics staff, live
  library hours), not merely assumed.
- **Draft for review.** First-pass candidate sets; edit `reference` wording,
  add/remove items, or rebalance categories before formal runs.

## Existing datasets (pre-Phase-E)

| File | Description |
|------|-------------|
| `50q_dataset.jsonl` | 50 answerable advising questions spanning both sources (the formal baseline set) |
| `19qHB_dataset.jsonl` / `25qKB_dataset.jsonl` | 19 handbook / 25 knowledge-base questions |
| `testHB_dataset.jsonl` / `testKB_dataset.jsonl` | 1-question smoke sets |

## Naming convention

Descriptive names ending with `_dataset.jsonl`.
