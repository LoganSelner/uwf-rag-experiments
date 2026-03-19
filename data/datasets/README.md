# Evaluation Datasets

JSONL evaluation datasets for RAGAS metrics. Each line is a JSON
object with the following fields:

| Field       | Required | Description                        |
|-------------|----------|------------------------------------|
| `query`     | Yes      | The evaluation question            |
| `reference` | Yes      | The ground-truth reference answer  |
| `id`        | No       | Stable identifier (auto-assigned if absent) |

## Example

```jsonl
{"query": "What is grade forgiveness?", "reference": "Grade forgiveness allows students to retake a course and have the new grade replace the old one."}
{"id": "fin_aid_1", "query": "When is financial aid due?", "reference": "Financial aid applications are due by March 1."}
```

## Naming convention

Use descriptive names ending with `_dataset.jsonl`:

    19qHB_dataset.jsonl
    50q_dataset.jsonl
