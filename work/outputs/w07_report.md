# ML-10 Content Action Playbook Report

Scored population: 150,675 pages, March 2026, `is_declining_proxy` label.

## Archetype breakdown

| Archetype | Count | Action |
|---|---:|---|
| `no_flag` | 65,567 | Monitor |
| `position_worsened_only` | 52,753 | Refresh content |
| `zero_clicks_only` | 14,457 | Overhaul title & meta |
| `zero_clicks_and_worsened` | 10,609 | Refresh content + overhaul title/meta |
| `model_only_catch` | 7,289 | Flag for manual review |

## Confidence split

- High: 111,255
- Low: 39,420 (26.2%)

## Rule-agreement check (model score vs. rule severity, rule-only archetypes)

| Archetype | Mean oof_rf_score |
|---|---:|
| `no_flag` | 0.401 |
| `position_worsened_only` | 0.432 |
| `zero_clicks_and_worsened` | 0.494 |
| `zero_clicks_only` | 0.472 |

## model_only_catch caution flag

42 of 7289 rows (0.6%) flagged for large position-swing caution.

## Top 20 queue preview

| content_hash_id          | archetype              | action                 | confidence   |   combined_score |
|:-------------------------|:-----------------------|:-----------------------|:-------------|-----------------:|
| content_403a36188e13fcc8 | position_worsened_only | Refresh content        | high         |         0.828892 |
| content_2435b8bb25eeebd9 | position_worsened_only | Refresh content        | high         |         0.824912 |
| content_df977de3b77ec57c | position_worsened_only | Refresh content        | high         |         0.822337 |
| content_b5a91be0a10cd899 | position_worsened_only | Refresh content        | high         |         0.819795 |
| content_4e48bd81bb37eb4f | position_worsened_only | Refresh content        | high         |         0.81735  |
| content_5effb301ded55c21 | position_worsened_only | Refresh content        | high         |         0.810445 |
| content_334bcb2761d0f9c7 | model_only_catch       | Flag for manual review | high         |         0.806296 |
| content_83a700e06cf9e676 | model_only_catch       | Flag for manual review | high         |         0.804104 |
| content_bab284527da6960a | model_only_catch       | Flag for manual review | high         |         0.804002 |
| content_f5e2cda099b4321c | position_worsened_only | Refresh content        | high         |         0.803841 |
| content_bfd481a760fa3064 | model_only_catch       | Flag for manual review | high         |         0.801034 |
| content_a11bd5663919f057 | model_only_catch       | Flag for manual review | high         |         0.796557 |
| content_10e8f76ed8c5c392 | position_worsened_only | Refresh content        | high         |         0.791671 |
| content_9d72beee0dffbc0f | model_only_catch       | Flag for manual review | high         |         0.788793 |
| content_dbdc47a245e38862 | position_worsened_only | Refresh content        | high         |         0.787441 |
| content_c531fc2673be69f1 | position_worsened_only | Refresh content        | high         |         0.785265 |
| content_47da45b084a73115 | position_worsened_only | Refresh content        | high         |         0.784826 |
| content_19412009bb676d79 | model_only_catch       | Flag for manual review | high         |         0.775432 |
| content_79b61512984892b6 | position_worsened_only | Refresh content        | high         |         0.772896 |
| content_39457d17e716086c | position_worsened_only | Refresh content        | high         |         0.770367 |

## Practical use

Use this queue as a reviewer aid, not an automatic action trigger. See Section 2 (intended use and limits) and Section 3 (human review and no-go list) in the notebook for the full detail — every action here is a starting point for review, not an instruction to execute unread.

## Generated files

- `work/outputs/w07_metrics.json`
- `work/outputs/charts/archetype_mix.svg`
- `work/outputs/charts/confidence_mix.svg`
- `work/outputs/w07_ranked_queue.csv` (local only — gitignored, not committed; rebuild from this notebook's Sections 0–1 rather than expecting this file to exist in a fresh clone)
