# ML-10 Content Action Playbook Report

Scored population: 150,675 pages, impressions before vs after 2026-03-16, month=2026-03, `is_declining_proxy` label.

Queue is ranked by the calibrated model score alone, no rule-based nudge (see
work/outputs/fold_representation_check.json and work/outputs/capstone_precision_at_k.json
for why — every combination method tested reintroduced fold skew, so the queue uses the
model score directly).

## Archetype breakdown

| Archetype | Count | Action |
|---|---:|---|
| `no_flag` | 65,570 | Monitor |
| `position_worsened_only` | 52,753 | Refresh content |
| `zero_clicks_only` | 14,457 | Overhaul title & meta |
| `zero_clicks_and_worsened` | 10,609 | Refresh content + overhaul title/meta |
| `model_only_catch` | 7,286 | Flag for manual review |

## Confidence split

- High: 111,980
- Low: 38,695 (25.7%)

## Rule-agreement check (model score vs. rule severity, rule-only archetypes)

| Archetype | Mean oof_rf_score (calibrated) |
|---|---:|
| `no_flag` | 0.386 |
| `position_worsened_only` | 0.492 |
| `zero_clicks_and_worsened` | 0.749 |
| `zero_clicks_only` | 0.642 |

## model_only_catch caution flag

9 of 7286 rows (0.1%) flagged for large position-swing caution.

## Top 20 queue preview

| content_hash_id          | archetype              | action                 | confidence   |   oof_rf_score |
|:-------------------------|:-----------------------|:-----------------------|:-------------|---------------:|
| content_334bcb2761d0f9c7 | model_only_catch       | Flag for manual review | high         |       1        |
| content_39457d17e716086c | position_worsened_only | Refresh content        | high         |       1        |
| content_f98166a30c643b7c | position_worsened_only | Refresh content        | high         |       1        |
| content_3e56218fa52d24b8 | position_worsened_only | Refresh content        | high         |       1        |
| content_c63784807a77c864 | position_worsened_only | Refresh content        | high         |       1        |
| content_7b37f22d2086ca56 | position_worsened_only | Refresh content        | high         |       0.999967 |
| content_9bcfb1e373c01b7a | position_worsened_only | Refresh content        | high         |       0.999967 |
| content_83a700e06cf9e676 | model_only_catch       | Flag for manual review | high         |       0.999967 |
| content_2dc954b9ae28df53 | position_worsened_only | Refresh content        | high         |       0.999967 |
| content_a8440d564c0facd9 | position_worsened_only | Refresh content        | high         |       0.999967 |
| content_9ad3c018b18825d9 | position_worsened_only | Refresh content        | high         |       0.999934 |
| content_8116752652923ef3 | model_only_catch       | Flag for manual review | high         |       0.999934 |
| content_bab284527da6960a | model_only_catch       | Flag for manual review | high         |       0.999934 |
| content_391ec530eb49b188 | position_worsened_only | Refresh content        | high         |       0.999934 |
| content_2e698ab32f21b07c | position_worsened_only | Refresh content        | high         |       0.999934 |
| content_913f91b1e9713a10 | position_worsened_only | Refresh content        | high         |       0.999901 |
| content_52ea1e7d6159d54b | model_only_catch       | Flag for manual review | low          |       0.9999   |
| content_bfd481a760fa3064 | model_only_catch       | Flag for manual review | high         |       0.9999   |
| content_d1db17521a55d9fc | model_only_catch       | Flag for manual review | high         |       0.9999   |
| content_57d6d64a175bb7ce | position_worsened_only | Refresh content        | high         |       0.9999   |

## Practical use

Use this queue as a reviewer aid, not an automatic action trigger. Every action here is a starting point for review, not an instruction to execute unread. 
See the source notebook (work/notebooks/w07_action_playbook.ipynb) for intended use, limits, human review rules, the no-go list, and monitoring/retrain triggers.

## Generated files

- `work/outputs/w07_metrics.json`
- `work/outputs/charts/archetype_mix.svg`
- `work/outputs/charts/confidence_mix.svg`
- `work/data/processed/w07_ranked_queue.csv` (local only — gitignored, not committed;
  regenerate by rerunning work/scripts/run_all.py rather than expecting this file to
  exist in a fresh clone)
