# Capstone Report — <your lane>

- **Author:**
- **Lane:**
- **Repo:**
- **Date:**

> Copy this file to `work/capstone_report.md` and fill it in as you build. Sections 1–8
> mirror the Pass / Needs-Work rubric axes, so nothing here is optional. Sections 0 and 9
> are **paper sections**: your deployed research paper must carry both, and they're here so
> you never rebuild them from memory at ship time.

## 0. Abstract

Five sentences, written last, placed first: question → data → method → headline result →
what the output is for. This is the top of your deployed paper.

## 1. Problem framing

Organizations that manage content across many sites face a real constraint: a team with a few dozen client sites and hundreds of thousands of pages can't have an editor read all of them every month. Something has to decide which pages get attention first — currently, that decision gets made by rule of thumb, or not at all.

The unit of analysis is a single page, on a single site, in a single month. The output is a ranked queue: for each page, a score, an archetype describing what kind of problem it likely has, and a suggested action — refresh the content, overhaul its title and meta, flag it for manual review, or, for the largest single group, simply monitor. The human in the loop is a content editor, who takes the top of the queue as their to-do list for the month.

Getting this wrong costs in both directions. Send an editor's limited hours to pages that were never actually declining, and the pages that needed help don't get reviewed. Send them somewhere too conservative, and real declines sit unreviewed because nothing surfaced them.

Given a fixed review budget: which declining pages should be reviewed first, and does a model do a better job answering that than a hand-built rule? Manual review doesn't scale to a portfolio this size, and search performance data — position trends, click-through patterns, impression volume — carries enough of the actual signal that a rule can take a first pass at ranking it. Whether a model finds more of that signal than the rule does, under an honest test, is what this work checks.

What follows tests that approach's viability: model-based ranking beats the hand-built rule on one month of historical data, under grouped cross-validation. It doesn't yet score new pages on its own every month — that gap is covered in Limitations and Recommendation.

## 2. Data safety

This work draws on one table from the FlyRank internship warehouse: fact_content_daily_performance, the March 2026 partition, at the report_date × client × content grain. The month splits at March 16 into a first half and second half; the first half further splits at March 8 into two weeks, to measure a within-first-half position trend. Every feature and the label are built from this one table and these boundaries — no other warehouse table contributes.

Two joins were tried and dropped. A join to the warehouse's content-metadata table brought in word count and publication status — dropped because that table is a current-state snapshot, risking information about the content's state now rather than its state when scored. A join to session data brought in engagement metrics — dropped because it's a whole-site denominator being asked to explain a single search channel's decline, and it pointed the wrong direction when tested. The final feature set is native to search performance: position, impressions, clicks, and how they move within the first half of the month.

A separate starter dataset in this project ships two fields, trend_direction and trend_pct, derived from the same later-window comparison the label would need to predict — a clear leak if used as features. Neither field, nor that dataset, is used anywhere in this pipeline; the label here is built independently from impression totals in the two halves of the month.

One leak did happen and was fixed: a full-month impression total and a full-month eligibility flag were briefly used as features, and both leaked the label, since the full-month total partly consists of the same second-half impressions the label is built from. Both were removed from the feature set. The eligibility flag still exists in the data — it gates which pages have enough traffic to be scored, and backs a data-volume tier — but is not passed to the model.

Two pseudonymous hashes run through every table and output: a client ID, used only to group pages for evaluation so a model is never tested on a client it trained on, and a content ID, used only to identify rows. Neither is a feature. Every notebook, script, JSON, CSV, and chart in work/ contains only these hashes and the metrics computed from them — nothing client-identifying appears anywhere in it.

## 3. Baseline

The transparent rule or score you built first. Why it's a fair comparison, and its numbers on
the same data and metric as your model.

## 4. Model / analysis

Your method and why it fits the lane. The exact feature list (and what you left out on
purpose). The target or proxy definition, in one sentence.

## 5. Evaluation

Your split (grouped by client? time-aware?) and why. Metrics, model vs baseline **on the same
split**. What the errors look like — a short error analysis beats a big metric table.

## 6. Interpretation

What the model/clusters actually found. Feature importances or cluster profiles in plain
words. Surprises and negative results — a well-understood "no effect" is a valid result.

## 7. Recommendation

The ranked actions or decisions your output supports, and how a FlyRank editor would use them
tomorrow. State your confidence and the limits explicitly.

## 8. Reproducibility

The exact commands to re-run everything from a fresh clone, your random seeds, and your
environment (`pip freeze` highlights or `requirements.txt` deltas). If you claim a sealed or
holdout evaluation, two things must be committed: the cell/script that builds the sealed
frame, and the metrics file it produced — "evaluated once, blind" should be checkable from
your repo, not taken on faith.

## 9. Acknowledgments & data credit

One short section at the bottom of the deployed paper: "Built on the FlyRank ML Internship
dataset" **linking to https://flyrank.ai**. Crediting your data source is standard research
practice — and it's on the capstone's required-section list, so a paper without it isn't done.

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> **Metrics vs. base rate:** report your task's base rate (majority-class %) next to any
> precision@K or accuracy — a high score can just be a high base rate. AUC / lift over
> baseline are the honest discrimination numbers.
> language everywhere · no causal claims without an experiment or causal design · no
> "predicted Google's algorithm" · no client-identifying details · numbers in this report
> match a fresh re-run.
