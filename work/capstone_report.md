# Capstone Report — <your lane>

- **Author:** Timothy Karhnak-Glasby
- **Lane:**
- **Repo:** FlyRank-ML-Track
- **Date:** 9/5/2026

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

The baseline is two signals, combined: has this page held a decent search position (top 10) while getting zero clicks, and has its position gotten worse from the first week of the month to the second. Each is a yes/no flag; a page scores 0–3 depending on how many apply, weighted so the zero-click signal counts double. No training, no parameters — just two conditions an editor could check by eye, which is the point: it's the formalization of what a careful reviewer already does manually, not a strawman built to lose.

Ranked by that score (ties broken by traffic volume) and measured with the same precision@K used for the model, on the same five folds: 0.490 at K=20, 0.420 at K=50, 0.412 at K=100, 0.433 at K=200. The model's corresponding numbers are 0.530, 0.556, 0.580, and 0.577 — a lead at every depth tested, and a clean sweep of all five folds specifically at K=50 and K=100.

One asymmetry in this comparison is worth stating plainly. Of the two signals, only the zero-click flag uses full-month data — the position-trend flag is a week-1-vs-week-2 comparison already confined to the first half, matching the model's window without any change needed. The zero-click flag's fuller window exists because the rule predates the model's own first-half-only restriction, introduced later specifically because full-month features leaked the label almost completely when tried as model inputs. The rule was checked for that same leak when it was built, and correctly cleared — but its definition was never revisited once the model's window tightened around it.

Whether that gap favors the model was worth checking rather than assuming. Holding everything else fixed and swapping in a first-half-only version of the zero-click flag made the rule's precision@K worse at every depth, not better (dropping 0.06–0.15 across K=20 to K=200, e.g. 0.42 → 0.34 at K=50). So where the mismatch has an effect at all, it works against the model's apparent advantage — meaning the comparison above, if imperfect on paper, is fair in the direction that matters: it doesn't inflate the model's win.

## 4. Model / analysis

This project is a Refresh / Content Opportunity Scoring problem — scoring pages for review priority, giving a reason each one was flagged, and suggesting an action to take — scoped specifically to pages that are declining rather than the fuller growing-recovering-declining spectrum a wider opportunity-scoring system might track. A logistic regression was tried first, since the label is a plain binary outcome. It lost to the baseline rule at every queue depth, and lost outright to random guessing unless the rule's own position-and-clicks interaction was handed to it as an explicit extra feature. A random forest doesn't need that help — it finds the same interaction on its own, by splitting on one signal and then the other. That's why the model in this pipeline is tree-based.

Each of the five fold models is shallow and leaf-floored on purpose: 300 trees, depth 6, a 20-row minimum per leaf. That's sized for a population in the hundreds of thousands, close to a bar the baseline rule could plausibly clear. Six features feed the model, all built from first-half-only search data: average position, the position change from week 1 to week 2 of the first half, log-scaled impressions, log-scaled clicks, click-through rate, and a flag for whether a page had position data in both weeks at all. Log-scaled impressions carries the most weight, with average position and click-through rate close behind it. Position change and the two remaining features trail further back.

Two things were left out on purpose. The rule's own flag, top-10 position with zero clicks rebuilt on first-half-only data, was constructed and tested but never added as a model feature — the tree-based model already finds that combination through its own splits, so handing it in explicitly would solve a problem the model doesn't have. The model also never sees a binary zero-clicks signal the way the rule does. It only sees the continuous click-through rate and click volume behind that signal, so a page's click behavior reaches the model as a spectrum rather than a threshold call. A page one point above the rule's zero-click floor and a page far above it look identical to the rule, and only somewhat different to the model.

A page counts as declining if its total search impressions were lower in the second half of the scored month than in the first half, among pages with any first-half impressions at all. This is a proxy for content decay, covering only pages that are already declining.

## 5. Evaluation

The model only sees the first half of the month; the label lives in the second half. That's a real temporal boundary, though a different thing from a time-based cross-validation split, which would need a third, later window to hold out for testing. The only later window here is the second half, already spent defining the label. Folds are grouped by client instead. This guards against a model that memorizes one site's quirks and looks artificially strong when that site appears on both sides of a split. Five folds, GroupKFold, no client in more than one fold's test set.

The deliverable is a ranked queue, so the metric is precision@K: what share of the top K pages actually declined. A model that scores well by other measures but ranks its best pages badly would still fail the job it's for. On the same five folds, the model beats the rule on average across every depth tested, K=20 through K=200. That average hides real variation. The win is a clean sweep across all five folds only at K=50 and K=100. At K=20 the rule wins two of five folds; at K=200 it wins one. The advantage is real but uneven across queue length, which is part of why the project's operating assumption is anchored specifically at K=50.

Where the model disagrees with the rule, the disagreement isn't random. Its scores rise in the same order the rule's own severity tiers do — lowest on pages the rule ignores, highest on pages the rule flags most severely — even on pages where the two land on different actions.

One failure mode that turned up during development is that a page whose position swings wildly can push the model's score up even if that page was never ranking well to begin with, something the rule's simple worse/not-worse check would never catch. A caution flag built for exactly this case — position swings in the top 10% by size — fires on 9 of the 7,286 pages the model catches that the rule misses entirely. It's rare in the delivered queue, but a real blind spot the rule doesn't share.

## 6. Interpretation

Three features carry most of the model's decisions, in a fairly even split: how many impressions a page gets, where it ranks, and its click-through rate. Week-over-week position change and the flag for whether trend data even exists matter less. Log clicks alone matters least, behind click-through rate, suggesting the rate a page converts impressions into clicks carries more signal than its raw click count.

One of those three, impression volume, has no equivalent in the baseline rule at all. The rule only ever checks position and clicks; it has no way to treat "the page is losing overall visibility" as a graded signal, only a discrete eligibility cutoff. That's the single feature the model leans on most, and it's the one thing structurally invisible to the rule.

A tier meant to flag reliable model output turned out backwards. Rows the model was least sure about had a higher real decline rate than rows it was most sure about, in every archetype. It was renamed to coverage, since data volume is what it actually measures.

One signal was tested and deliberately left out for a different reason. Pages with weaker session engagement were expected to decline more often however it was instead found that pages below the median engagement rate declined less often than pages above it, 35.4 percent versus 43.9 percent. The reason surfaced on a second look: pages with fewer impressions naturally inflate that same engagement ratio and independently decline more often on their own, so two separate effects were tangled into one number rather than pointing at anything real. It's a genuine limit of what the available signals can measure — a page that gets clicks but loses readers immediately has no safe way to be detected here.

The model's win at K=50 is real but concentrated. zero_clicks_and_worsened pages score highest of any archetype on average, yet supply almost none of the top 50. position_worsened_only, lower-scoring on average, fills most of it instead. Position change ranks near the bottom of feature importance, so a favored feature doesn't explain this. Population size does — position_worsened_only is roughly five times as populous, and that size fills the top 50 regardless of average score. The advantage documented here is currently a visibility-loss story more than a click-through-loss one.

About a fifth of the pages scored have no rule signal at all. No valid week-over-week position comparison exists for them, so they're treated as non-worsening by default rather than flagged. Most of that fifth gets no rule-based flag whatsoever and depends entirely on the model to be noticed. Whatever the model gets right or wrong there, the rule was never going to catch it either way.

## 7. Recommendation

Everything below describes how to use the output of a proof-of-concept. The queue that exists right now, for the pages already scored this month, is ready to hand to a reviewer. Turning it into something that regenerates itself every month is a separate step, covered below.

An editor works down the queue in ranked order, archetype and suggested action attached to every page: refresh the content, overhaul its title and meta, do both, flag for manual review, or monitor. Each of these is a starting point for a person to look at the actual page before acting.

zero_clicks_and_worsened pages score highest on average of any archetype, 0.749, but rank position won't surface most of them. The model's very top ranks are dominated by the much larger position_worsened_only group, so working strictly top-down turns up almost none of this higher-scoring archetype in the first 50 to 100 reviews. A reviewer who wants these caught should filter for the archetype directly rather than rely on rank order.

The other two archetypes have their own tradeoffs. no_flag pages, the largest group at 43.5 percent and the lowest-scoring, are built for monitoring rather than routine review. model_only_catch pages carry no rule backing at all, but the caution flag built for the model's one known failure mode, a large position swing mistaken for real decline, fires on only 9 of 7,286 of them, so most are worth the look despite the missing backup.

Coverage measures how much data backs a score, and it's worth reading as a cost signal. Low-coverage pages, about a quarter of the queue, carry thinner evidence and deserve closer scrutiny before acting. The tier was renamed from confidence once examination revealed that low-coverage rows turned out to have a higher real decline rate than high-coverage rows in every archetype, so a low-coverage flag calls for a careful read regardless of priority.

How much to trust the model's edge over the rule isn't the same at every queue depth. It's most solid around 50 to 100 pages a month, where the model beat the rule in every fold tested. At the very top, the first 20 pages, the rule won two of five folds, worth extra scrutiny before trusting rank order alone.

This only works for the March 2026 population already scored. A new page next month doesn't belong to any of the five folds this queue was built from, so reusing these scores on it isn't valid. Generalizing to another month isn't automatic either. The month and the two boundary dates derived from it are hardcoded constants rather than a parameter the pipeline accepts. Pointing this at a different month currently means editing those constants by hand and rerunning everything from scratch. That's a small, well-scoped change — accept a month argument, derive the boundary dates from it — but it hasn't been made or tested yet.

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
