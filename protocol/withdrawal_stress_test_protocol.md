# Cycle 8: withdrawal stress test — problem definition and pre-registered protocol (version 2)

Date: 2026-11-13. Follows `reports/Cycle 8 sovereignty topic discovery.md` (decision B). That report is unchanged.

**Status: not run.** No outcome data have been analysed. The only real-data work so far is feasibility checks of the public files, marked [DATA]: file access, question alignment, prompt identity, and model counts. Version 1 of this protocol was reviewed adversarially (`research_notes/Cycle 8 sovereignty discovery/h_review_4_protocol.md`), judged "not ready", and rewritten. The changes are listed in Section 10.

Evidence tags: [FT], [FT-part], [ABS], [SN], [REC], [OWN], [DATA].

## 1. Research question

A decision-support system answers questions by aggregating several AI assistants. Groups of assistants can be lost together: all models of one developer, or all models reachable only through a foreign hosted API. The question has three parts:

1. If hosted foreign APIs are withdrawn, how much does a panel of locally deployable open-weight models retain?
2. How should such a panel and its aggregation rule be chosen?
3. What accuracy and error correlation must a new (for example domestic) model have to be worth adding?

This makes operational autonomy, the ability to act when components are withdrawn (Edler et al. 2023), a measured property of a system. It is not proposed as an index.

## 2. Prior art and what is new

| Work | What it establishes | What it does not do |
|---|---|---|
| Kim, Garg, Peng, Garg, arXiv 2506.07962 (ICML 2025) [FT-part] | Error agreement among 349 + 71 LLMs; same company and architecture raise it | No ensembles, no withdrawal |
| Turkmen, Buyukates, Bastopcu, arXiv 2602.08003 (2026) [ABS, FT-part] | Budgeted LLM ensemble selection by greedy mutual information; error floor from correlated errors; MMLU | Optimises for full availability only |
| Kohli, arXiv 2605.29800 (2026) [ABS] | Nine LLM judges ≈ 2 effective votes; aggregation closes at most 11% of the gap | Judges, no withdrawal |
| Denisov-Blanch et al., arXiv 2603.06612 (2026) [ABS] | Polling-style aggregation fails for truthfulness because errors are correlated | No withdrawal |
| Goel et al., arXiv 2502.04313 (ICML 2025) [FT-part] | Chance-adjusted probabilistic agreement (CAPA) | No withdrawal |
| Reliability importance measures (Birnbaum 1969; risk achievement worth; joint importance, Hong and Lie 1993; Borgonovo and Apostolakis 2001) [SN]/[REC] | Loss with components failed; second-order interaction | Not applied to AI ensembles, as far as found [SN] |
| Classifier ensembles with missing members (dissertation found by the audit agent) [SN] | Stacking degrades and voting does not when members are missing | Not LLMs; unverified |

Established: LLM errors are correlated, so nominal diversity overstates reliability. **New: measurement under group withdrawal against valid null models; selection of panels for robustness, tested on scenarios not used in selection, and compared with a trivial diversity cap; an admission chart for new models.** The contribution is empirical and methodological. The measures are borrowed and attributed; no theorem is claimed. The prior-art search is mostly snippet-level, and ErrorTrace (NeurIPS 2025) was not found or read.

## 3. Data (feasibility checked, outcomes not analysed)

- **Main benchmark: HELM MMLU** (`crfm-helm-public/mmlu`). It records per-question answers and the correct key for 81 models from 18 developers across releases v1.0 to v1.13, and needs no account [DATA]. HELM documents public download of its raw results; its code is Apache-2.0; a separate data licence was not found [DATA].
- **Checks already made [DATA]:**
  - 57 subjects are present for the models checked;
  - question identifiers and texts match across releases for three models (100 of 100 on one subject);
  - the 5-shot prompt is identical for three of four models checked, but differs in length for one base model (Llama 3 8B);
  - decoding uses temperature 0 and one output token.
- **Replication benchmark: HELM Capabilities MMLU-Pro** (10 options). Runs exist for 70 models, newer than the MMLU set and including Llama 4, Qwen3, Kimi K2 and gpt-oss [DATA]. Whether all items or a subsample are scored is to be checked.
- **Step 0 of S1, before any outcome is computed:**
  1. Require every model to have all items of the benchmark; drop models that do not.
  2. Hash each model's prompt for every item. The primary analysis uses only models whose prompts equal the modal prompt for every item. All models are used in a sensitivity analysis.
  3. Record release, model type (base or instruction-tuned), and hosting path (direct API or third-party endpoint).
  4. Deduplicate to one model per lineage and size class, keeping the latest snapshot, by a rule written down before outcomes are seen. Example: GPT-4o 2024-05-13 and 2024-08-06 are one lineage and one size class.
  5. Classify each model as open-weight or API-only from its model card, with the source recorded.

## 4. Definitions

- Items k = 1..N with correct answers y_k; assistants i in the panel; answer a_ik; correctness c_ik.
- **Aggregation rules:**
  1. best single assistant (selection-split accuracy);
  2. plurality vote, with ties broken by the highest selection-split accuracy among the tied options' voters;
  3. weighted vote with weights log((m−1)p_i/(1−p_i)), with p_i clipped to [1/m + 0.01, 0.99]. This is optimal only under independence with equiprobable wrong options, which real errors violate, so it is a baseline;
  4. Dawid-Skene with per-model confusion matrices, fitted only if at least three assistants remain, otherwise plurality;
  5. stacking (multinomial logistic regression on one-hot answers).
- **Withdrawal.** The system uses the rule on the remaining assistants, re-fitted on the selection split.
  - The frozen mode (parameters kept) is reported for the weighted vote and Dawid-Skene only. For the other rules frozen and re-fitted coincide or are implementation artefacts.
  - Frozen "best single" means the next best by selection-split accuracy.
  - If no assistant remains, the system abstains; error is reported over answered items together with coverage.
- **Measures:**
  - the error increase ΔE_W = E(P \ W) − E(P), the primary measure. It may be negative, because Dawid-Skene and stacking can improve when a member is removed; negative values are reported, not clipped;
  - RAW_W = E(P \ W)/E(P), secondary, because it is unstable when E(P) is small;
  - joint importance JI(W1, W2) = E(P \ (W1∪W2)) − E(P \ W1) − E(P \ W2) + E(P). It is positive when losing both groups costs more than the sum, meaning the groups cover for each other. It is negative when, once one group is lost, the other matters less. This is an analogue on error of the Hong-Lie measure, not their definition;
  - worst-case error over a scenario family, with the scenario that attains it reported;
  - expected error with equal scenario weights, secondary (market-share weights are not available).
- **Null models,** both fitted on the selection split:
  - N1 (independence): each assistant is correct with its own probability p_i, and wrong answers are uniform over distractors;
  - N2 (conditional independence given the item): a two-parameter item-response model of correctness, with wrong answers drawn from the item's pooled distribution of wrong answers. N2 keeps item difficulty and the global popularity of distractors and removes all other dependence.

  "Excess dependence" is the difference between observed quantities and their N2 values. Version 1's within-item permutation is dropped: it equalises accuracies and is not an independence baseline.

## 5. Hypotheses and primary analysis

**Primary analysis (all else is secondary):**
- panel: the deduplicated, prompt-consistent MMLU models;
- scenario family: withdrawal of each developer, plus withdrawal of all API-only models;
- portfolio size n = 5; aggregation: plurality, re-fitted; λ = 0.5.

The protocol file's SHA-256 hash is recorded in `record.md` before S1 starts.

- **H1 (developer-level shared errors beyond item difficulty).**
  - Setup: on the full reference panel (not a selected portfolio), withdraw each developer with at least two models after deduplication. Compare ΔE with that of matched random sets: the same number of models from other developers, each matched within ±0.02 in accuracy, 1,000 draws.
  - Statistic: the mean over developers of the standardised excess (observed ΔE minus the matched-set mean, divided by the matched-set standard deviation). The null distribution comes from N2-simulated panels.
  - Supported if the one-sided p-value is below 0.05 and the mean excess exceeds the minimal effect that S0 shows is detectable with power 0.8. Refuted if the 90% interval lies below that effect. Otherwise inconclusive.
  - Interpretation: "developer" covers shared data, training, the post-training pipeline and the shared hosting wrapper. These mechanisms cannot be separated with these data, and the paper will say so.
- **H2 (the most accurate rule is not the most robust).**
  - In the re-fitted mode, the rule with the lowest full-availability error differs from the rule with the lowest worst-case error.
  - Both differences must be at least 1 percentage point, with 95% intervals excluding 0, for at least two of the four portfolio sizes (3, 5, 7, 9).
  - Refuted if, at every size, the full-availability winner is also the worst-case winner, or is within 0.5 points of it. Otherwise inconclusive.
- **H3 (robust selection beats a trivial diversity cap on unseen scenarios).**
  - Scenario-robust selection minimises (1−λ)·E(P) + λ·max over the training scenarios. It is evaluated by leave-one-developer-out: select with developer X's withdrawal excluded from the scenario family, then measure the error when X is withdrawn. This is repeated for every X whose models enter any selected portfolio.
  - Comparators: accuracy-greedy, greedy mutual information (Turkmen et al.), and the trivial cap of at most one model per developer, chosen by accuracy.
  - Supported if robust selection lowers the held-out worst-case error relative to the diversity cap by at least 1 point (95% interval excluding 0), with full-availability error no more than 1 point worse.
  - Refuted if the cap is as good or better, within 0.5 points. Otherwise inconclusive.
  - Against accuracy-greedy, the expected large gain is reported but not tested, because it follows from concentration by construction.

H1 to H3 are separate questions. Each has a pre-stated three-way rule. Conjunctions within a hypothesis are intersection-union tests and need no correction. No family-wise correction across H1 to H3 is claimed.

**Descriptive outputs:**
- ΔE, RAW and JI for all scenarios;
- the open-weight-only panel under withdrawal of API-only models, against two bounds: the full panel, and the best single open-weight model;
- a compute-constrained variant: total parameters at most 100B, with open-weight models only;
- the gap between observed values and N1 and N2.

**Admission chart.** Fit the latent model of Section 6 to the open-weight panel. Add simulated candidate models with accuracy a and error correlation r with the panel (conditional on the item), on a grid. Report the gain in panel accuracy. This answers the third question in Section 1. It is a model-based extrapolation and is labelled as such.

## 6. Stage S0: synthetic validation, before any real data

Simulated assistants follow a latent model:

- c_ik = 1{θ_i + u_{d(i),k} + v_{ℓ(i),k} − b_k + ε_ik > 0}, with a developer factor u, a lineage factor v and item difficulty b;
- θ_i is calibrated so that accuracies span 0.25 to 0.90 and correlate with developer;
- wrong answers come from a global distractor preference per item plus a developer-level preference;
- 3% of keys are wrong, a label-error rate taken as a typical value, not estimated;
- developer and lineage sizes copy the real panel after Step 0.

S0 must show three things:

1. **Correct size.** With u = v = 0, the H1 test's false-positive rate is at most 6% at a nominal 5%.
2. **A defined minimal detectable effect.** The smallest mean standardised excess detected with power 0.8; H1 uses it as its margin.
3. **Recovery.** ΔE, JI and the N2 fit recover their simulated values within bootstrap intervals, and the admission chart recovers the simulated gain.

If any of the three fails, the protocol is revised before real data are touched.

**Amendment A1 (2026-11-13, written before S0 was run, no data seen).** Concrete settings and pass criteria for S0, fixed in `checks/cycle8_s0_synthetic_validation.py`:

- **Panel structure.** Step 0 has not been run, so S0 uses a provisional deduplicated structure: 64 models in 18 developer groups, sizes copied from the HELM model counts and reduced to about two thirds for deduplication, with lineages of up to three models. API-only status is assigned by developer. S0 is re-run after Step 0 if the real structure differs by more than 20% in the number of models or of developers with at least two models.
- **Items.** 57 subjects × 246 items = 14,022, split by subject into two halves of 123. H1 and the recovery checks use the evaluation half.
- **Plurality tie-break, changed from Section 4.** Ties are broken by the *sum* of the tied options' voters' selection-split accuracies, not by the highest accuracy. Withdrawal is then a subtraction, which makes the 10⁵ to 10⁶ evaluations of S0 feasible on a CPU. The same rule will be used on real data.
- **N2 for H1.** H1 involves no selection, so N2 is fitted on the evaluation half itself (a parametric bootstrap). This clarifies Section 4, which says that null models are fitted on the selection split; for H1 that split plays no role.
- **H1 test settings.** 100 matched random sets per developer; 100 N2-simulated null panels per test; one-sided p-value (count + 1)/(K + 1); reject at 0.05.
- **Under the alternative,** the developer and lineage factors have the same standard deviation σ, and the weight of the developer-level distractor preference is min(σ, 0.5). Grid: σ ∈ {0.1, 0.2, 0.3, 0.5}. Under the null, σ = 0 and the weight is 0.
- **Criterion 1 (size).** Over 200 null datasets, the rejection rate is at most 0.06.
- **Criterion 2 (minimal detectable effect).** Over 100 datasets per σ, the smallest σ with power at least 0.8 exists. The minimal detectable effect is the mean of the H1 statistic at that σ. If no σ reaches 0.8, the criterion fails.
- **Criterion 3 (recovery).** Measured on 100 datasets at σ = 0.3, for an accuracy-greedy portfolio of five chosen on the selection half:
  - 95% subject-bootstrap intervals (300 resamples) of ΔE (each developer present and the API-only group) and of one joint importance cover their population values in at least 90% of cases. Population values come from 60,000 fresh items with the same models;
  - the Spearman correlation between fitted and true item difficulty is at least 0.85 on average;
  - under σ = 0, the N2-predicted ΔE lies within max(0.005, the interval half-width) of the population ΔE in at least 90% of cases;
  - the admission-chart gain computed with the fitted N2 differs from the gain under the true generator by at most 0.5 percentage points at every grid point (a ∈ {0.5, 0.7, 0.85}, r ∈ {0, 0.3, 0.6}).

## 7. Uncertainty

- Cross-fitting with two folds over items, stratified by subject; selection and fitting are repeated inside each fold.
- A cluster bootstrap over subjects (57 clusters), with selection repeated inside every bootstrap resample.
- For claims about developers, a leave-one-developer-out jackknife, because the finite set of models, not the items, is the dominant source of uncertainty.
- The 20 random halvings of version 1 are dropped: they are not independent replications.

## 8. Sensitivity and replication

- Sensitivity analyses:
  - all models, prompt-inconsistent ones included;
  - without the largest developer;
  - without models below 40% accuracy;
  - excluding items flagged as erroneous in the MMLU-Redux re-annotation, if available [REC, to check].
- Replication of H1 to H3 on HELM MMLU-Pro, for which 70 models are available [DATA].

## 9. What would disprove the value of the study, and limitations

**Negative outcome.** If H1, H2 and H3 are all refuted and the open-weight-only panel keeps nearly all performance, the finding is: *with locally held open weights and a simple diversity cap, hosted-API withdrawal costs little, and nothing more elaborate is needed.* It would be reported as a negative result.

**Limitations:**
- MMLU and MMLU-Pro are English multiple choice. They stand for knowledge questions, not operational decisions.
- Real decision-support systems often combine assistants by discussion or synthesis, not voting.
- The models date from 2023 to 2025.
- Contamination is possible.
- Open-weight models in HELM were often run on hosted endpoints, so their answers may differ from local, quantised deployments.
- Developer country is not where a model is hosted.
- There are no Russian models. The study therefore speaks to the situation of holding foreign open weights locally, not to domestic models; the admission chart is the only bridge to domestic models.
- Running five 70B-class models locally is costly; the compute-constrained variant addresses this only in part.

## 10. Changes from version 1, after review 4

| Review item | Change |
|---|---|
| 1. H1 undefined and ill-posed | H1 runs on the full reference panel, with matched random sets defined model by model and a test statistic with a null distribution |
| 2. Lineage, harness and hosting confounded | Deduplication rule; prompt-hash restriction; hosting recorded; interpretation limited to "developer-level" |
| 3. Invalid independence null | Replaced by N1 (independence) and N2 (conditional independence given the item) |
| 4. H3 true by construction | Held-out scenarios by leave-one-developer-out; the diversity cap is the comparator; the gain over accuracy-greedy is no longer a hypothesis |
| 5. H2 degenerate | Re-fitted mode only for the test; frozen mode only where it differs; 1-point margins with intervals |
| 6. Definitions with holes | Abstention with coverage; Dawid-Skene only with three or more members; tie rule; frozen best single; negative ΔE allowed; ΔE primary; JI interpretation corrected |
| 7. Misspecified weights | Weights stated as a baseline and clipped; confusion-matrix aggregation (Dawid-Skene) included |
| 8. Data validity | Step 0: full coverage, prompt hashes, release, type and hosting recorded; MMLU-Redux sensitivity |
| 9. Sovereignty claim did not follow | Question reframed around withdrawal of hosted APIs with local open weights; admission chart added; access-against-autarky comparison dropped |
| 10. Uncertainty | Cross-fitting, cluster bootstrap with selection inside, developer jackknife |
| 11. Multiplicity | One primary analysis named; no Holm claim; protocol hash before S1 |
| 12. S0 too thin | Global and developer distractor preferences, hierarchy, calibrated accuracies, label errors; minimal detectable effect defined |
| 13. Worst case only | Expected loss added; the worst scenario reported |
| 14. One benchmark | MMLU-Pro replication |
| 15. Missing bounds and cost | Full-panel and best-single bounds; compute-constrained variant; deliberation-based aggregation listed as a limitation |

## 11. Gate

Next step if approved: write S0 (synthetic validation) as `checks/cycle8_s0_synthetic_validation.py` and run it on the laptop CPU. Real data are downloaded only after S0 passes and the protocol hash is recorded. Estimated effort: S0 about one working session; S1 about two, including the download.

## 12. S0 result (2026-11-13): FAIL, protocol revision needed

Run: 600 simulated datasets (200 under the null, 100 at each σ in {0.1, 0.2, 0.3, 0.5}). 459 datasets were computed on the department server and 141 on the laptop, with fixed per-dataset seeds. Script `checks/cycle8_s0_synthetic_validation.py` (SHA-256 prefix 0c0fff8f936d17fd); per-dataset rows in `checks/cycle8_s0_results_merged.jsonl`; summary in `checks/cycle8_s0_synthetic_validation.out`. No real data were used.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Size of the H1 test | rejection rate 0.055 under the null (n = 200) | pass |
| 2. Minimal detectable effect | power 0.06, 0.10, 0.11 and 0.12 at σ = 0.1, 0.2, 0.3 and 0.5; never near 0.8; the mean statistic does not rise with σ | **fail** |
| 3a. ΔE interval coverage | 0.923 over 470 estimates | pass |
| 3b. Joint-importance coverage | 0.906 over 85 | pass |
| 3c. Fitted against true item difficulty | mean Spearman 0.774 (threshold 0.85) | **fail** |
| 3d. N2-predicted ΔE under the null | within tolerance in 0.975 of 871 cases | pass |
| 3e. Admission chart | mean absolute difference from truth at most 0.17 points at every grid point (threshold 0.5) | pass |

**Why criterion 2 failed (our reading, not yet tested).** Withdrawing one developer's 2 to 9 models from a 64-model plurality panel barely changes the vote. A block of correlated models also helps the vote on some items and hurts it on others, so its withdrawal effect has no consistent sign. The withdrawal loss of a small group from a large panel therefore says almost nothing about whether that group shares errors. H1 tested the right idea with the wrong statistic. This also bears on the real study: withdrawal matters for small portfolios (n = 5, the primary analysis) and for large group withdrawals (all API-only models), not for single developers in a large panel.

**Why criterion 3c failed.** The threshold was unattainable. A diagnostic on two further simulated panels gives the following Spearman correlations with the true difficulty: the simple proportion of models answering correctly, 0.82; the 2PL estimate, 0.74 to 0.76. Against the proportion correct of an independent second panel on the same items, the 2PL estimate reaches 0.90 to 0.91. With 64 models per item and a generator with guessing and wrong keys, no estimator reaches 0.85 against the latent difficulty. The quantities that N2 is used for (3d, 3e) were recovered well.

**Amendment A2, as first proposed (adopted 2026-11-13 by the authors; the adopted text is in Section 13).**

1. **Replace H1's statistic.** Test developer-level shared errors directly, as excess pairwise agreement on the same wrong answer over its N2 expectation. The statistic is the mean excess within developers minus the mean excess between developers, with an N2 parametric-bootstrap null and inference clustered by developer. Withdrawal loss of single developers becomes descriptive.
2. **Replace criterion 3c** by replicate reliability: the Spearman correlation between the fitted difficulty and the proportion correct of an independent simulated panel on the same items must be at least 0.85. This change is made after seeing the S0 result, before any real data; it is stated here as such.
3. **Test a simpler N2.** Compare a one-parameter model with guessing fixed at 1/4 against the 2PL on 3d and 3e. Keep whichever recovers better; the choice is made in S0, not on real data.
4. **Re-run S0** with the same seeds scheme and criteria 1 and 2 applied to the new H1. Expected time: about an hour on the server.

## 13. Amendment A2 (adopted 2026-11-13, before any real data)

The authors approved A2 after the S0 failure in Section 12. A2 replaces the following parts of Sections 4 to 6 and Amendment A1. Everything else stands.

**H1 (replaces Section 5, H1): developer-level shared wrong answers beyond item difficulty.**
- Definitions:
  - S_ij is the number of items on which assistants i and j give the *same wrong* answer.
  - Its N2 expectation is E_ij = Σ_k (1 − p_ik)(1 − p_jk) Σ_o w_ko², where p is the N2 probability of a correct answer and w_k is the item's pooled distribution over wrong options.
  - The excess is x_ij = (S_ij − E_ij)/N.
- Statistic: T = (mean over developers of the mean x within that developer) − (mean x over all pairs from different developers). Each developer with at least two models gets equal weight.
- Null distribution: an N2 parametric bootstrap with 100 panels.
- Decision rule on real data: supported if the one-sided p-value is below 0.05 *and* T exceeds the minimal detectable effect from S0. Refuted if the developer-jackknife 90% interval for T lies below that effect. Otherwise inconclusive.
- Withdrawal loss of single developers is reported descriptively.

**N2 (replaces Section 4, N2, for all uses).** Two variants are fitted:
- 2PL;
- 1PL-g, a one-parameter model with guessing fixed at 1/4: P(correct) = 1/4 + 3/4·σ(θ_i − b_k).

Wrong answers are drawn from w_k in both. Rule, fixed now: use the variant that passes all S0 criteria. If both pass, use the one with the higher share of N2-predicted ΔE within tolerance (criterion 3d); if that is tied, the one with the smaller maximum admission-chart error.

**S0 criteria (replace A1 criteria 2 and 3c).**
- Criterion 1 is unchanged and applies to the new H1 under each N2 variant.
- Criterion 2: the smallest σ in {0.1, 0.2, 0.3, 0.5} at which the new H1 has power of at least 0.8 exists. The minimal detectable effect is the mean T at that σ.
- Criterion 3c (replicate reliability): the mean Spearman correlation between fitted difficulty and the proportion of models answering correctly in an independent simulated panel on the same items is at least 0.85. This was changed after seeing S0, as Section 12 explains.
- Criteria 3a, 3b, 3d and 3e are unchanged. 3c, 3d and 3e are evaluated per N2 variant.

**Pre-registered secondary analysis: engineering-science subset, for the fuel-and-energy framing.**
- MMLU subset: electrical engineering, college physics, high-school physics, conceptual physics, college chemistry and high-school chemistry, 936 test items. This count is recalled, not checked, and will be confirmed in Step 0.
- All descriptive measures and H2 are reported for this subset, with selection still fitted on the full selection half.
- MMLU-Pro check [DATA]: the HELM Capabilities MMLU-Pro runs contain 1,000 items without category labels. Categories would have to come from matching question texts to the public MMLU-Pro release. Its engineering category is [REC], to be verified, and would give only about 70 items. That analysis is therefore exploratory and labelled as such.
- The subset is a proxy for technical questions in the energy sector, not operational energy decisions.

**Amendment A2.1 (2026-11-13, during implementation of A2, before the full S0 re-run, no real data).**
A timing run on one simulated *null* dataset rejected H1 (p = 0.01). A diagnostic on four more null datasets and two alternative datasets (30 bootstrap panels each) showed why. Models of the same developer have similar abilities. Similar-ability models share wrong answers that N2 does not predict, for example on items with wrong keys, where all strong models give the same "wrong" answer. The unmatched statistic was therefore biased upwards under the null, strongly so for 1PL-g (T between 0.002 and 0.005, against a bootstrap spread of 0.0005). The fix, adopted before the full run: each within-developer pair is compared only with between-developer pairs whose two accuracies match its two accuracies within 0.03 (widened up to 0.12 until at least five pairs match). In the diagnostic the matched statistic was close to 0 under the null for both variants and about 0.005–0.006 under σ = 0.3. This bears directly on real MMLU data, which contain wrong keys: an unmatched comparison would confuse shared ability with shared developer.

## 14. S0 re-run under A2 and A2.1 (2026-11-14): FAIL on one criterion, by a margin within noise

Run: 600 simulated datasets on the department server, 16 processes, about 2.8 hours (median 262 s per dataset); seeds from master seed 20261114. Script `checks/cycle8_s0b_synthetic_validation.py` (SHA-256 prefix 4a01fa8bc7b8bdb6), rows in `checks/cycle8_s0b_results.jsonl`, summary in `checks/cycle8_s0b_synthetic_validation.out`. No real data.

| Criterion | 2PL | 1PL-g | Verdict |
|---|---|---|---|
| 1. Size of the new H1 (nominal 0.05) | 0.035 | 0.050 | pass, both |
| 2. Power at σ = 0.1 / 0.2 / 0.3 / 0.5 | 0.45 / 1.00 / 1.00 / 1.00 | 0.58 / 1.00 / 1.00 / 1.00 | pass, both |
| Minimal detectable effect (mean T at σ = 0.2) | 0.0027 | 0.0028 | — |
| 3a. ΔE interval coverage | 0.912 (454) | same | pass |
| 3b. Joint-importance interval coverage | 0.890 (91) | same | **fail** (threshold 0.90) |
| 3c. Replicate reliability | 0.897 | 0.939 | pass, both |
| 3d. N2-predicted ΔE within tolerance | 0.980 | 0.981 | pass, both |
| 3e. Admission chart, worst mean error | 0.18 pp | 0.09 pp | pass, both |

**Reading.** The new H1 works. It has correct size and full power from σ = 0.2. Its minimal detectable effect is an excess same-wrong-answer agreement of about 0.28 percentage points within developers. The one failure is the joint-importance interval: 81 of 91 intervals cover the population value, against a threshold of 90%. The binomial standard error of that estimate is 0.031, so 0.890 is not distinguishable from 0.90. In the first S0 run the same check gave 0.906 over 85 cases. Seven of the ten misses lie below the interval, a slight sign of undercoverage on one side. As pre-stated, the criterion has failed. By the variant rule, 1PL-g would be selected (3d 0.981 against 0.980).

**Amendment A3 (adopted 2026-11-14 by the authors).** Validate the joint-importance intervals properly, instead of re-running everything or relabelling the failure.

1. Use a bias-corrected cluster bootstrap for JI: percentile intervals after small-sample inflation of the bootstrap deviations by √(G/(G−1)) with G = 57 subjects.
2. Validate it on 400 fresh datasets at σ = 0.3, a new master seed, and the JI computation only (no H1, no admission chart): about 20 minutes on the server. The pass threshold stays at 0.90 and is now estimated with a standard error of about 0.015.
3. If it still fails, JI becomes a descriptive point estimate without intervals in the paper. It is not part of H1 to H3.

Everything else from this run stands, and 1PL-g is the N2 variant for S1.

**A3, implementation details fixed before the run.**
- Interval for JI: draw 300 cluster-bootstrap resamples over the G = 57 subjects. Centre the bootstrap distribution on the point estimate and inflate its spread: θ̂ + √(G/(G−1))·(θ*_b − mean θ*). Take the 2.5% and 97.5% quantiles. "Bias-corrected" in the A3 text means this centring on the estimate, not the BCa method.
- Validation: 400 new datasets at σ = 0.3, master seed 20261115, with the same generator, portfolio rule (accuracy-greedy, n = 5), choice of the first two withdrawal scenarios, and population values from 60,000 fresh items as in S0. Datasets without two scenarios, or where both together empty the portfolio, are skipped and counted.
- Pass: coverage of at least 0.90. The coverage of the old percentile interval on the same datasets is reported for comparison and decides nothing.
- Script: `checks/cycle8_s0c_ji_validation.py`.

## 15. A3 result (2026-11-14): PASS. S0 is complete.

Run: 400 datasets at σ = 0.3 (367 used, 33 skipped for having fewer than two scenarios or an empty joint withdrawal), on the laptop in 3.5 minutes. Script `checks/cycle8_s0c_ji_validation.py`, rows in `checks/cycle8_s0c_results.jsonl`, summary in `checks/cycle8_s0c_ji_validation.out`.

- A3 interval coverage: **0.940** (standard error 0.016 at 0.90) → pass.
- The old percentile interval on the same datasets also covers 0.940. The earlier 0.890 over 91 cases was therefore sampling noise, as suspected. The √(G/(G−1)) inflation widens the interval by only about 1%; it is kept because it was pre-stated.
- Misses are asymmetric: 16 below the interval and 6 above. The point estimate of JI may be slightly biased upwards at this sample size; this will be stated next to every JI value.

**S0 overall, after A2, A2.1 and A3: PASS.**
- H1 test: correct size, with a minimal detectable effect of about 0.0028 excess same-wrong-answer agreement.
- Withdrawal-loss and joint-importance intervals: valid.
- Null model for S1: 1PL-g.
- The admission chart recovers the true gain within 0.1 percentage points.

**Next gate: S1 Step 0 on real data.** Protocol Section 3: download, coverage, prompt hashes, deduplication, open-weight classification. It starts only after the protocol hash is recorded in `record.md`. If Step 0 changes the panel structure by more than 20%, S0 is re-run (Amendment A1).

## 16. Step 0 rules, fixed before any answers were examined (2026-11-14)

The rules below were written while the download was running, from model names and model cards only. They are implemented in `checks/cycle8_s1_step0_checks.py`, which does not compute accuracy.

- **Release.** Each model uses the latest HELM MMLU release that contains all 57 subjects.
- **Deduplication.** Two models count as one when they share developer, family, version and size and differ only by snapshot date, preview status or alias. Models documented as sharing frozen text weights also count as one: the Llama 3.2 Vision models carry the Llama 3.1 text weights [REC]. The latest non-preview snapshot is kept. Eleven drops follow, listed in the script with reasons. Different versions of a family are kept as separate models: Llama 2, 3, 3.1 and 3.3; Mistral Large 2402 and 2407; Jamba Instruct and Jamba 1.5.
- **Access classification.**
  - *Open-weight* means the weights are published for local deployment. Checked on 2026-11-14 against the Hugging Face model API, with the licence tag recorded. Non-commercial (Command R, CC-BY-NC) and gated or custom licences are kept as open-weight but flagged.
  - One case is unverified: the DBRX repository returned 401, so its access is marked UNVERIFIED.
  - Everything else is *API-only*. That includes models whose developer publishes other open models: Jamba Instruct, Mistral Large 2402, Mistral Small 2402, Solar Pro 241126 and Yi Large preview.
- **Primary panel.** Deduplicated models that cover every item and use the modal prompt on every item.
- **S0 re-run rule (A1).** If the primary panel differs from the S0 structure (64 models, 13 developers with at least two) by more than 20% in either count, S0 is re-run with the real structure before any hypothesis is tested.

## 17. Step 0 result (2026-11-14): data are valid; the strict prompt rule triggers an S0 re-run

The public HELM MMLU outputs were downloaded and parsed on the department server in about 6 minutes. Parsed data: `data/helm_mmlu_parsed.tsv.gz` (14 MB) and `data/helm_mmlu_runs.json`. Checks: `checks/cycle8_s1_step0_checks.py`, output `checks/cycle8_s1_step0_checks.out`, panel definition `data/helm_mmlu_panel.json`. No accuracy was computed and no hypothesis was tested.

- **Coverage.** 80 of 81 models have a complete release. Gemini Pro has none and was in any case an alias of Gemini 1.0 Pro. All 80 cover all 14,038 items, and the correct keys agree across models on every item.
- **Format.** Unparseable completions are below 1% for most models. The exceptions are Mistral Large 2402 (19.7%), GPT-4 1106 preview (5.3%, dropped by deduplication), Mistral Small 2402 (4.7%), Mistral 7B v0.1 (4.3%) and Phi-3 Medium (1.6%). These count as errors, so the affected models look less accurate than their knowledge would suggest.
- **Deduplication.** 10 drops apply (the 11th, Gemini Pro, was absent), leaving **70 models** from 18 developers, 14 of them with at least two models; 42 open-weight and 28 API-only.
- **Prompt consistency.** The prompts differ only in an optional leading format instruction. A text comparison of one item across five models showed the following:
  - the modal prompt starts with "Answer with only a single letter.";
  - 39 models (mostly base models and older API models) got the same prompt without that line;
  - the three Amazon Nova models got a different instruction line instead.

  The few-shot examples, question and options are identical, and all models use temperature 0 and one output token.
- **The strict rule's consequence.** The pre-registered rule, the modal prompt on every item, leaves a primary panel of **30 models** from 11 developers, only 7 of them with at least two models; 15 are open-weight. That is 53% and 46% away from the S0 structure, so **under Amendment A1 S0 must be re-run before any test.**
- **Engineering subset confirmed:** 936 items in 6 subjects.

**Decision needed (not yet taken).**
- **(a) Keep the strict rule.** The primary panel has 30 models. S0 is re-run with that structure, which takes about 3 hours on the server. The 70-model panel becomes the sensitivity analysis.
- **(b) Amendment A4.** Treat prompts as equivalent when they differ only in the leading format instruction. The primary panel is then 70 models, within 20% of the S0 structure, so no S0 re-run is needed. Two safeguards come with it:
  - the prompt class becomes a matching variable in H1: comparison pairs from different developers must share the within pair's prompt classes, so that shared formatting cannot pass as a shared developer;
  - the strict 30-model panel becomes the pre-registered sensitivity analysis.

  The change would be decided after seeing which models fall under the strict rule, but before any answer was analysed.

## 18. Amendment A4 (adopted 2026-11-14 by the authors) and analysis details fixed before any answer is analysed

**A4.**
- The primary panel is the 70 deduplicated models. Prompts that differ only in the leading format instruction count as equivalent.
- Each model's prompt class is recorded: "modal" (the modal prompt), "none" (no instruction line) or "nova" (the Amazon Nova instruction).
- In H1, comparison pairs from different developers must match the within pair's prompt classes as well as its accuracies. Pairs without matches are dropped, which by construction drops Amazon.
- The strict 30-model panel is a pre-registered sensitivity analysis.
- This was decided after seeing which models the strict rule excludes, before any answer was analysed.

**Fixed implementation details.** These are written before the analysis script exists, so that it is held to them.

- **Items and folds.** All 14,038 MMLU test items. Two cross-fitting folds, split by subject at random with seed 20261114. Fitting and selection use one fold and evaluation uses the other; each estimate is the item-weighted combination of both directions.
- **Invalid completions.** Unparseable answers count as errors for single models and as abstentions (no vote) in voting rules.
- **Rules.**
  - The *Dawid-Skene* confusion matrices are fitted by EM on the fitting fold, without labels, and applied to the evaluation fold.
  - *Stacking* is a multinomial logistic regression on one-hot answers with L2 penalty 0.01, fitted on the fitting fold.
  - With fewer than three remaining models, Dawid-Skene is replaced by plurality.
- **Uncertainty.** A cluster bootstrap over subjects, 200 resamples, with every selection and fit repeated inside each resample through item weights. Intervals are percentile intervals, except joint importance, which uses the A3 interval.
- **H1.** Run on all items, with the 1PL-g N2 and 200 parametric-bootstrap panels. The minimal detectable effect is 0.0028, from S0. The developer jackknife leaves out one developer at a time; its 90% interval is T ± 1.645·SE.
- **H2.** For each n in {3, 5, 7, 9}, the portfolio is the n most accurate models on the fitting fold, so the choice is the same for all rules. The worst case is taken over the withdrawal of each developer present and of all API-only models. Scenarios that empty the portfolio hit every rule equally; they are excluded from the ranking and reported separately.
- **H3.**
  - Settings: n = 5, plurality, re-fitted, λ = 0.5. The training scenario family is the withdrawal of each of the 18 developers plus all API-only models.
  - Robust selection: greedy construction, then single swaps until no swap improves the objective.
  - Leave-one-developer-out: for every developer X that appears in any portfolio, robust selection is refitted without X's scenario and evaluated with X withdrawn. The held-out worst case is the maximum over those X.
  - Comparators: the diversity cap (at most one model per developer, the most accurate first) and accuracy-greedy, both evaluated on the same X. Greedy mutual information is computed as a plug-in estimate of I(Y; answers). This only approximates the Turkmen et al. estimator, and is reported as such.
  - The full-availability error of robust selection is that of the selection trained with all scenarios.
- **Descriptive outputs.**
  - Primary portfolio (n = 5, accuracy-greedy, plurality): ΔE, RAW and joint importance for every scenario, with the observed ΔE against N1 and N2 simulations (20 each).
  - Sovereignty comparison: under API-only withdrawal, the primary portfolio is set against the best open-weight portfolio (n = 5), the full 70-model plurality vote, and the best single open-weight model. A compute-constrained open-weight portfolio is also reported: at most 100B total parameters, with parameter counts taken from model names and cards and listed in the script.
  - Admission chart: for the open-weight portfolio, model-based, labelled as such.
- **Engineering subset.** The descriptives and H2 are re-run with evaluation restricted to the 936 engineering-science items; selection stays on the full fitting fold.
- **MMLU-Pro replication.** Not part of this run. It needs a separate download and parsing of chain-of-thought answers, and it will be reported later.

**Implementation notes added after testing the analysis script on simulated data, before running it on real data.**
- The robust-selection objective first skipped scenarios that empty the portfolio. That let it escape the worst case by taking all five models from one developer. Such a scenario now counts as error 0.75 (random guessing), as in the held-out evaluation.
- The H1 sensitivity analyses use half as many bootstrap panels (100) as the primary test, to save time: strict 30-model panel, without the largest developer, without models below 40% accuracy, open-weight models only.
- Prompt classes in H1 are exact prompt texts, which is stricter than A4's three-way description. The models without the instruction line form four variants that are never identical to each other: 30 models; Yi Large preview with Command R and R+; Claude 2.1 with Claude Instant; Nemotron. Within-developer pairs whose classes have fewer than five matching between-developer pairs are dropped. This removes the Cohere pair and every Anthropic pair that involves an older Claude model. It was found when the real data were loaded, before any statistic was computed.

## 19. S1 primary analysis result (2026-11-14)

**Run.**
- The run used the department server: 16 workers, 200 cluster-bootstrap resamples, 200 N2 panels for H1 (100 for each H1 sensitivity analysis). It took 4,307 s.
- Script `checks/cycle8_s1_analysis.py`, SHA-256 `29ea5de5…`, identical on the server and here and as recorded in `record.md` before the run.
- Outputs: `checks/cycle8_s1_analysis.out`, `.json`, `checks/cycle8_s1_bootstrap.jsonl`, `checks/s1_run.log`.
- Panel: 70 models, 14,038 items, 18 developers, 28 API-only.

**Protocol hash.** The last bullet of Section 18 (exact prompt classes) was appended after the run had started and before any result was seen. It documents what the script already did. Without that bullet the file hashes to the recorded `8ff337fb…`; this was checked.

All errors below are in percent; intervals are 95% cluster-bootstrap intervals unless stated otherwise.

### H1: SUPPORTED

| Panel | Models | Developers used | T | p (one-sided) | 90% jackknife CI | Verdict |
|---|---|---|---|---|---|---|
| Primary (A4) | 70 | 12 (170 within pairs) | 0.00462 | 0.005 | [0.00162, 0.00762] | supported |
| Strict prompt panel | 30 | 7 | 0.00991 | 0.010 | [0.00502, 0.01479] | supported |
| Without largest developer (Google) | 61 | 11 | 0.00422 | 0.010 | [0.00096, 0.00748] | supported |
| Without accuracy below 40% | 69 | 11 | 0.00458 | 0.010 | [0.00149, 0.00766] | supported |
| Open-weight only | 42 | 9 | 0.00595 | 0.010 | [0.00333, 0.00857] | supported |

- Every p-value is the smallest possible for its number of null panels (1/201 and 1/101): no simulated panel reached the observed T.
- The N2 null has mean 0.00005 and standard deviation 0.00034, so the primary T lies about 13 null standard deviations out. T is 1.65 times the minimal detectable effect of 0.0028.
- Meaning: two models from the same developer give the same wrong answer on about 0.46% more items (about 65 of 14,038) than accuracy- and prompt-matched pairs from different developers. That is beyond what item difficulty and the pooled wrong-answer distribution explain.
- As stated in Section 5, "developer" bundles shared data, training, post-training and hosting wrappers. These mechanisms are not separated.

### H2: REFUTED

Portfolio = the n most accurate models, re-fitted rules; the worst case is over the withdrawal of each developer present and of all API-only models. No scenario emptied a portfolio.

| n | Full-availability winner (error) | Worst-case winner (error) | Plurality worst case | Best single, full / worst |
|---|---|---|---|---|
| 3 | plurality, 11.23 | stacking, 13.41 | 13.66 | 12.36 / 13.66 |
| 5 | plurality, 11.16 | plurality, 13.24 | 13.24 | 12.36 / 13.66 |
| 7 | stacking, 11.12 | plurality, 13.24 | 13.24 | 12.36 / 13.66 |
| 9 | stacking, 11.16 | plurality, 12.93 | 12.93 | 12.36 / 13.66 |

- At every n the full-availability winner is the worst-case winner, or within 0.5 points of it: the refutation rule of Section 5.
- The four voting rules (plurality, weighted vote, Dawid-Skene, stacking) differ by at most 0.5 points, both under full availability and in the worst case.
- Engineering subset (936 items, descriptive), plurality full / worst case:
  - n = 3: 14.42 / 19.34;
  - n = 5: 15.17 / 17.41;
  - n = 7: 16.56 / 17.95;
  - n = 9: 16.56 / 17.63.
- On that subset stacking has the lowest worst case at every n (18.27, 16.67, 17.20, 17.63; tied with plurality at n = 9). Its advantage over plurality is at most 1.07 points, on a subset whose intervals are several points wide. This is not evidence for H2.

### H3: REFUTED

n = 5, plurality, λ = 0.5; held out over every developer that enters any portfolio.

| Selection | Full availability | Held-out worst case | Held-out mean |
|---|---|---|---|
| Diversity cap (≤1 per developer) | 10.91 | 11.65 | 11.16 |
| Accuracy-greedy | 11.06 | 11.65 | 11.26 |
| Scenario-robust | 11.12 | 11.81 | 11.33 |
| Greedy mutual information (plug-in) | 11.75 | 12.65 | 11.88 |

- Gain of robust selection over the cap: −0.16 points (95% CI [−0.63, 0.20]). Its full-availability cost against the cap is 0.21 points.
- The cap is as good or better, within 0.5 points: refuted.
- The cap also has the lowest full-availability error of all four.
- The held-out evaluation covers single developers only. The API-only scenario is always in the training family, so it is never held out.

### Descriptive outputs

**Primary portfolio.**
- Accuracy-greedy, n = 5, plurality: Claude 3.5 Sonnet (Oct 2024), Yi 34B, Gemini 1.5 Pro 002, DeepSeek-V3, GPT-4o (Aug 2024). It is the same in both folds.
- Error with all models available: 11.06 [8.89, 14.65].

| Withdrawn | ΔE | RAW (point, derived as E(P\W)/E(P)) |
|---|---|---|
| All API-only | 2.19 [1.24, 5.28] | 1.20 |
| Anthropic | 0.59 [0.26, 1.27] | 1.05 |
| DeepSeek | 0.46 [0.02, 0.83] | 1.04 |
| Google | 0.38 [0.06, 0.93] | 1.03 |
| OpenAI | 0.27 [−0.10, 0.69] | 1.02 |
| 01.AI | 0.10 [−0.51, 0.31] | 1.01 |

- **Joint importance, between developers.** Withdrawing any two of Anthropic, DeepSeek, Google and OpenAI together costs more than the two separate losses added together. The excess runs from 0.60 [0.21, 1.00] (DeepSeek + OpenAI) to 1.26 [0.57, 1.91] (Anthropic + DeepSeek); all six intervals exclude 0. A five-model vote degrades non-linearly as it shrinks to three models.
- **Joint importance, artefacts that are not findings.**
  - Pairs of an API-only developer with the API-only group are nested, so JI equals −ΔE(developer) by construction. These are the values −0.59, −0.38 and −0.27.
  - DeepSeek + API-only (10.14 [2.18, 13.64]) leaves Yi 34B alone. It measures that single weak model, not an interaction.

**Sovereignty comparison (withdrawal of all API-only models).**

| Configuration | All items | Engineering subset |
|---|---|---|
| Primary portfolio, all available | 11.06 | 15.81 |
| Primary portfolio, API-only withdrawn (DeepSeek-V3 + Yi 34B; the tie-break makes it DeepSeek-V3 alone) | 13.24 | 16.24 |
| Best single open-weight model (DeepSeek-V3) | 13.24 | 16.24 |
| Best open-weight portfolio, n = 5 (greedy on open models) | 13.77 | 20.41 |
| Scenario-robust portfolio, API-only withdrawn | 13.41 | – |
| Best single model of any kind (Claude 3.5 Sonnet) | 12.36 | – |
| Full 70-model plurality | 17.04 | not computed |
| Full open-weight plurality (42 models) | 19.08 | 26.92 |
| Compute-constrained open portfolio, ≤100B total (Qwen2.5 72B, Yi 6B, Phi-3 medium, Phi-3 small) | 19.03 | – |

- After API-only withdrawal the best achievable configuration is the best open-weight model on its own. Building an ensemble from the remaining open models did not help.
- Unweighted plurality over large or unequal panels is worse than the best single member, because many weak voters outvote strong ones (17.04 against 12.36). The same holds for the compute-constrained portfolio: Qwen2.5 72B alone has a plain error of 16.70, computed without cross-fitting from the parsed answers; there is no selection step, so none is needed. Greedy selection fills all n slots even when an added model makes the vote worse. This is a property of the pre-registered selection rule, and it is reported as such.
- Engineering subset: the primary portfolio loses 0.43 points under API-only withdrawal (15.81 → 16.24). Intervals on 936 items are several points wide.

### Pre-registered outputs not produced in this run

- RAW with intervals: only point values derived from ΔE are given above.
- The observed ΔE against N1 and N2 simulations (20 each, Section 18): not implemented in the script.
- The admission chart on real data (Sections 5 and 18): not implemented.
- Full 70-model plurality on the engineering subset under full availability.
- Sensitivity analyses for H2 and H3 (Section 8 lists them without restricting them to H1; Section 18 specified only the H1 ones).
- MMLU-Redux exclusion sensitivity.
- MMLU-Pro replication of H1 to H3 (deferred in Section 18).

### Reading against Section 9

The outcome is the mixed case the protocol did not name.
- Developer-level shared errors are real and robust (H1).
- Neither the choice of aggregation rule (H2) nor scenario-robust selection (H3) adds anything over a one-model-per-developer cap.
- Losing all API-only models costs 2.19 points [1.24, 5.28] of error, about 20% relative, or 2.5% of accuracy. The pipeline falls back to the level of the best open-weight model.

The defensible message is therefore a measurement result: shared errors cluster by developer, so a diversity cap is the right cheap policy. The size of the sovereignty cost is bounded by the best locally held open-weight model, not recovered by ensembling. Elaborate robust selection is not needed.

## 20. Supplementary pass S1b: the missing pre-registered outputs, details fixed before running (2026-11-14)

The authors chose to produce the missing outputs before the MMLU-Pro replication. The script is `checks/cycle8_s1b_supplementary.py`. It imports the frozen S1 script and does not modify it. The primary results of Section 19 are not re-estimated and do not change. Details below are fixed before the script is run on real data.

- **RAW intervals.** RAW_W = (E(P) + ΔE_W)/E(P), computed in each of the 200 existing S1 bootstrap resamples (`checks/cycle8_s1_bootstrap.jsonl`), with percentile intervals. Resamples in which the scenario is absent from the reselected portfolio are dropped, and their number is reported. No new run is needed.
- **ΔE against N1 and N2.**
  - 20 simulated 70-model panels under each null model, fitted on all items, as in H1 (a parametric bootstrap). The real developer and API labels are kept.
  - N1: model i is correct with its overall accuracy, and wrong answers are uniform over the three distractors.
  - N2: the 1PL-g fit used in H1, with wrong answers drawn from the item's pooled wrong-answer distribution.
  - On each simulated panel the full primary pipeline is re-run: accuracy-greedy n = 5, cross-fitted, plurality. ΔE is recorded for each scenario present.
  - Reported, against the observed values: the mean and the 5th–95th percentile range of simulated ΔE for the API-only scenario and for the largest single-developer ΔE in the portfolio. A scenario that empties the portfolio is counted and left out.
- **Admission chart.**
  - Base: the open-weight portfolio of Section 18 (greedy n = 5 on open-weight models), cross-fitted.
  - For each direction, 1PL-g is fitted to the 42 open-weight models' answers on the evaluation fold. The candidate's ability is solved so that its mean correct probability on those items equals a.
  - With probability r the candidate copies the portfolio's plurality answer; otherwise it answers independently from the fit, with wrong answers from the pooled distribution. This is the S0-validated construction.
  - Grid as validated in S0: a ∈ {0.5, 0.7, 0.85}, r ∈ {0, 0.3, 0.6}; 20 draws per cell. The gain is the reduction in plurality error, the mean over draws and both directions. It is model-based and labelled so.
- **Full 70-model plurality on the engineering subset,** under full availability, with intervals from the same 200 bootstrap weights as S1 (same seeds).
- **Sensitivity analyses for H2 and H3** (Section 8): the strict 30-model panel, without the largest developer (Google, 61 models), and without models below 40% accuracy (69 models).
  - The S1 analysis is re-run on each sub-panel with the same folds and decision rules, and 100 bootstrap resamples (half the primary number, as for the H1 sensitivity analyses).
  - The decision code is copied from the S1 script. It is checked by reproducing the primary H2 and H3 verdicts and numbers from the stored S1 point estimates and resamples before it is applied to the sub-panels.
- **Not in S1b:** MMLU-Redux (needs the re-annotation data) and MMLU-Pro (needs a separate download).

## 21. S1b result (2026-11-14)

**Run.**
- Department server, 16 workers, 4,542 s. Script SHA-256 `64ac427c…`, identical on the server and as recorded before the run.
- Outputs: `checks/cycle8_s1b_supplementary.out`, `.json`, `checks/cycle8_s1b_sens_bootstrap.jsonl`, `checks/s1b_run.log`.
- The copied decision code reproduced the S1 verdicts and the H3 gain and interval exactly before it was used.

**RAW, primary portfolio** (95% percentile intervals from the 200 S1 resamples):

| Withdrawn | RAW | 95% CI | Resamples used |
|---|---|---|---|
| All API-only | 1.198 | [1.123, 1.556] | 200 |
| Anthropic | 1.053 | [1.024, 1.135] | 200 |
| DeepSeek | 1.042 | [1.002, 1.072] | 198 |
| Google | 1.035 | [1.006, 1.094] | 200 |
| OpenAI | 1.024 | [0.989, 1.052] | 180 |
| 01.AI | 1.009 | [0.953, 1.024] | 200 |

**ΔE against the null models** (20 simulated 70-model panels each; the primary pipeline is re-run on each; values in percentage points, mean [5th, 95th percentile]):

| | Portfolio error | ΔE, API-only withdrawn | Largest single-developer ΔE |
|---|---|---|---|
| N1 (independence) | 1.20 [1.06, 1.33] | 11.08 [6.83, 17.39] | 2.90 [1.51, 4.65] |
| N2 (item difficulty, pooled wrong answers) | 9.98 [9.62, 10.45] | 2.87 [2.36, 4.05] | 1.35 [0.52, 2.46] |
| Observed | 11.06 | 2.19 | 0.59 |

- N1 is far from the data: under independence a five-model vote would err on about 1% of items, and withdrawal would cost 11 points. Independence is not a usable baseline for withdrawal risk.
- N2 comes close, but it overstates both the ensemble's accuracy (9.98 against 11.06) and the cost of withdrawal (2.87 against 2.19). The observed API-only ΔE lies below the 5th percentile of the 20 N2 panels.
- Reading: real errors are more dependent than item difficulty alone implies, consistent with H1. So the ensemble gains less from its members, and losing members costs less, than an N2-based assessment predicts. This is descriptive (20 panels, no test).

**Admission chart, model-based.**
- Base: the open-weight portfolio of Section 19, with plurality error 13.77.
- A candidate is added with accuracy a. With probability r it copies the portfolio's plurality answer; otherwise it follows the 1PL-g fit to the 42 open-weight models.
- Gain in percentage points, mean over 20 draws (standard deviation over draws 0.03–0.09):

| a \ r | 0 | 0.3 | 0.6 |
|---|---|---|---|
| 0.50 | −0.29 | −0.19 | −0.11 |
| 0.70 | −0.11 | −0.09 | −0.04 |
| 0.85 | +0.96 | +0.65 | +0.40 |

- A sixth voter helps only when it is about as accurate as the strongest members. At a = 0.85 the gain is about 1 point if its errors are independent of the panel given the item, and it falls by more than half when it copies the panel 60% of the time. Weaker candidates make the vote worse.
- The construction allows no dependence beyond copying and item difficulty. The S1 data (H1, and the N2 gap above) show that real errors are more dependent than that, so the r = 0 column is likely optimistic.

**Full 70-model plurality, engineering subset, full availability:** 24.89 [18.01, 35.35]. With API-only models withdrawn it was 26.92 (Section 19), a loss of about 2 points.

**Sensitivity analyses for H2 and H3** (100 resamples each):

| Panel | Models | H2 | H3: robust / cap held-out worst case; gain [95% CI]; full-availability cost |
|---|---|---|---|
| Strict prompt panel | 30 | refuted | 11.65 / 11.65; 0.00 [−0.56, 0.27]; 0.00 |
| Without Google | 61 | refuted | 12.56 / 12.39; −0.16 [−1.04, 0.23]; 0.31 |
| Without accuracy below 40% | 69 | refuted | 11.81 / 11.65; −0.16 [−0.50, 0.21]; 0.21 |

Both refutations hold in every sensitivity analysis.

**Still missing:** the MMLU-Redux exclusion sensitivity and the MMLU-Pro replication.

### Overall reading after S1 and S1b

- Shared errors cluster by developer (H1, robust across four sensitivity panels).
- A one-model-per-developer cap is as good as anything more elaborate (H2 and H3 refuted, robust across three sensitivity panels).
- Losing hosted API access costs about 2.2 points of error (RAW 1.20). The system falls back to the level of the best locally held open-weight model.
- Assessments that assume independence overstate this cost about fivefold. Assessments that control only for item difficulty overstate it by about a third, because they overstate how much the ensemble contributes in the first place.
- A new model, for example a domestic one, would add about 1 point only if it is as accurate as the best open models and errs independently of them. That is model-based and likely optimistic.

## 22. MMLU-Pro replication: rules fixed before any answer is parsed (2026-11-14)

**What the data look like.** Checked from the bucket listing and one run, `allenai_olmo-2-0325-32b-instruct`.
- HELM Capabilities has one MMLU-Pro run for each of 70 models, under releases v1.0.0 to v1.15.0. The scenario is `mmlu_pro:subset=all,use_chain_of_thought=true,use_few_shot=false`; 5 runs add `num_output_tokens=2048`.
- Each run has 1,000 test instances with up to ten options, zero-shot.
- Answers are free-text reasoning. The prompt asks for the final line "The correct answer is (X)", and HELM scores them with its GPQA chain-of-thought metric.
- Instances carry an id but no subject category.
- The one run inspected uses temperature 1 and one completion per item.

These features change the design in five ways, fixed here:

1. **Items and models.**
   - The primary item set is the instance ids common to all retained models. If fewer than 950 ids are common to all models, models are dropped, those with the fewest items first, until 950 are common.
   - Adapter settings (maximum tokens, temperature, number of outputs, prompt text) are recorded per model. The "prompt class" is the exact combination of these settings, as in A4.
2. **Answer parsing.**
   - Only the first completion is used. The answer letter is taken from the *last* match of `answer is\s*\(?\s*([A-J])\b` in the completion. If there is none, it comes from the last match of `\(([A-J])\)`. Otherwise the answer is invalid.
   - The letter is mapped to an option through HELM's `output_mapping`.
   - Validation: per model, the correctness from this parsing is compared with HELM's own per-instance chain-of-thought correctness. Models with agreement below 0.95 are excluded from the primary panel and kept in a sensitivity panel.
   - The agreement table is reported. Computing it shows accuracies, which cannot be avoided; no hypothesis statistic is computed before the S0 decision below.
3. **Deduplication and access.**
   - The Section 16 rules apply unchanged: earlier snapshots are dropped (Claude 3.5 Sonnet 20240620, GPT-4o 2024-08-06).
   - One case is added: runs of the same weights with a different reasoning budget count as one model, and the run without the budget is kept (Claude Opus 4 and Claude Sonnet 4 "thinking-10k").
   - Different versions are kept separately: for example Qwen3 235B and its 2507 instruct release, GPT-5 and GPT-5.1.
   - Open-weight status is checked against the Hugging Face model API and its licence tag recorded, as before. Where an open release exists but HELM may have queried a different hosted version (Writer Palmyra Fin and Med), the model is flagged.
4. **Uncertainty without subjects.**
   - Items are the bootstrap clusters (an item bootstrap, 200 resamples). The A3 JI interval is used unchanged; its inflation factor is about 1 for 1,000 clusters.
   - The cross-fitting folds are a random half split with seed 20261114.
   - The engineering subset and the compute-constrained variant are not run: there are no categories, and parameter counts are unknown for most API models.
5. **Ten options and 1,000 items change the H1 calibration.**
   - The H1 statistic, the 1PL-g null (guessing floor 1/10 instead of 1/4) and the pooled wrong-answer distribution are generalised to ten options. Items with fewer options use their own number of options.
   - The minimal detectable effect from S0 does not transfer to 1,000 items. Before any hypothesis statistic is computed on MMLU-Pro, a reduced S0 ("S0-Pro") is run with the real MMLU-Pro panel structure: developers, sizes, API status, 1,000 items, ten options.
   - S0-Pro checks the H1 test's size (at most 0.06 over 200 null datasets) and finds its minimal detectable effect (power 0.8 over 100 datasets per σ in {0.1, 0.2, 0.3, 0.5}). It also checks the coverage of the item-bootstrap ΔE and JI intervals (at least 0.90 over 100 datasets at σ = 0.3).
   - If S0-Pro fails, the replication of that part is reported as not possible, not run anyway.

**Replication criteria.**
- H1 replicates if it is SUPPORTED under the Section 13 rule with the S0-Pro minimal detectable effect. It fails to replicate if REFUTED, and is otherwise inconclusive.
- H2 and H3 replicate their S1 outcome if they are again REFUTED under the Section 5 rules.
- The descriptive outputs are reported as in Section 19: ΔE, JI and RAW for the primary portfolio, and the sovereignty comparison.
- The H1 sensitivity analyses are the strict prompt-class panel, without the largest developer, without models below 40% accuracy, and open-weight only.

**Known limits.**
- Sampling at temperature 1 adds noise that is unrelated to other models. This dilutes shared errors, so it biases H1 towards the null.
- Reasoning models and long chain-of-thought produce more unparseable answers. They count as errors for single models and as abstentions in votes, as in S1.

## 23. S0-Pro result (2026-11-14): FAIL on two criteria; amendment A5 proposed

**Run.** On the department server, 600 datasets in 294 s. Script `checks/cycle8_s0pro_validation.py`, hash in `record.md`. Results: `checks/cycle8_s0pro_validation.out` and `checks/cycle8_s0pro_results.jsonl`.

| Criterion | Result | Limit | Verdict |
|---|---|---|---|
| 1. Size of the H1 test (200 null datasets) | 0.070 (14 rejections) | ≤ 0.06 | FAIL |
| 2. Minimal detectable effect | power 0.04 / 0.51 / 0.88 / 1.00 at σ = 0.1 / 0.2 / 0.3 / 0.5; MDE = 0.00358 (σ = 0.3) | power ≥ 0.8 at some σ | PASS |
| 3. Item-bootstrap coverage at σ = 0.3 | ΔE 0.871 (458 intervals), JI 0.927 (96) | ≥ 0.90 each | FAIL (ΔE) |

Following Section 22, no MMLU-Pro hypothesis statistic has been computed.

**Diagnosis**, from the stored results only, with nothing re-run:
- **Size.** 14 rejections in 200 is what a test of exact size 0.05 produces with probability 0.13: the binomial standard error at 0.05 is 0.015. The mean null T is 0.00002. The excess over 0.06 is therefore within noise, as the JI coverage of 0.890 was in Section 14. It is not yet shown to be noise.
- **ΔE coverage.**
  - The intervals are unbiased: the mean error is 0.0003 for developer scenarios and 0.0007 for API-only, and misses fall evenly on both sides (22 below and 22 above for developers).
  - They are about a quarter too narrow: mean width 0.031, implying a standard error of 0.0079, against an actual estimation error with standard deviation 0.0104.
  - The likely cause is a mismatch in the simulation, not in the method. S0-Pro draws its "true" values from fresh items in *new* subjects, with new subject-level difficulty shifts. The item bootstrap cannot reflect that variance. In MMLU-Pro the 14 categories are fixed, and the inference target is new items from the same categories.
  - The S1 design was different: there, the subject-cluster bootstrap matched a target of new subjects.

**Amendment A5 (adopted 2026-11-14 by the authors; implemented in `checks/cycle8_s0pro_a5.py`, master seed 20261117).** Two targeted re-checks with fresh seeds, criteria unchanged:
1. **Size.** 400 new null datasets, with 200 null panels per test as in the real analysis. Pass if the rejection rate is at most 0.06.
2. **Coverage.** 100 new datasets at σ = 0.3. The true values are computed on 50,000 fresh items drawn from the *same* 14 subject shifts as the observed items. Pass if ΔE and JI coverage are each at least 0.90.

If A5 passes, the replication runs as fixed in Section 22, with the S0-Pro MDE of 0.00358. If coverage still fails, the MMLU-Pro intervals are reported as approximate and only the H1 test and the point estimates are used. If size still fails, H1 is not replicated on MMLU-Pro.

## 24. A5 result (2026-11-14): size PASS, coverage FAIL by 0.004; fallback of Section 23 applies

**Run.** Department server, 500 datasets in 356 s. Results: `checks/cycle8_s0pro_a5.out` and `checks/cycle8_s0pro_a5_results.jsonl`.

| Re-check | Result | Limit | Verdict |
|---|---|---|---|
| 1. Size (400 null datasets, 200 null panels per test) | 0.050 (binomial SE 0.011) | ≤ 0.06 | PASS |
| 2. ΔE coverage, fixed subjects | 0.896 over 431 intervals (SE 0.015); misses 19 below and 26 above; API-only 0.899, developers 0.895 | ≥ 0.90 | FAIL |
| 2. JI coverage, fixed subjects | 0.910 over 89 | ≥ 0.90 | PASS |

- The earlier size of 0.070 was noise. The H1 test on MMLU-Pro is calibrated, and its minimal detectable effect is 0.00358 (Section 23).
- With fixed subjects, ΔE coverage rose from 0.871 to 0.896. That is still below the limit, by much less than one standard error.
  - The intervals remain somewhat too narrow: mean width 0.0342 implies a standard error of about 0.0087, against an error standard deviation of 0.0104.
  - Their nominal level is 95%; the criterion of 0.90 was already lenient.

**Consequence, fixed in Section 23 before A5 was run.**
- The replication runs as specified in Section 22, with MDE 0.00358.
- All MMLU-Pro bootstrap intervals (ΔE, JI, RAW, H2 and H3 gaps) are reported as approximate, probably too narrow by about a sixth.
- Decisions use only the H1 test (its size is validated) and point estimates:
  - H2 and H3 can be refuted, because their refutation rules use point estimates only.
  - Neither can be supported on interval evidence: an outcome that would need its interval excluding 0 is reported as inconclusive.

## 25. MMLU-Pro replication result (2026-11-14)

**Run.**
- Department server, 366 s. Script `checks/cycle8_pro_analysis.py`, SHA-256 `bcf07fa5…`, identical on the server and as recorded before the run.
- Outputs: `checks/cycle8_pro_analysis.out`, `.json`, `checks/cycle8_pro_bootstrap.jsonl`, `checks/pro_run.log`.
- Panel: 58 models, 1,000 items, 15 developers, 31 API-only; MDE 0.00358.
- Following Section 24, all intervals are approximate (probably too narrow), and decisions rest on the H1 test and point estimates.

### H1: SUPPORTED (replicates), with weaker robustness than on MMLU

| Panel | Models | T | p | 90% jackknife CI | Verdict |
|---|---|---|---|---|---|
| Primary | 58 (11 developers used, 132 within pairs) | 0.00414 | 0.005 | [−0.00253, 0.01081] | supported |
| Strict setting class | 22 | 0.01313 | 0.010 | [0.00297, 0.02330] | supported |
| Without largest developer (OpenAI) | 46 | 0.00534 | 0.010 | [−0.00364, 0.01431] | supported |
| Without accuracy below 40% | 50 | 0.00304 | 0.030 | [−0.00046, 0.00655] | inconclusive (T below MDE) |
| Open-weight only | 27 | 0.00576 | 0.020 | [−0.00794, 0.01946] | supported |
| With the 8 low-agreement models | 66 | 0.00600 | 0.010 | [0.00102, 0.01098] | supported |

- The test rejects the N2 null clearly: the null mean is 0.0001 and the null standard deviation 0.0011, so T lies about 3.6 null standard deviations out.
- T exceeds the MDE, so the decision rule gives SUPPORTED. On MMLU, T was 0.00462 against an MDE of 0.0028.
- The developer jackknife interval includes 0, unlike on MMLU: the excess is uneven across developers.
- **Post-hoc diagnostic, not pre-registered.** T recomputed with each developer's models removed from the panel ranges from 0.00333 (without Google) to 0.00696 (without Mistral), except without Writer: 0.00151, below the MDE. On MMLU-Pro the effect leans substantially on the Writer pair (Palmyra Med and Palmyra X5).
- Reading: developer-level shared errors replicate on a second benchmark with newer models, 10 options and sampled reasoning at temperature 1. The evidence is weaker and less evenly spread than on MMLU.

### H2: INCONCLUSIVE (the S1 refutation does not replicate)

| n | Full availability: best single / plurality / best rule | Worst case: best single / plurality / best rule |
|---|---|---|
| 3 | **9.80** / 10.50 / – | 14.00 / 13.90 / stacking **13.10** |
| 5 | **9.80** / 10.10 / – | 14.00 / 12.40 / Dawid-Skene **12.30** |
| 7 | 9.80 / **9.60** (plurality) | 14.00 / **11.40** (plurality) |
| 9 | 9.80 / **9.70** (plurality) | 14.00 / **10.90** (plurality) |

- At n = 3 and 5 the most accurate rule is the best single model, Gemini 3 Pro preview, with plain error 9.8%. That is 3.2 points better than the next model. It is not the most robust: when Google is withdrawn the single-model rule falls to 14.00, while the voting rules keep 12.3–13.1.
- This is the direction H2 predicted. It misses the pre-registered magnitude: the full-availability gaps are 0.5 and 0.9 points against a required 1 point, so no size counts as a flip.
- It also exceeds the 0.5-point closeness margin, so H2 is not refuted either.
- On MMLU no single model dominated, and H2 was refuted.

### H3: REFUTED (replicates)

| Selection | Full availability | Held-out worst case |
|---|---|---|
| Diversity cap | 9.80 | 11.10 |
| Scenario-robust | 10.70 | 13.10 |
| Accuracy-greedy | 10.90 | 14.80 |
| Greedy mutual information | 13.60 | 57.20 |

- Robust selection is 2.00 points worse than the cap in the held-out worst case (approximate CI [−4.61, −0.30]) and 0.90 points worse under full availability.
- The plug-in mutual-information selection picks weak models whose answers look informative. Once Google is withdrawn, the vote rests on Llama 3.1 8B, OLMo 2 and Palmyra Med.

### Descriptive outputs (approximate intervals)

- **Primary portfolio.** It is accuracy-greedy, so it differs by fold:
  - fold 0: Gemini 3 Pro, Gemini 2.0 Flash Lite, o3, Nova Pro, Claude Sonnet 4;
  - fold 1: Gemini 3 Pro, Kimi K2, Grok 4, Claude 3.5 Sonnet, OLMo 2 7B.

  Its error is 10.90 [8.89, 13.50]. Withdrawing Google costs ΔE 3.90 (RAW 1.36). The other developers cost between −1.00 and +1.00: removing a member sometimes helps the vote.
- **API-only withdrawal** empties the fold-0 portfolio, which is all API, so no ΔE is defined (the script's 75.00 is the guessing penalty).
- **JI.** No interval clearly excludes 0.

**Sovereignty comparison** (error, %):

| Configuration | All available | API-only withdrawn |
|---|---|---|
| Best single model (Gemini 3 Pro preview) | 9.80 | – |
| Diversity-cap portfolio | 9.80 | – |
| Best single open-weight model (Qwen3 235B 2507) | – | 16.30 [13.80, 20.50] |
| Best open-weight portfolio, n = 5 | – | 15.00 [12.40, 17.90] |
| Scenario-robust portfolio | 10.70 | 15.80 |
| Full panel plurality (58 / 27 models) | 14.30 | 17.70 |

- Losing hosted API access costs about 5.2 points on MMLU-Pro: from 9.80 with the cap to 15.00 with the best open portfolio. That is about 1.5 times the error. On MMLU the cost was 2.2 points.
- The gap between the frontier API models and the best open models is wider on this newer, harder benchmark.
- Unlike on MMLU, an ensemble of open models (15.00) beats the best single open model (16.30).

### Replication summary

| | MMLU (S1) | MMLU-Pro | Replicated? |
|---|---|---|---|
| H1 developer-level shared errors | supported, robust | supported; jackknife interval includes 0, leans on one developer pair | yes, weaker |
| H2 accuracy-robustness trade-off of rules | refuted | inconclusive: the single frontier model is most accurate but fragile | no (the direction favours H2) |
| H3 robust selection beats the cap | refuted | refuted (cap better by 2 points) | yes |
| Cost of losing API access | 2.2 points (RAW 1.20) | about 5.2 points (≈ ×1.5) | larger |

**Not done:** the MMLU-Redux sensitivity, and the S1b-type supplementary outputs (N1/N2 ΔE, admission chart) on MMLU-Pro.

## 26. Russian open-weight models: run design (proposed 2026-11-14, not yet started)

The authors approved running Russian open models on the department server GPU. The state of the server, checked read-only:
- RTX 4070 with 12 GB VRAM, and 733 GB free in `/home`.
- [Line about the internal administration of the department server redacted for publication.]
- [Line about the internal administration of the department server redacted for publication.]
- [Line about the internal administration of the department server redacted for publication.]

**Benchmark.** HELM MMLU, all 14,038 items, so the new answers can be set beside the 70-model panel. The rules are fixed before any local answer exists:
- **Prompt.** The exact HELM prompt of the modal prompt class for each item (5-shot), taken from one modal-class model's `scenario_state.json`.
- **Generation.** It is sent as the user message through the model's chat template (Ollama `/api/generate` with the template applied). Greedy decoding (temperature 0), at most 8 output tokens, reasoning switched off where the model has a switch.
- **Answer.** The first option letter of the completion, as in Step 0 for HELM.
- **Load.** One model at a time, at most 4 parallel requests (the service setting), with the default context.

**Models.** All candidates have official or widely used GGUF builds; quantisation is Q4_K_M unless noted. Licences were checked on the Hugging Face API on 2026-11-14.

| Model | Developer | Declared base (HF `base_model`) | GGUF, size |
|---|---|---|---|
| YandexGPT-5-Lite-8B-instruct | Yandex | YandexGPT-5-Lite-8B-pretrain (own) | official, 4.9 GB |
| GigaChat3.1-10B-A1.8B | Sber (ai-sage) | GigaChat3-10B-A1.8B-base (own) | official Q6_K, 8.8 GB |
| GigaChat-20B-A3B-instruct-v1.5 | Sber (ai-sage) | own | official Q5_0, 14.2 GB (partly on CPU) |
| T-lite-it-2.1 | T-Bank (t-tech) | **Qwen3-8B** | official, 5.0 GB |
| Vikhr-Nemo-12B-Instruct-R-21-09-24 | Vikhr | **Mistral-Nemo-Instruct-2407** | bartowski, 7.5 GB |
| saiga_nemo_12b | I. Gusev | not declared (Mistral Nemo by name) | official, 7.5 GB |
| saiga_yandexgpt_8b | I. Gusev | not declared (YandexGPT-5-Lite by name) | official, 4.9 GB |
| Cotype-Nano | MTS AI | not declared | official 8-bit, 1.6 GB |

- T-pro-it-2.1 (Qwen3-32B base, 19.8 GB) is excluded: it does not fit in 12 GB, and running it on the CPU would take most of a day.
- **Foreign anchors**, run the same way:
  - Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct and Mistral-Nemo-Instruct-2407 are in the HELM panel. They measure how far local quantised answers differ from HELM's hosted ones.
  - Qwen3-8B is not in HELM. It is the declared base of T-lite-it-2.1.
- Estimated total: about 76 GB of model files, and roughly 6–10 GPU hours.

**Analyses**, descriptive unless stated otherwise:
1. **Bridge.** For each anchor that is also in HELM: answer agreement and accuracy difference between the local run and HELM. A local-deployment effect would make all local models look alike. Analyses 2 and 3 therefore compare models within the local set, or against HELM models with the bridge stated.
2. **Inheritance.** The H1 statistic (excess same-wrong-answer agreement over N2, accuracy-matched) for each derivative–base pair against pairs from other local models:
   - T-lite-it-2.1 with Qwen3-8B;
   - Vikhr-Nemo and saiga_nemo with Mistral-Nemo;
   - saiga_yandexgpt with YandexGPT-5-Lite.

   The question is whether a "domestic" model built on a foreign base keeps that base's errors. If it does, it adds no independence against withdrawal of the base family.
3. **Admission and sovereignty.**
   - The gain from adding each Russian model to the open-weight portfolio of Section 19.
   - A domestic-only portfolio (n = 5) against the open-weight portfolio and the best open model.
   - The same comparison under withdrawal of all foreign open-weight models, as an extreme scenario.

**Limits.**
- Local quantised runs differ from hosted ones.
- English MMLU tests English knowledge; Russian models may do relatively better in Russian.
- These are 8–20B models only, set against a panel that includes frontier models.

## 27. Russian-model run: implementation details fixed before the full run (2026-11-14)

- [Line about the internal administration of the department server redacted for publication.]

**Prompts.** 14,038 exact HELM prompts, taken from Qwen2.5-7B-Instruct (release v1.9.0, modal prompt class). All prompt hashes match Step 0. HELM itself queried with temperature 0 and `max_tokens` 1.

**Pulled.** Eleven models went into the shared store: the eight Russian models of Section 26 plus Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct and Mistral-Nemo-Instruct-2407 (all Q4_K_M). Qwen3-8B was already present. Cotype-Nano is the official 8-bit GGUF; GigaChat-20B is Q5_0 and partly runs on the CPU.

**Format checks.** Each model was tested on the first 200 items, checking the share of valid answers only.
- **T-lite-it-2.1.** Its GGUF template always opens a `<think>` block whatever the reasoning switch says, so all 200 answers were invalid. It is now sent raw in the official Qwen3 non-thinking chat format (an empty `<think>` block): 0 invalid.
- **GigaChat-20B-A3B-instruct-v1.5.** Its GGUF has no chat template (`{{ .Prompt }}`), so 172 of 200 answers were invalid. It is now sent raw in its official chat template, taken from `tokenizer_config.json` on Hugging Face, including the BOS token: 0 invalid. A variant without the explicit BOS also gave 0 invalid; the template-faithful one is used.
- **All other models.** Their own templates work (0–4 invalid out of 200). Qwen3-8B honours `think: false`.

**Parsing amendment.** A leading `…</think>` block is removed before the first character is read. It applies to all models; it is needed only where a template leaks the tag.

Settings for the full run: 4 parallel requests (the service setting), context 4,096 tokens (the longest test prompt was 440 tokens; the prompt token count is recorded for every item), 8 output tokens, temperature 0, seed 0.

## 28. Russian-model run: corrections and results (2026-11-15)

### 28.1 Corrections after the full run (post hoc, stated as such)

The full run of Section 27 finished for all 12 models. Three faults were found **after** the full-run accuracies were seen. Each fix is limited to the model's prompt format or context window, and none touches the answer rule. The superseded files are kept in `data/ru_superseded/`.
- **saiga_yandexgpt_8b: wrong template.** It scored 0.290. Its GGUF build carries a ChatML template, but the official template (`tokenizer_config.json`) uses Llama-3-style headers. The model is re-run raw in the official format.
- **YandexGPT-5-Lite-8B-instruct: broken template.** Its official GGUF template is broken: it puts the literal text "Response }}" and an assistant turn before every prompt. The model is re-run raw in the official format: ` Пользователь: {prompt}\n\n Ассистент:[SEP]`. Accuracy 0.7294 → 0.7327; 92.5% of answers are unchanged.
- **Double BOS.** While testing these fixes, the Ollama log showed that Ollama adds the BOS token itself for GGUFs that ask for one. A manual `<s>` therefore doubles it, so the raw formats above omit it. GigaChat-20B (`gigachat-bos`, Section 27) is not affected: its GGUF does not ask for a BOS, and the log shows no warning for it. No double-BOS warnings appeared during the re-runs.
- **GigaChat3.1-10B: prompts too long.** Its tokenizer needs more tokens for English, and 61 prompts did not fit in the 4,096-token context: 60 failed with HTTP 400 and 1 was cut to 4,095 tokens. Those 61 items are re-run with an 8,192-token context; their longest prompt is 4,373 tokens. Accuracy is 0.7177.
- **Other models.** Their GGUF templates were checked against the official ones and kept. The Vikhr-Nemo and saiga_nemo builds differ from the official templates only in content trimming and the explicit BOS; Cotype-Nano adds no default system prompt in either.

`checks/cycle8_ru_run.py` (SHA-256 `7596de7e…7e`) gained the two formats and a `--num-ctx` option. `checks/cycle8_ru_analysis.py` (SHA-256 `a52a9bcd…2a`) is unchanged since it was tested on simulated answers. It ran once, on the corrected runs, in 99 s. Output: `checks/cycle8_ru_analysis.{out,json}`.

### 28.2 Accuracy (HELM MMLU, 14,038 items)

| Model | Role | Accuracy | Invalid |
|---|---|---|---|
| YandexGPT-5-Lite-8B | Russian | 0.733 | 0.0001 |
| T-lite-it-2.1 (Qwen3-8B base) | Russian | 0.724 | 0.0001 |
| GigaChat3.1-10B-A1.8B | Russian | 0.718 | 0.0002 |
| saiga_yandexgpt_8b | Russian | 0.663 | 0 |
| Vikhr-Nemo-12B (Mistral-Nemo base) | Russian | 0.654 | 0.0019 |
| saiga_nemo_12b | Russian | 0.639 | 0.0001 |
| GigaChat-20B-A3B v1.5 | Russian | 0.616 | 0.0001 |
| Cotype-Nano | Russian | 0.539 | 0.0003 |
| Qwen2.5-7B-Instruct | anchor | 0.713 | 0.0027 |
| Qwen3-8B (no thinking) | anchor | 0.673 | 0 |
| Mistral-Nemo-Instruct-2407 | anchor | 0.642 | 0 |
| Llama-3.1-8B-Instruct | anchor | 0.456 | 0.0380 |

The best Russian model, YandexGPT-5-Lite (0.733), is 13.5 points below the panel's best open model, DeepSeek-V3 (0.868).

### 28.3 Bridge (analysis 1)

| Local run vs HELM | Answer agreement | Accuracy local / HELM | x (same model) | Closest other HELM model to the HELM run: agreement, x |
|---|---|---|---|---|
| Qwen2.5-7B | 0.911 | 0.713 / 0.714 | 0.142 | Qwen2.5-72B: 0.768, 0.033 |
| Mistral-Nemo | 0.865 | 0.642 / 0.646 | 0.156 | Mistral Large 2407: 0.696, 0.023 |
| Llama-3.1-8B | 0.748 | 0.456 / 0.542 | 0.156 | Mistral Small 2402: 0.587, 0.004 |

- **Same model, two deployments.** A model run locally (quantised) and on HELM's hosted service keeps 87–91% of its answers and the same accuracy within 0.4 points; Llama is the exception.
- **Llama-3.1-8B is 8.7 points lower locally.** The cause is mostly format: in 3.8% of items it writes a sentence ("Here are the answers…") instead of a letter. Its error pattern is still recognisably the same (x 0.156).
- **No local-deployment effect.** Unrelated local–local pairs have a lower mean x (0.0096, 58 pairs) than local–HELM pairs within 0.05 accuracy (0.0138, 216 pairs). Running on the same server does not make models look alike, so analyses 2 and 3 are not inflated by deployment.
- **A ceiling for x.** The same model under two deployments gives x ≈ 0.14–0.16, two orders of magnitude above the S1 within-developer mean (T = 0.0046).

### 28.4 Inheritance (analysis 2)

x is each pair's excess same-wrong agreement. It is compared with the mean x of unrelated local pairs matched on both accuracies (±0.03, all pairs at the first tolerance), and tested with an N2 null (200 panels; p = 0.005 is the floor).

| Pair | Relation | x | Excess over matched | p | P(same answer \| both wrong) |
|---|---|---|---|---|---|
| Vikhr-Nemo ~ Mistral-Nemo | declared fine-tune | 0.086 | **0.078** | 0.005 | 0.743 |
| Vikhr-Nemo ~ saiga_nemo | siblings, same base | 0.072 | 0.063 | 0.005 | 0.706 |
| saiga_nemo ~ Mistral-Nemo | fine-tune by name | 0.052 | 0.037 | 0.005 | 0.655 |
| T-lite-it-2.1 ~ Qwen3-8B | declared fine-tune | 0.041 | 0.018 | 0.005 | 0.647 |
| saiga_yandexgpt ~ YandexGPT-5-Lite | fine-tune by name | 0.033 | 0.011 | 0.005 | 0.646 |
| GigaChat3.1-10B ~ GigaChat-20B | same developer | 0.030 | 0.010 | 0.005 | 0.624 |

- **All six pairs keep shared errors** beyond matched unrelated pairs and beyond the N2 null. The excess is 2–17 times the S1 within-developer T.
- **The Mistral-Nemo derivatives are closest to their base.** Vikhr-Nemo is about half-way to the same-model ceiling (0.086 against 0.14–0.16). When the two fine-tunes are both wrong, they agree on 71% of items.
- **T-lite keeps less of Qwen3-8B** (excess 0.018); it was trained further and is 5 points more accurate than its base.
- **The GigaChat pair** shares about as much as the YandexGPT fine-tune pair: a developer signature at the same scale as a fine-tune.
- **Answer to the question of Section 26:** yes. A "domestic" model fine-tuned on a foreign base keeps a large part of that base's errors. Vikhr-Nemo and saiga_nemo therefore add little independence against the loss of the Mistral family, and T-lite little against the loss of Qwen.

### 28.5 Admission and sovereignty (analysis 3)

Error is in %, with 95% subject-bootstrap CIs (200 resamples). Portfolios use the plurality rule with S1 cross-fitting.

- **Open-weight portfolio of the HELM panel** (open5): 13.77 [10.63, 17.77]. The best single open model, DeepSeek-V3, alone: 13.24.
- **Adding any Russian model to open5 raises the error** by between +0.08 (T-lite, CI [−0.10, +0.54]) and +0.31 (Vikhr-Nemo, CI [+0.17, +0.63]). Every other CI excludes 0.
- **Greedy re-selection with the Russian models in the pool picks none** (0 of 5 in both folds), so the portfolio is unchanged at 13.77.
- **Withdrawal of all foreign open-weight models** (domestic-only portfolio, domestic5): 24.85 [19.69, 32.00]. That is 11.1 points worse than open5. It beats the best single domestic model (YandexGPT-5-Lite, 26.73) by 1.9 points.
- **domestic5 is not fully domestic.** Both folds pick YandexGPT-5-Lite, T-lite-it-2.1, GigaChat3.1, saiga_yandexgpt and saiga_nemo. Two of these sit on foreign bases (Qwen3, Mistral-Nemo). By 28.4, saiga_yandexgpt shares errors with YandexGPT-5-Lite.
- **Consistency with the admission chart of S1b (Section 21).** A sixth model helps only near accuracy 0.85; the Russian models are at 0.54–0.73. The empirical result therefore matches the model-based prediction.

**Error chain from full availability to domestic only** (HELM MMLU, plurality portfolios):

| Availability | Error, % |
|---|---|
| All models | 11.06 |
| Open-weight only | 13.24–13.77 |
| Domestic only (8–20B, local) | 24.85 |

### 28.6 Limits

- **Size, not origin.** The domestic models here are 8–20B and run quantised. The gap to open5 is mostly a size gap: the foreign 7–12B anchors are at the same accuracy as the Russian models. T-pro (32B) and larger GigaChat models were not run.
- **English only.** HELM MMLU is in English; Russian-language performance is not measured.
- **Post-hoc corrections.** The template fixes of 28.1 were made after the first accuracies were seen. They are limited to restoring each model's official format, and the superseded runs are kept.
- **Llama's local accuracy** is depressed by answer format. It is reported, not corrected.
- **Not pre-registered: the own-base subset.** A portfolio restricted to Russian models on their own bases (YandexGPT, GigaChat, saiga_yandexgpt) was not pre-registered and was not computed.
- **Cotype-Nano's base is not declared.** Its chat format is Qwen-style, which is not evidence of its base.

## 29. Post-hoc robustness analyses of the Russian-model run: specification (2026-11-15, written before computing)

The authors asked for these two analyses after seeing Section 28, so both are **post hoc**. They do not change H1–H3, the primary results (Sections 19, 21, 25) or Section 28. Data: the 12 local runs and 70 HELM models of Section 28, with the same folds, plurality rule, cross-fitting and 200 subject-bootstrap resamples (the same resample seeds as Section 28). Script: `checks/cycle8_ru_posthoc.py`.

**P1. Size-matched origin comparison.** It asks whether the domestic–foreign gap of 28.5 is a size gap.
- **Groups.**
  - Russian 8–12B (6 models): YandexGPT-5-Lite-8B, T-lite-it-2.1, GigaChat3.1-10B-A1.8B, Vikhr-Nemo-12B, saiga_nemo_12b, saiga_yandexgpt_8b.
  - Foreign 7–12B (4 models): Qwen2.5-7B, Qwen3-8B, Llama-3.1-8B, Mistral-Nemo-12B.
  - Excluded: GigaChat-20B (20B) and Cotype-Nano (1.6 GB 8-bit file, far below 7B).
- **Outputs**, as group means and Russian − foreign differences with 95% CIs:
  - (a) accuracy;
  - (b) the change in open5 error when the model is added (the Section 28 admission measure, now also computed for the anchors);
  - (c) the cross-fitted error of a greedy n = 3 plurality portfolio drawn from each group.
- **Llama sensitivity.** Each output is repeated without Llama-3.1-8B, whose local accuracy is depressed by answer format (28.3).
- **Reading.** "No origin gap at matched size" if the CI of (a) includes 0.

**P2. Own-base domestic portfolio.** It answers the question of which domestic capacity remains if foreign base families are also withdrawn.
- **Pool:** the Russian models on Russian bases: YandexGPT-5-Lite-8B, saiga_yandexgpt_8b (fine-tuned from it), GigaChat3.1-10B and GigaChat-20B. Excluded: T-lite (Qwen3 base), Vikhr-Nemo and saiga_nemo (Mistral-Nemo base), and Cotype-Nano (base not declared).
- **Outputs**, as error in % with 95% CIs:
  - the best single model;
  - a greedy n = 3 plurality portfolio;
  - plurality over all four;
  - the difference from domestic5 of 28.5 (24.85).
- **Developer count.** Only two independent developers remain (Yandex with the Gusev fine-tune, and Sber), and 28.4 shows the fine-tune shares errors with its base.

## 30. Post-hoc robustness analyses: results (2026-11-15). POST HOC

Script `checks/cycle8_ru_posthoc.py` (SHA-256 `448cf49b…fb`), run once with 200 resamples in 30 s. A code test with 2 resamples printed point estimates before the full run; the point estimates do not depend on the resamples. Output: `checks/cycle8_ru_posthoc.{out,json}`. The CIs resample subjects only. They do not cover the choice of models, so with 6 and 4 models per group they understate the uncertainty of the group comparisons.

### P1. Size-matched origin comparison (values in %, 95% CI)

| Measure | Russian 8–12B (6) | Foreign 7–12B (4) | Difference | Without Llama: foreign (3) | Difference |
|---|---|---|---|---|---|
| (a) Mean accuracy | 68.85 | 62.10 | **+6.75** [+5.79, +7.82] | 67.61 | **+1.24** [+0.56, +1.97] |
| (b) Change in open5 error when added | +0.21 | +0.21 | +0.01 [−0.06, +0.11] | +0.22 | −0.01 [−0.08, +0.10] |
| (c) Greedy n = 3 portfolio error | 24.92 | 29.21 | −4.30 [−4.90, −2.27] | 27.92 | −3.01 [−3.70, −2.18] |

- **Reading rule (Section 29).** The rule said "no origin gap" only if the CI of (a) includes 0. It does not: at matched size the Russian models are **more** accurate, by 1.2 points without the format-affected Llama run and by 6.8 points with it. The gap that exists is in favour of the Russian models.
- **Admission is the same for both groups.** Adding a 7–12B model of either origin raises open5 error by about 0.2 points. Model size decides admission, not origin.
- **Three-model portfolios.** The Russian trio (YandexGPT-5-Lite, T-lite-it-2.1, GigaChat3.1 in both folds) beats the foreign trio (Qwen2.5-7B, Qwen3-8B, Mistral-Nemo) by 3.0 points. It includes T-lite, which is built on Qwen3.
- **Consequence for 28.5.** The 11-point gap between domestic5 and open5 is a gap between 8–20B local models and large open models, not between Russian and foreign models.

### P2. Own-base domestic portfolio (error in %, 95% CI)

| Portfolio | Error | Minus domestic5 (24.85) |
|---|---|---|
| Best single (YandexGPT-5-Lite) | 26.73 | +1.88 [+0.93, +2.61] |
| Greedy n = 3 | 26.40 | +1.55 [+0.06, +2.18] |
| Plurality of all four | 26.16 | +1.31 [+0.01, +2.19] |

- **Greedy n = 3 members.** Fold 0: YandexGPT-5-Lite, GigaChat-20B, GigaChat3.1. Fold 1: YandexGPT-5-Lite, saiga_yandexgpt, GigaChat3.1.
- **Effect of withdrawing the foreign base families as well.** Removing the Qwen- and Mistral-based Russian derivatives costs a further 1.3–1.9 points. Two of the three CIs barely exclude 0.
- **Diversification within the own-base pool is weak.** The best own-base portfolio is only 0.6 points better than its best single model. That pool has two independent developers (Yandex and Sber), and 28.4 shows the saiga_yandexgpt fine-tune shares errors with its base.

**Full error chain on HELM MMLU:**

| Models available | Error, % |
|---|---|
| All models | 11.06 |
| Open-weight only | 13.77 |
| Domestic only | 24.85 |
| Domestic on own bases | 26.16 |

The large step is from large open models to 8–20B models, not from foreign to domestic.

## 31. Positioning against prior work (2026-11-15)

Two literature passes. Notes, with full references and read levels, are in `research_notes/Cycle 8 positioning/llm_correlated_errors.md` and `research_notes/Cycle 8 positioning/reliability_diversity.md`. Read levels: [FT] full text, [FT-part] relevant parts, [SN] abstract or snippet only. Appendices of the long ML papers were not read, so every "not found" below covers only the text read.

### 31.1 Correlated LLM errors

**Kim, Garg, Peng, Garg, ICML 2025, arXiv 2506.07962 [FT-part]**
- **Measure and data.** They measure P(same wrong option | both wrong) against a constant chance level (1/3 on HELM MMLU), with no item-difficulty control. The authors list difficulty as a limitation. Their HELM data are the same benchmark as ours.
- **Effects.** Same company adds 0.022 on HELM and 0.066 on the Open LLM Leaderboard; same architecture adds 0.076.
- **What they do not do.** No declared fine-tune–base pairs; no ensembles; no withdrawal.
- **Consequence for H1.** H1's direction is a **replication** of Kim et al. What it adds:
  - a null with per-item difficulty (1PL-g) and pooled distractors;
  - an excess-over-null effect size, which is small (T ≈ 0.004);
  - the MMLU-Pro replication.

  The two scales are not comparable, so their 0.022 and our 0.0046 must not be set side by side as effect sizes. Our result answers the question their limitation leaves open: the company effect survives difficulty control, but it is small.

**Goel et al., ICML 2025, arXiv 2502.04313, CAPA [FT-part]**
- **Measure.** A chance-adjusted, accuracy-conditional agreement with a uniform-over-wrong-options chance term. It controls accuracy, not item difficulty.
- **Developer lineage.** They exclude same-developer pairs as a confound; they do not study it.
- **Consequence.** The idea of chance-adjusted error agreement is not ours. Our null is stricter on both counts (difficulty and distractor popularity).
- **Possible robustness check:** CAPA computed on our data.

**ErrorTrace (Zang et al., NeurIPS 2025, DOI 10.52202/085713-2306) [FT-part]**
- **What it does.** It attributes a model to its base family from binary error sets, at 85.18% accuracy for unseen models, and also traces fine-tuned, pruned and merged variants.
- **What it does not do.** It reports no difficulty null and no effect size for shared errors between derivative and base. Its data are NLP classification tasks, not MMLU.
- **Consequence.** It supports the inheritance result qualitatively. Our magnitudes (0.078, 0.063, 0.018, and the ceiling of 0.14–0.16) are new. We do not offer an attribution method.

**PhyloLM (Yax, Oudeyer, Palminteri, ICLR 2025, arXiv 2404.04671) [FT-part]; LLM DNA (ICLR 2026) [SN]**
- These recover lineage from output similarity, without regard to correctness.
- They are complementary: they show that lineage is visible in outputs, and we show that it is visible in shared **wrong** answers beyond difficulty.

**Turkmen, Buyukates, Bastopcu, arXiv 2602.08003 [FT-part]**
- **What it does.** Greedy mutual-information ensemble selection under copula-correlated binary errors, on 12–13 models.
- **What it does not do.** No developer groups, withdrawal or caps in the text read.
- **Consequence.** Our withdrawal-robust portfolio analysis is new relative to it. We do not claim a better selection algorithm, since we did not compare against theirs.

**"Li et al. 2025" on lineage**, cited in the cycle-8 discovery notes, could not be identified and is **not cited**.

### 31.2 Reliability engineering and ensemble diversity

**Withdrawal measures and their reliability counterparts:**

| Reliability concept | Source (read level) | Our measure |
|---|---|---|
| Loss PV0 − PV2 in the resilience ratio (recovery/loss) | Henry & Ramirez-Marquez 2012, RESS 99:114–122 ([SN]; formula from the 2010 INCOSE precursor [FT]) | ΔE |
| Recovery | same | Not measured. A re-selection scenario (error after withdrawal minus error after re-optimising) would supply it. |
| Risk achievement worth | Vesely et al. 1986, via Borgonovo & Smith 2012 ([SN]) | RAW = E(P\W)/E(P), with group withdrawal as the event |
| Joint reliability importance, J = h(1,1) − h(0,1) − h(1,0) + h(0,0) | Hong & Lie 1993 ([SN]); Armstrong 1995 | JI, its finite-difference analogue on error: the departure of group ΔE from additivity |
| Differential importance measure, additive over groups to first order | Borgonovo & Apostolakis 2001 ([SN]; read via a Borgonovo working paper [FT-part]) | Explains why JI is needed beside ΔE |
| Common-cause failure (beta factor) | NUREG/CR-4780, 1988 ([SN]) | H1 |

- **Classification as resilience or robustness.** Our measures are static, so they describe the "impact" or robustness side of resilience. This is our inference: Hosseini et al. 2016 (RESS 145:47–61) was read only in part, and its classification section was not read.
- **The beta-factor analogy.** H1 is a common-cause effect: models of one developer co-fail beyond what difficulty predicts.
- **Coincident failure in multiversion software** (Eckhardt & Lee 1985; Littlewood & Miller 1989; Knight & Leveson 1986, all [SN]):
  - Independently developed versions still fail together on hard inputs.
  - In our design, that coincidence is the **null model** (shared item difficulty), and the developer effect is what remains beyond it.
  - "Forced diversity" (Littlewood–Miller) is the design-side counterpart of a developer cap. Its benefit depends on assumptions that failed in Feldt's 1998 experiment [FT-part]. This parallels H3: diversity-optimised selection does not beat a simple one-per-developer cap.

**How this differs from conventional ensemble diversity** (Kuncheva & Whitaker 2003 [FT-part]; Brown et al. 2005 and Krogh & Vedelsby 1995 [SN]):
- Diversity measures there are computed on a fixed set of models, without conditioning on difficulty, and they serve to explain ensemble accuracy. They mix every source of coincidence.
- We separate three things:
  1. coincidence explained by item difficulty (the null);
  2. common-cause dependence from shared provenance: developer (H1) and lineage (28.4);
  3. the loss of a whole provenance group (ΔE, RAW, JI).
- In the sources read, none of them separates (2) from (1) or measures (3).

### 31.3 What the paper can claim

**Replication**
- Same-developer error correlation, now under a difficulty-controlled null: small but present on two benchmarks.

**New, empirical**
1. Measured withdrawal costs: API-only withdrawal 2.2 points (MMLU) and 5.2 points (MMLU-Pro); domestic-only +11 points over open5 on MMLU, which is a size gap by Section 30.
2. A one-per-developer cap is not beaten by diversity-optimised selection, on two benchmarks.
3. Magnitudes of error inheritance by fine-tunes, including two independent fine-tunes of one base, against a same-model ceiling.

**New, framing**
- Borrowing importance measures (RAW, joint importance) and the common-cause concept from reliability engineering for portfolios of AI providers.

**Not claimed**
- Novelty of chance-adjusted error agreement.
- A new selection algorithm.
- Russian-language performance.

**Sources still to read before citing as read:** Henry & Ramirez-Marquez 2012 (equations), Hosseini et al. section 3, Borgonovo & Apostolakis 2001, the original RAW and beta-factor sources, Eckhardt & Lee, Littlewood & Miller, and Brown et al. These are IEEE and Elsevier papers, so full texts need library access.

## 32. Inheritance at scale on the Open LLM Leaderboard: pre-registration (2026-11-15, written before any outcome is computed)

**Why.** The authors planned an international journal submission. The inheritance result of 28.4 rests on six pairs, so this section scales it up.

**Data and what has been seen.**
- **Item-level answers are gated.** The leaderboard's per-item answers (`open-llm-leaderboard/*-details`) require a logged-in account that accepts each dataset's terms (gated "auto"). That needs a personal account, so this section does not use them.
- **What it uses instead.** The public pair table of Kim et al. 2025 (repository `nikhgarg/llm_correlated_errors_public`, file `data/hugging_face/hf_featured_models.csv`). It gives MMLU-Pro answer statistics for all 60,726 pairs of 349 leaderboard models.
- **Lineage** is joined from the leaderboard's own metadata (`open-llm-leaderboard/contents`, field "Base Model", the declared base at the root of the model tree; 4,576 models).
- **Licence.** The repository has no licence. The table is used for analysis with citation and is not redistributed.
- **Seen before writing this section:**
  - model and pair counts: 349 models, 32 parent–child pairs over 8 bases, 737 sibling pairs over 15 bases;
  - the column names;
  - one row (a same-developer sibling pair, agreement when both are wrong 0.937).

  No class means or comparisons were computed.

**Pair classes** (both models must have metadata; the root base of a model that declares none is the model itself):
- **L3 parent–child:** one model's declared base is the other model.
- **L2 siblings:** the same declared base, and neither model is that base.
- **L1 same architecture class, different base:** the same `architecture` field, different root bases.
- **L0 unrelated:** different architecture class and different root base.

**Statistic.** *s* = P(same answer | both wrong), the table's `agreement_rate_when_both_wrong`.
- For each pair in L1–L3, the comparison set is the L0 pairs whose two accuracies each lie within ±0.02 of the pair's accuracies, in either order. If fewer than 10 pairs match, the tolerance doubles, up to 0.08; with fewer than 10 at that point the pair is dropped.
- The pair's excess is *s* minus the mean *s* of its comparison set. Accuracy matching takes the place of the item-level null: shared item difficulty raises *s* through accuracy, and the matching removes that.

**Hypotheses** (primary, with the order prediction fixed):
- **I1.** Mean excess is positive for L3 and for L2.
- **I2.** The classes are ordered L3 > L2 > L1: lineage adds shared errors beyond architecture class.
- **I3, dose–response, descriptive.** Within L3, excess falls with the child's generation depth, and it differs by the child's leaderboard type (fine-tuned vs chat/RLHF vs continued pretraining).

**Uncertainty.**
- 95% intervals from a cluster bootstrap over root-base families (1,000 resamples, seed 20261115). Comparison sets are recomputed in every resample.
- **I1 is supported** for a class if the lower bound of its interval exceeds 0.
- **I2 is supported** if the interval of each difference (L3 − L2, L2 − L1) is above 0.
- **Few clusters.** L3 has only 8 base families, so its interval is reported with that caveat.

**Sensitivity.**
1. Restrict to pairs with equal parameter count (`param_diff` = 0) for L1–L3.
2. Use `agreement_rate_when_either_wrong` as the statistic.
3. Drop the largest family (Qwen2.5-14B) from L2 and L3.

**Not claimed.**
- An item-level, difficulty-null estimate; that remains the 12-run result of 28.4.
- Causal attribution to training method.

**Script:** `checks/cycle8_olb_inheritance.py`.

## 33. Inheritance at scale: results (2026-11-15)

Script `checks/cycle8_olb_inheritance.py` (SHA-256 `a8095807…f4`), run once with 1,000 family-bootstrap resamples in 532 s. Output: `checks/cycle8_olb_inheritance.{out,json}`. A code test on shuffled statistics ran first; its outputs were deleted.

**Data.** 60,031 pairs had metadata. Class sizes:

| Class | Pairs |
|---|---|
| L0 unrelated | 41,130 |
| L1 same architecture, different base | 18,132 |
| L2 siblings | 737 |
| L3 parent–child | 32 |

Six target pairs had no match even at ±0.08 and were dropped; every other target pair matched at the first tolerance (±0.02).

**Primary results.** Excess of *s* = P(same answer | both wrong) over accuracy-matched unrelated pairs; 95% family-bootstrap CI.

| Class | Excess | Families |
|---|---|---|
| L3 parent–child | **+0.167** [+0.114, +0.192] | 8 |
| L2 siblings | **+0.101** [+0.061, +0.288] | 15 |
| L1 same architecture, different base | +0.077 [+0.063, +0.092] | 253 |
| L3 − L2 | +0.066 [−0.146, +0.115] | |
| L2 − L1 | +0.023 [−0.024, +0.211] | |

- **I1: SUPPORTED** for both L3 and L2. When a fine-tune and its declared base are both wrong, they pick the same wrong option about 17 points more often than unrelated models of the same accuracies. For comparison, Kim et al.'s same-company coefficient on this leaderboard is 0.066, on a regression scale.
- **I2: NOT SUPPORTED.** The point estimates are in the predicted order, but neither difference is resolved.

**Sensitivity**

| Check | L1 | L2 | L3 | Note |
|---|---|---|---|---|
| S1 equal parameter count | 0.246 | 0.303 | 0.155 | Order reverses: siblings > parent–child, L3 − L2 −0.148 [−0.225, −0.087] |
| S2 agreement when either wrong | 0.078 | 0.107 | 0.163 | Same pattern as primary |
| S3 without the Qwen2.5-14B family | 0.077 | 0.082 | 0.151 | Same pattern as primary |

I1 holds in all three. I2 fails in all three.

**I3 (descriptive)**
- **By the child's generation depth:** 1: +0.159 (n 12); 2: +0.161 (n 14); 3: +0.193 (n 6). There is no sign that inheritance fades further down the model tree.
- **By type:** chat/RLHF children +0.157 (n 20); domain fine-tunes +0.182 (n 12).

**Post-hoc diagnostic (not pre-registered).** It asks why L1 is so high.
- Only 41 L1 pairs have roots of the same model family by name (for example X and X-Instruct). Their excess is 0.267.
- The other 18,086 L1 pairs still show +0.077.
- Among those, roots from the same organisation give 0.050 and roots from different organisations give 0.079.
- **Reading.** Most of the 349 models are community fine-tunes, and many declare no base, so their root is themselves. "Same architecture class" then largely stands for undeclared lineage (for example two undeclared Llama-3 derivatives). L1 is therefore not an architecture-only baseline.
- **Consequence.** The leaderboard metadata cannot separate lineage from architecture class, so I2 cannot be tested cleanly with these data. That is a limitation of the declared metadata, not evidence against an ordering.

**Reading for the paper**
- **Error inheritance is general.** The six-pair, item-level result of 28.4 generalises to a 349-model population, with a large effect (about +0.15 to +0.17 in conditional same-wrong agreement) that does not decay with tree depth.
- **Siblings vs base (post hoc, from S1).** At equal size, two fine-tunes of one base share more errors with each other than either does with the base. This is consistent with fine-tuning moving models in a common direction away from the base.
- **Limits.**
  - Pair-level statistics with accuracy matching, not an item-level difficulty null.
  - Only 8 base families in L3, so its interval rests on few clusters.
  - The "Base Model" field is self-declared and incomplete.

## 34. Post hoc: the H1 contrast under alternative agreement measures (2026-11-15). POST HOC

**Why.** A referee will ask whether H1 depends on our N2 excess measure, and how it relates to Kim et al. (conditional agreement) and Goel et al. (CAPA).

**Run.**
- Script `checks/cycle8_h1_measures_posthoc.py` (SHA-256 `3b4b2c27…05fd`), HELM MMLU, 70 models, 200 N2 panels, 171 s. Output: `checks/cycle8_h1_measures_posthoc.{out,json}`.
- The same developers and the same accuracy and prompt-class matching as H1. CAPA is computed with discrete answers.
- The matched x contrast reproduces the pre-registered T = 0.00462 exactly.

| Measure | Matched: observed | Matched: N2 mean | z | Unmatched: observed | Unmatched: N2 mean | z |
|---|---|---|---|---|---|---|
| x, excess over N2 (H1) | 0.00462 | 0.00005 | 14.1 | 0.00036 | 0.00013 | 0.8 |
| Raw same-wrong rate | 0.00581 | 0.00118 | 13.6 | 0.00624 | 0.00587 | 1.2 |
| P(same \| both wrong) | 0.01756 | −0.00034 | 9.6 | 0.01422 | −0.00178 | 9.4 |
| CAPA | 0.01850 | −0.00026 | 16.8 | 0.01028 | −0.00060 | 10.5 |

**Reading**
- **Matched contrasts.** With accuracy and prompt matching, every measure detects the developer effect far beyond the N2 null (z 9.6–16.8). H1 does not depend on the choice of measure.
- **Unmatched contrasts.** Without matching, the raw same-wrong rate is fully explained by shared difficulty and ability (observed 0.0062 against an N2 mean of 0.0059). The x measure also loses the effect, because similar-ability pairs share wrong answers on items with wrong keys (cf. A2.1). The accuracy-conditional measures (conditional agreement, CAPA) keep it.
- **Consequence for the claim.** The essential control is accuracy matching or an accuracy-conditional measure; with matching, the N2 null itself is not needed to detect the effect. The paper must not claim that our null is the only valid control. Its advantage is calibration: every measure can be tested against the same simulated difficulty-only world.

## 35. Post-hoc robustness analyses requested by the internal referee review: specification (2026-11-15, written before computing). POST HOC

The referee-style review of the manuscript found claims stronger than the outputs support. These analyses were written down before any of them was computed. They do not change any earlier decision.

**R1. A like-for-like sovereignty cost on both benchmarks.**
- C = E(best open-weight portfolio, n = 5, greedy plurality on open models) − E(diversity-cap portfolio under full availability). The cap is the best full-availability selection of H3 on both benchmarks.
- Secondary: C′ = min(open5, best single open model) − min(cap, accuracy-greedy, best single model).
- Intervals: paired percentile intervals over the existing bootstrap resamples (MMLU 200 subject resamples; MMLU-Pro 200 item resamples, approximate).
- Script: `checks/cycle8_sovereignty_cost.py`.

**R2. Origin vs scale with own-base models only, plus model resampling.**
- The P1 comparison of Section 29 is repeated with the Russian group restricted to own-base 8–12B models (YandexGPT-5-Lite, saiga_yandexgpt, GigaChat3.1) against the foreign anchors, with and without Llama.
- A leave-one-model-out range of the group difference is reported for both the original and the own-base versions.
- Added to `checks/cycle8_ru_posthoc.py` as a new output block; earlier outputs are unchanged.

**R3. Leaderboard placebo, depth intervals and within-family triplets.** Data and matching as in Section 32.
- **Placebo.** For each L3 (and L2) pair, take the L1 pairs whose two models' parameter counts each lie within ±10% of the target pair's, in either order (same architecture class, different declared base, same sizes). Report the mean excess of these placebo pairs. Also report L3 − placebo and L2 − placebo with family-bootstrap 95% intervals (1,000 resamples, seed 20261116).
- **Depth.** L3 excess by the child's generation depth, with family-bootstrap intervals.
- **Triplets.** For every base b with two children c1, c2 that are both in L3, compute excess(c1, c2) − mean(excess(c1, b), excess(c2, b)). Report the mean over triplets with a family-bootstrap interval.
- Script: `checks/cycle8_olb_robustness.py`.

## 36. Post-hoc robustness analyses: results (2026-11-15). POST HOC

**R1. Like-for-like sovereignty cost** (`checks/cycle8_sovereignty_cost.{py,out,json}`). C = best open-weight 5-model portfolio minus the diversity-cap portfolio under full availability.

| Benchmark | C, points | C′, points |
|---|---|---|
| MMLU | 2.86 [1.89, 3.41] | 2.33 [1.46, 2.96] |
| MMLU-Pro | 5.20 [2.90, 7.90] | 5.20 [3.00, 7.60] |

The MMLU-Pro intervals are approximate. The earlier MMLU figure of 2.19 is a different quantity: the ΔE of the greedy primary portfolio. The paper now reports C on both benchmarks and keeps ΔE for the withdrawal-measure tables.

**R2. Origin vs scale** (`checks/cycle8_ru_posthoc_r2.{py,out,json}`; a separate script instead of a new block in `cycle8_ru_posthoc.py`, so the earlier output is untouched).
- **Own-base Russian 8–12B (3 models) vs anchors without Llama:**
  - accuracy +2.84 points [1.95, 3.74], leave-one-model-out range [+1.15, +4.91];
  - admission change −0.00 [−0.09, +0.15].
- **All six Russian 8–12B vs anchors without Llama:** accuracy +1.24 [0.56, 1.97], but the leave-one-model-out range is [−0.45, +3.09]. Wording for this group: "at least as accurate".
- **Reading:** at 8–12B there is no origin deficit. Larger domestic models were not run.

**R3. Leaderboard robustness** (`checks/cycle8_olb_robustness.{py,out,json}`). Family bootstrap, 1,000 resamples.

| Comparison | Excess | Families |
|---|---|---|
| L3 (parent–child) | +0.161 [0.132, 0.177] (n 31) | 8 |
| Placebo for L3 (same architecture, same size ±10%, different declared base) | +0.180 [0.117, 0.224] | |
| **L3 − placebo** | **−0.020 [−0.052, +0.025]** | |
| L2 (siblings) | +0.111 | |
| Placebo for L2 | +0.098 | |
| L2 − placebo | +0.012 [0.002, 0.097] | |
| Depth 1 | +0.159 [0.116, 0.182] | 7 |
| Depth 2 | +0.161 [0.101, 0.189] | 6 |
| Depth 3 | +0.193 | 1 (no interval) |
| Triplets: siblings − mean(child–base) | **+0.054 [0.045, 0.140]** (112 triplets) | 6 |

**Reading**
- **Parent–child pairs vs the placebo.** They share no more errors than same-architecture, same-size pairs with a different declared base. On the leaderboard such pairs are typically derivatives of the same pretrained model (declared differently or not at all) or of the same developer's model of that size.
- **What the population data show.** Error sharing follows the pretrained model family (architecture and size). A declared fine-tuning link adds nothing measurable beyond that.
- **Section 33 must be reworded.** Its I1 result stands (excess over unrelated pairs), but it is not evidence for an effect of declared lineage beyond architecture and size.
- **Triplets.** Within a family, two fine-tunes of one base share more errors with each other than with the base. This holds within triplets, not only across class means (6 families).
- **Depth.** The depth pattern is flat where it can be estimated (depths 1 and 2); depth 3 has one family.

## 37. Second referee re-check and last diagnostics (2026-11-15). POST HOC

**Verdict.** The re-check found the revision close to minor. Remaining issues:
- the leaderboard conclusion overstated R3;
- the scale claim in the abstract was unhedged;
- four small inconsistencies (9.6–16.8 SDs; two different N2 null draws; figure values from different samples; "items" instead of "subjects").

All of them are fixed in the manuscript.

**Diagnostics** (computed inline from `checks/cycle8_olb_robustness.py` objects; not a separate script):
- **Placebo pairs.** 2,889 distinct placebo pairs serve the 31 L3 targets. We read 20 randomly drawn pairs (seed 1) by model name:
  - about 11 are derivatives of the same pretrained model (Qwen2.5-14B merges and fine-tunes, Gemma-2-9B, Qwen2-72B);
  - about 6 are models of the same developer's family at that size (Llama 3 vs 3.1/3.3 70B; Qwen2 vs Qwen2.5 72B);
  - 1 crosses families;
  - 2 are unclear.

  This is a manual reading by name, not a verified lineage check.
- **Triplets per family:**

| Family | Triplets | Mean |
|---|---|---|
| Qwen2.5-14B | 91 | +0.044 |
| Qwen2.5-72B | 10 | +0.058 |
| Qwen2-72B | 6 | +0.153 |
| Qwen2.5-32B | 3 | +0.086 |
| Llama-3-70B | 1 | +0.113 |
| Mistral-Small-2409 | 1 | +0.100 |

  The difference is positive in all six families, with a family-weighted mean of +0.093.

**Wording now used.** Error sharing tracks the architecture-and-size class of a model family, which mostly means a common pretrained model or developer. An extra effect of a declared parent–child link is not detected with 8 families. Siblings exceed their placebo slightly.

**Local-run correction policy.** A packaged chat template was replaced only where it was missing, broken or different from the official template on Hugging Face:
- saiga_yandexgpt and YandexGPT-5-Lite (both re-run);
- GigaChat-20B and T-lite (fixed before the full run, Section 27).

Llama-3.1-8B uses its official template, so its sentence answers are model behaviour and were not corrected. The rule is template fidelity, not origin.

## 38. Post hoc: null-model comparison with 100 panels (2026-11-15). POST HOC

**Run.** Script `checks/cycle8_null_panels_100.py` (SHA-256 `7e89af51…bdb1`). It uses the S1b code and seed scheme, and panels 0–19 reproduce S1b exactly. Laptop, 39 s. Output: `checks/cycle8_null_panels_100.{out,json}`.

| Null | Full error, % | API-only ΔE, points | Panels at or below observed ΔE |
|---|---|---|---|
| N1 | 1.22 [1.05, 1.38] | 10.23 [4.42, 17.37] | 0% |
| N2 | 10.00 [9.64, 10.37] (all 100 below the observed 11.06) | 2.59 [2.00, 2.91] | 12% |
| Observed | 11.06 | 2.19 | |

**Corrections to Section 21's reading**
- **N1:** it still overstates the withdrawal cost about 4.7-fold.
- **N2:** it does **not** clearly overstate the withdrawal cost. With 20 panels the observed value lay below the 5th percentile; with 100 it lies at the 12th. "Overstates by about a third" is withdrawn.
- **What stands:** N2 overstates ensemble accuracy in every panel.
- The manuscript is updated accordingly.

## 39. MMLU-Redux exclusion sensitivity: specification (2026-11-15, written before computing)

This sensitivity analysis was pre-specified in Section 8 and not run until now; it is weak point 4 in `docs/weak-points.md`. The rules below were written after downloading the re-annotation and counting its labels, before any HELM answer was joined to it.

**Data.**
- MMLU-Redux 2.0 (Gema et al., NAACL 2025; `edinburgh-dawg/mmlu-redux-2.0`, CC BY 4.0): 5,700 re-annotated MMLU test items, 100 per subject in 57 subjects.
- Label counts: ok 5,330; bad_question_clarity 132; wrong_groundtruth 106; multiple_correct_answers 39; no_correct_answer 36; expert 32; bad_options_clarity 25.
- Stored in `data/redux/mmlu_redux_2.0.json.gz`.

**Matching to HELM items.**
- The HELM item text is the target question of the exact HELM prompt (`data/helm_mmlu_prompts.jsonl.gz`): the text after the last "Question: " and before "\nA. ".
- It is matched to the Redux question within the same subject, after lower-casing and collapsing whitespace.
- The match rate is reported. Unmatched Redux items are ignored. A Redux question that matches several HELM items flags all of them.

**Flagged items:** matched items whose `error_type` is not "ok".

**Primary sensitivity (as pre-specified in Section 8).** The S1 analyses are repeated on all 14,038 items minus the flagged ones:
- **H1:** T, p (200 N2 panels) and the developer jackknife, with the same decision rule and MDE (0.0028).
- **Descriptives:** primary-portfolio error; ΔE for API-only and each developer; RAW; sovereignty cost C; H3 (cap vs robust, held-out worst case); H2 (spread of the voting rules).

  Excluded items get weight 0, so they influence neither fitting, nor selection, nor evaluation.
- **Intervals:** 200 subject-bootstrap resamples, with the same seeds as S1, multiplied by the exclusion mask.

**Secondary.** The same point estimates with only the wrong_groundtruth and no_correct_answer items excluded: the two labels that make the key itself wrong.

**Reading.** Conclusions stand if the H1 verdict, the H2/H3 verdicts and the sign and rough size of ΔE(API-only) and C are unchanged. Any change is reported as found.

**Script:** `checks/cycle8_redux_sensitivity.py`.

## 40. Leaderboard lineage from model cards: specification (2026-11-15, written before computing). POST HOC

This is weak point 6a. The leaderboard's "Base Model" field is a derived root and is often missing or self-referential.

**Relabelling.**
- For each of the 349 models, read `cardData.base_model` from the public Hugging Face model API (`/api/models/<id>`; no login). The field can be a string or a list.
- The card parent of a model is the single listed base. A model listing several bases is a merge, with all of them as parents.
- The **card root** follows card parents upwards, through models of any kind, until a model lists no base, the chain loops, or an API error occurs. At most 10 steps; loops and errors are counted.

**Classes,** computed the same way as in Section 32 but with card lineage:
- **C3 direct parent–child:** one model is the other's card parent.
- **C2 card siblings:** the same card root, with neither model the root and no direct parent link.
- **C1 same architecture, different card root.**
- **C0 unrelated:** different architecture and different card root.

**Measure and inference.** The same accuracy-matched excess of P(same | both wrong) over C0 pairs, with a 1,000-resample family bootstrap over card roots (seed 20261117). Also the placebo of Section 35 R3 applied to C3 and C2 (same architecture, same size ±10%, different card root).

**Reading.**
- If C3 − placebo is now positive with an interval above 0, declared lineage carries information beyond the architecture-and-size class once it is measured better.
- If not, the Section 36–37 reading stands.

**Script:** `checks/cycle8_olb_cardlineage.py`. The API answers are cached in `data/olb/card_base_models.json`.

## 41. Leaderboard lineage from model cards: results (2026-11-15). POST HOC

**Run.** Script `checks/cycle8_olb_cardlineage.py`, first run SHA-256 `71c22afa…865d`. After that run, a split of C3 into fine-tune and merge links was added (final SHA-256 `0cd2c735…236a`); it changes no earlier output. 1,000 resamples, 103 s. Output: `checks/cycle8_olb_cardlineage.{out,json}`. Of the 349 models, 213 have a `base_model` in their card; 47 API calls failed (gated or removed models).

| Class (card lineage) | Excess of P(same \| both wrong) over matched C0 | Pairs | Families |
|---|---|---|---|
| C1 same architecture, different card root | +0.075 [0.067, 0.082] | 18,157 | |
| C2 card siblings | +0.220 [0.179, 0.249] | 421 | 15 |
| C3 direct parent (fine-tune or merge) | **+0.309 [0.279, 0.341]** | 168 | 64 |
| C3 − placebo (same architecture, same size, different card root) | **+0.101 [0.073, 0.134]** | 161 | |
| C3 fine-tune only − placebo | **+0.101 [0.061, 0.162]** | 65 | 17 |
| C3 merge only − placebo | +0.102 [0.072, 0.131] | 96 | |
| C2 − placebo | +0.007 [−0.011, 0.060] | | |

**Triplet diagnostic with card lineage** (inline; a base with two direct fine-tune children; not pre-specified):
- 123 triplets in 12 families; the family-weighted mean of excess(c1, c2) − mean(excess(c, base)) is **−0.021**, positive in only 3 families.
- The three positive families are exactly those whose parent is a pretrained base model (Qwen2.5-14B +0.074, Qwen2.5-72B +0.125, Qwen2-72B +0.014).
- Where the parent is an instruction-tuned model, the result is negative (for example Qwen2.5-14B-Instruct −0.045, n 45; Qwen2.5-7B-Instruct −0.040, n 28; Mistral-Small-Instruct −0.065).

**Reading (this supersedes the leaderboard reading of Sections 36–37)**
- **The earlier null was a labelling problem.** With lineage taken from the models' own cards (more pairs, 64 families instead of 8), a direct fine-tuning link raises shared wrong answers by about 0.10 beyond what the architecture-and-size class explains, and merges do the same. The "nothing beyond architecture and size" result of Section 36 came from the leaderboard's derived root field, which lumps distant descendants and mislabels roots.
- **Card siblings** do not exceed their placebo.
- **The sibling-versus-base pattern depends on the parent.**
  - Fine-tunes of a *pretrained* base resemble each other more than the base: common instruction tuning moves them the same way.
  - Fine-tunes of an *instruction-tuned* model resemble that parent more than each other.
  - The earlier "siblings share more than child–base" result was dominated by pretrained-base families (Qwen2.5-14B). It holds only for them.
- **Implication for the paper.** Inheritance at population scale is now supported directly: a fine-tune keeps a measurable part of its parent's error pattern beyond its family. The triplet claim is restated as conditional on the parent type, descriptive only.

## 42. Developer jackknife of the main withdrawal quantities: specification (2026-11-15, written before computing). POST HOC

This is weak point 10: the intervals so far resample items, not models.
- **Procedure.** For each developer d in the panel, the full pipeline is rerun on the panel without d's models: selection, cap and the open portfolio are all re-selected; point estimates only. Done on MMLU (70 models, 18 developers) and MMLU-Pro (58 models, 15 developers).
- **Quantities:**
  - the sovereignty cost C = open5 − cap(full);
  - ΔE(API-only) of the re-selected primary portfolio, on MMLU only, because on MMLU-Pro an all-API fold portfolio leaves it undefined;
  - the H3 gain (cap minus robust, held-out worst case).
- **Reported:** the range over the leave-one-out panels, and the jackknife standard error √((k−1)/k · Σ(θ₍₋d₎ − θ̄)²).
- **Reading:** a conclusion is fragile if some leave-one-out panel changes its sign.

**Script:** `checks/cycle8_dev_jackknife.py`.

## 43. MMLU-Redux exclusion sensitivity: results (2026-11-15)

**Run.** Script `checks/cycle8_redux_sensitivity.py` (SHA-256 `35b0f4d8…3004`), department server with 16 workers, 4,748 s. Output: `checks/cycle8_redux_sensitivity.{out,json}`. Matching: 5,686 of 5,700 Redux items (99.8%); 374 HELM items flagged (not "ok"); 144 with a wrong or missing key.

| Quantity | Without flagged items | S1 (all items) | Without key-wrong items only |
|---|---|---|---|
| H1 T | **0.00447**, p 0.005, jackknife 90% CI [0.00151, 0.00743] → **SUPPORTED** | 0.00462 | |
| Primary portfolio error | 10.19 [7.81, 13.72] | 11.06 | 10.47 |
| ΔE API-only | 2.26 [1.36, 4.13] | 2.19 | 2.22 |
| Sovereignty cost C | 2.85 [1.99, 3.50] | 2.86 | 2.89 |
| H3 gain (robust vs cap) | −0.12 [−0.68, 0.27] | −0.16 | −0.14 |
| ΔE Anthropic | 0.61 | 0.59 | |
| ΔE DeepSeek | 0.47 | 0.46 | |
| ΔE Google | 0.42 | 0.38 | |
| ΔE OpenAI | 0.23 | 0.27 | |
| ΔE 01.AI | 0.09 | 0.10 | |

**H2 (voting-rule spread, full / worst case):** n = 3: 0.07 / 0.31; n = 5: 0.05 / 0.04; n = 7: 0.18 / 0.04; n = 9: 0.33 / 0.53. The rules stay within about half a point.

**Reading.**
- By the rule of Section 39, every conclusion stands: the H1, H2 and H3 verdicts, and the sign and size of ΔE(API-only) and C.
- Mislabelled items raise all error levels by about 0.9 points and change nothing else.
- The pre-specified MMLU-Redux sensitivity of Section 8 is now delivered.

## 44. Developer jackknife: results (2026-11-15). POST HOC

**Run.** Script `checks/cycle8_dev_jackknife.py` (final SHA-256 `372d9bf0…cfceb`). A first local run crashed while formatting the report, after computing everything and before saving it. The report code was fixed (raw results now saved first; an undefined quantity is skipped) and the script rerun unchanged otherwise on the server: 16 workers, 384 s. Outputs: `checks/cycle8_dev_jackknife.{out,json}` and `checks/cycle8_dev_jackknife_raw.json`.

Leave-one-developer-out results, with the full pipeline re-selected on each reduced panel:

| Quantity | Full panel | Leave-one-out range | Jackknife SE | Sign changes |
|---|---|---|---|---|
| MMLU sovereignty cost C | 2.86 | 2.15 (without Anthropic) to 3.48 (without DeepSeek) | 1.23 | 0 of 18 |
| MMLU ΔE(API-only) | 2.19 | 1.68 (without Google) to 7.62 (without DeepSeek) | | 0 |
| MMLU H3 gain | −0.16 | −0.88 to +0.07 | | 3 |
| MMLU-Pro C | 5.20 | 4.10 (without IBM) to 6.80 (without DeepSeek) | 2.21 | 0 of 15 |
| MMLU-Pro H3 gain | −2.00 | −2.80 to −1.00 | | 0 |

**Reading**
- **C.** The sovereignty cost is positive whichever developer is removed, so the cost of open-weight-only operation does not hinge on one developer.
- **ΔE(API-only).** Its size depends strongly on DeepSeek. Without DeepSeek, no strong open-weight model remains, and losing the hosted models costs 7.6 points.
- **H3.** The refutation is robust on both benchmarks: the largest gain of robust selection over the cap is +0.07 points, far below the 1-point bar.
- **Jackknife SEs.** They are large because single developers are influential. They show that model sampling dominates the uncertainty, and they are reported as such.
