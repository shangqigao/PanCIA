# PanCIA main.tex (v7 + G1): review of remaining weaknesses (18 Oct 2026)

This review covers `documents/main.tex` (md5 c9a3ac), the version after fixes 1–3 and π_two = 0.75. It reads the manuscript as a reviewer would. The numbers come from pilot run 2 (152 TCGA series, 3,220 anchors). The TCGA pilot is used only to show how the method behaves, never to judge accuracy. No code was changed for this review.

**Verdict.** The structural problems of v6/G0 are largely fixed:

- candidates stay open until the image is consulted;
- priors are fixed and inference is one pass;
- host and extent are chosen jointly;
- policy is separate from inference;
- constants have provenance.

Host attribution now has a learned, expert-validated compatibility model, and that is the part that visibly works. Three weaknesses remain.

1. **Existence is decided by hand-set priors, not by the image.** Image evidence is too weak to move them.
2. **Several constants that drive decisions are design choices.** The paper calls the result a posterior, but those constants are not calibrated, so it is not one.
3. **Expert validation does not cover the agents, the cancers or the lesion types the method is applied to.**

---

## 1. Existence: the image barely acts; agreement decides

### 1.1 Decisions follow agent agreement almost exactly

In run 2, decisions line up with how many agents claimed the anchor:

| Decision | Anchors | Two-agent anchors | Single-agent anchors |
|---|---|---|---|
| Accepted | 226 | 225 | 1 |
| Rejected | 1,732 | 16 | 1,619 |

The image barely moves these outcomes:

- **Λ is small.** The median Λ^ex is 1.0 and the 95th percentile is 1.64. For 72% of anchors, Λ^ex lies within a factor of 2 of 1.
- **Accepted anchors rarely have strong evidence.** Only 12% have Λ^ex ≥ 2.
- **Many accepted anchors had no grounding at all.** 49 were accepted on degraded scans with grounding switched off.

Changing one constant, π_two from 0.9 to 0.75, cut acceptances from 475 to 226. Together these show that p_a is essentially π(a,h), the support level, with small corrections. The paper's central claim, that image grounding "weighs, for every anchor, whether it is tumor" (Fig. 1 caption, §2.1), is not what happens.

### 1.2 The posterior claim is not supported

- **Calibrated part:** Λ^ex is estimated on expert tumours against expert normals. That half is calibrated.
- **Uncalibrated part:** the prior π is not the base rate of anything measured. π_two = 0.75, π_single = 0.5 and ρ = 1/3 are design constants. The text admits that agents cannot be calibrated on expert data.
- **Consequence:** p_a is not a posterior probability. The sentence "Eq. (case_posterior) is a posterior, not a score" (§2.1) is an overclaim, and the thresholds 0.75 and 0.25 have no probabilistic meaning.

### 1.3 π_two sits on the acceptance threshold, and the stated reason is wrong

- **The text** (§2.3, existence prior) says π_two = 0.75 "sits at the acceptance threshold, so agreement of two agents alone does not accept an anchor."
- **What happens:** with a single host and Λ = 1, p_a = 0.75 ≥ 0.75, so the anchor is accepted. Any Λ slightly above 1 also accepts.
- **Fragility:** in run 2, 125 anchors have p_a in [0.74, 0.76], so the decision is a coin-flip on floating-point detail.
- **Fix:** move π_two off the threshold, or use a strict inequality. More importantly, give π_two a source (Section 4).

### 1.4 Evidence is weak by construction

- **The tree is coarse.** It has three outcomes (tumour-like, organ-like, uncertain), and 73% of anchors with a computed outcome are "uncertain".
- **The corrections are asymmetric and can only weaken evidence:**
  - Λ^ex is clipped to [1/100, 100];
  - below 3 ml, evidence against is capped at 1/3;
  - evidence for is capped at 1 outside the host;
  - frequencies are mixed with 10% of the pooled table;
  - log ratios are shrunk by n/(n+10).
- **No ablation.** None of these corrections is ablated, so the reader cannot tell whether the weak evidence is real or produced by the patches.

### 1.5 Unsegmented hosts get prior-only decisions

- **What the text says:** an unsegmented host has no normal tissue, so the context sign is skipped.
- **Why that matters:** context is the only sign that relates the anchor to its host. On a KB-region host, existence is decided by internal structure and elongation alone. The uncertain outcome dominates, so the decision collapses to the prior.
- **Who it affects:** this is the usual situation for the ovary, which has no TotalSegmentator class and is often missed by VoxTell. It also applies to other small organs, so OV, one of the main cancers, is decided almost entirely by agreement.

### 1.6 The normal likelihood depends on a designed mix

- **What p(E|N,h) is:** a frequency over a designer-chosen set of normal structures: salient intra-organ structures, tube segments, edges, parts, adjacent organs, fluid and whole organs.
- **Why that matters:** the frequency is only a likelihood relative to the distribution of things the agents actually mistake for tumour, which is never estimated.
- **The tumour side has the same issue:** p(E|T,h) includes partial cores (30–70%) and three-slice cuts in chosen proportions.
- **Status:** this was point 4.5 of the previous review and is still unaddressed. The mixing proportions should be stated, and the effect of changing them should be reported.

### 1.7 Transfer to agent candidates is asserted, not shown

- **The claim:** §2.1 says the signs "were found to transfer from expert data to unseen agent candidates".
- **Correction (19 Oct, user):** rated TCGA candidates were never used to select signs.
- **Fix:** state how transfer was established on expert data, for example held-out collections and simulated candidates, or rephrase as "properties that the agents do not model".

### 1.8 Independence assumptions are hidden

Three assumptions are implicit in the inference.

- **Agreement and image evidence:** p_a treats agreement (inside π) and image signs (inside Λ) as conditionally independent given z. Agents select locally abnormal regions, and closedness correlates with local contrast, so the two are not independent.
- **Anchors within a lesion:** existence is computed separately for each anchor even though they share a host.
- **The scan-level p_prim:** it assumes anchors are independent, which overstates the chance of having found the primary when one lesion has many anchors.

---

## 2. Host reasoning

### 2.1 w_h · g_h is not a posterior

- **What g_h is:** a logistic score fitted to "true host versus neighbours" on the testbed. It estimates P(true host | φ) under the testbed's case mix. It is not a likelihood p(φ | h).
- **Why the product is wrong:** multiplying it by w_h (1 / 0.3 / 0.15) double-counts base rates and mixes a discriminative score with a prior.
- **Fix:** a conditional-logit (choice) model is the principled form: P(h|L) = exp(β·φ_h + log w_h) / Σ_h' exp(β·φ_h' + log w_h'). The prior then enters as a fixed offset and β is fitted with that offset in place. g_∅ would become one explicit no-host alternative with its own fitted intercept.

### 2.2 The host weights are hand-set and untestable on the testbed

- **The weights are design choices.** 1 / 0.3 / 0.15 are knowledge-base preferences.
- **The testbed contains only primary tumours of their own organ.** It has no metastases and no invasion across organs, so it cannot test whether local-invasion and distant hosts are attributed correctly.
- **This is the decisive case in practice.** Ovary versus colon in X42 is exactly this situation.

### 2.3 The weakest testbed scenario is the commonest real case

- **Testbed result:** host top-1 is 0.363 when the true host is unsegmented.
- **Real use:** that is the ovary case on most OV scans.
- **The conflict:** β_KB = −5.45 tells inference to distrust KB regions. The primary-not-detected override then forces the primary back in. In X42, inference chose colon and the override reassigned the lesion to the ovary.
- **Consequence:** the paper cannot claim both that unsegmented hosts are weak hypotheses and that the primary is guaranteed. The relationship needs stating, and the override needs to report the inferred host next to the forced one.

### 2.4 The "possible contiguous extension" role cannot occur

- **The rule:** a contiguous extension is an accepted anchor whose host is at risk of invasion and which lies within 10 mm of an accepted primary-host tumour.
- **Why it never fires:** lesions are formed by grouping anchors within τ = 10 mm, and every anchor of a lesion shares one host. Any anchor within 10 mm of the primary tumour is therefore in the same lesion and gets the primary host.
- **Consequence:** primary-plus-invasion is forced into a single host. That is biologically the main multi-host situation (OV into colon, BLCA into prostate or uterus, KIRC into adrenal or liver).
- **Fix:** either a host variable per anchor coupled within a lesion, or a lesion-level host *set* with a contiguity term.

### 2.5 The no-host baseline is fragile

- **The definition:** g_min is the compatibility of the least compatible expert tumour with its true host. That is an extreme value, so it depends on testbed size and moved from 0.036 to 0.207 after the refit.
- **The prior:** the no-host prior π(a,∅) = π_s ρ is a further design choice.
- **Fix:** use a quantile, or a fitted intercept (Section 2.1).

### 2.6 Paired organs

- **The text:** "the planner records the side of the primary tumor for paired organs" (§2.2).
- **The manifests:** both sides are listed as primary.
- **Fix:** say which is true. If both sides are primary, the weight of 1 is effectively doubled for paired-organ cancers compared with single-organ cancers.

---

## 3. Extent

### 3.1 The extent is chosen by the prior alone

- **Why:** extent evidence was a NO-GO (ν = ∞), so ξ̂ = argmax of π̄ · ρ^f · ψ.
- **What that means:** extent selection is a learned size prior plus flags, with no image input. "Delineation is a choice among the agents' proposals" is accurate. Implying that the image makes that choice is not.
- **Fix:** keep Eq. (extent_evidence) only as an ablation, or move it to the supplement. A reviewer will ask why an unused term is in the main method.

### 3.2 ψ reflects how the testbed outlines were simulated

- **What ψ is:** a density ratio over the simulated outline families (part80, exact, dil4, host10, nb15, organ, lobe), whose proportions were designed.
- **Why it matters:** the chosen-extent distribution on the testbed (part80 175, exact 127, dil4 38) mirrors that design.
- **Fix:** report how sensitive ψ is to the proportions of the outline families.

### 3.3 Inconsistent support levels

- **The mismatch:** π̄ uses 0.9 and 0.5 for voxel support, while π_two is now 0.75.
- **Fix:** either tie them together, or explain why voxel support and anchor support differ.

---

## 4. Hand-set constants (your main concern)

Table S1 says "sensitivity reported", but no sensitivity has been done yet. The constants fall into three groups by how much they affect results:

| Group | Constants | Effect | What is needed |
|---|---|---|---|
| **A. Decide outcomes directly** | π_two 0.75, π_single 0.5, ρ 1/3, w_h 1/0.3/0.15, accept/reject 0.75/0.25, g_∅ rule, π(a,∅) | Run 2 shows A drives most decisions | Calibration from an external source, or report results as curves over A |
| **B. Shape candidates** | τ 10 mm, envelope 10 mm, lobe opening 3 mm / 0.5 ml, claim minimum 0.5/0.2 ml, prompt mismatch 50%, flatness 0.25/40%, organ-overlap 0.6/0.6/0.8, leak 50%, inside/straddle 0.9/0.1, μ rules (2v_max, 0.5v_min, 10 mm) | Moderate | One-at-a-time sensitivity on the testbed |
| **C. Regularise evidence** | ζ 1.5, 30 mm closedness window, 1/3 small cap, clip 100, 10% mixing, n/(n+10), stratum ≥ 30, size bands 3/20 ml, 5 mm grid | Probably small, but untested | Ablation (each removed in turn) |

Options for group A, from most to least defensible:

> **Update 19 Oct (user):** no further annotated cohort is available; only R2Seg and TotalSegmentator can be used, and the agents are not run on R2Seg, so option 1 is not possible. The 152 pilot series were inspected only for bugs (for example the big green box) and are not a development set. The follow-up plan is `claude/hyperparameter_plan_2026-10-19.md`: replace priors and thresholds with error-rate rules on expert data.

1. **(Not possible, see update) Calibrate on an external public annotated dataset where the agents can be run.**
   - **Data:** not R2Seg (excluded by our rule), and checked for overlap with the agents' training data.
   - **What it gives:** the real frequency of tumour among two-agent and single-agent anchors, and the odds ratio for each warning signal. That turns π and ρ into estimates and makes p_a a posterior.
   - **Status:** this is the only route that fully answers the reviewer, and it stays within the expert-annotation and public-data constraints.
2. **Reduce and tie the free constants.**
   - Use one odds factor for all signals (already done).
   - Express the thresholds as a stated cost ratio, accepting when p ≥ c/(1+c).
   - Set π_single to the uninformative 0.5.
   - Make π_two the single free agreement constant.
3. **Report outcomes as functions of the constants.**
   - Vary π_two ∈ {0.6, 0.75, 0.9}, ρ ∈ {1/2, 1/3, 1/5} and the host weights.
   - Report accepted, rejected and uncertain counts and the testbed metrics for each setting.
   - Use rank-based summaries (for example, the ranking of anchors by p_a) that do not depend on the exact threshold.
4. **Separate development from evaluation in TCGA.**
   - The 152 pilot series were used for visual tuning (§2.10 of the previous review; three changes in this round).
   - Declare them a development set, exclude them from every reported TCGA number, and freeze before the full run.

**The paper contradicts itself here.**

- **§3.1 (Data)** says "no other rule or constant was changed on TCGA" beyond the anatomy-check softenings.
- **§3.2 (Implementation)** now lists three further changes.
- **The Table S1 caption** says "No constant is estimated on TCGA."

All three statements need to agree, and π_two's row now says it was "lowered after the pilot visual check".

---

## 5. Validation gaps

1. **The testbed does not cover the TCGA cancers.**
   - The host and extent testbed covers kidney, liver, pancreas, lung and cervix (and prostate per the text).
   - OV, UCEC, BLCA, BRCA, COAD and STAD hosts are untested, although their expert tumours exist in R2Seg (bladder, breast, colon, uterus).
   - Ovary has no expert data at all (OV04 is excluded) and no expert normal anatomy.
2. **The candidates are simulated, not agent outputs.** Existence validation compares expert tumour cores with designer-chosen normal candidates. It cannot estimate the precision of agent candidates.
3. **The yardsticks measure consistency, not correctness.**
   - Spread, cross and disagree reward stable outputs.
   - Miss penalises abstention, so it interacts with the 36% of anchors left uncertain.
   - A method that confidently picks the same wrong lesion scores well.
4. **The blind ratings are thin.** One reader, no radiologist stated, and no inter-rater agreement.
5. **Ablations and sensitivity are all TBD.** Several ablations listed in §3.6 test variants that no longer matter, such as iteration. The group A sensitivity is missing.
6. **Override and uncertain outcomes are not evaluated.** In run 2 there were 110 overrides in 77 series, and 36% of anchors were uncertain. What a user gets on such a scan needs reporting, for example how many series have no segmented primary at all.

---

## 6. Consistency between text and implementation

| Topic | Text | Code / data |
|---|---|---|
| Override choice | "primary-host lesion with the largest p" | largest p × P(primary \| tumour) |
| Override labels | not described | host reassigned but host class not updated (X42 bug); whole lesion accepted, including uncertain anchors |
| p_prim | "reported, not imposed" | the override imposes it |
| π_two at threshold | "agreement alone does not accept" | it does (p = 0.75 ≥ 0.75) |
| Extent voxel support | 0.9 / 0.5 | 0.9 / 0.5 (inconsistent with π_two 0.75) |
| Testbed organs | includes prostate | host refit used Kidney, Liver/Pancreas, Lung_01, Cervix |
| Paired-organ side | planner records the primary side | both sides primary in the manifest |

---

## 7. Presentation

- **Placeholders:** the abstract and Introduction are still placeholder bullets.
- **Outdated figures:** Fig. 1 still describes v6/G0 ("[figure to be updated]"), and the qualitative figure refers to G0. The comment at lines 32–34 still cites ledger v6 and G0.
- **Overloaded notation:**
  - π for the existence prior, π̄ for voxel support and π_two/π_single for support levels;
  - ρ for the odds factor and for Spearman's ρ;
  - w_h for host weights and w^0 for the extent prior.
- **The limitations list understates the main issue.** It should say plainly that existence is dominated by agent agreement in the current configuration. A reviewer will find it from Table S1 and the pilot regardless.

---

## 8. Priorities

1. **Decide how the group A constants are justified:** external calibration (option 1) or tied constants with sensitivity curves (options 2–3). Rewrite "posterior" to "score" until then.
2. **Fix the text and code inconsistencies** in Section 6, including the X42 override label and the π_two threshold argument.
3. **Declare the TCGA development set** and make §3.1, §3.2 and Table S1 consistent.
4. **Extend the host testbed:**
   - add bladder, breast, colon and uterus;
   - add synthetic invasion cases, where the tumour straddles the primary and a neighbour, to test local-invasion hosts and the lesion single-host assumption.
5. **Allow multiple hosts within a lesion** (Section 2.4), or drop the contiguous-extension role.
6. **Run the group B sensitivity and group C ablations** on the testbed.
7. **Decide what to do about existence:** either strengthen the image evidence (finer outcome space, fewer caps, measured mixing proportions), or reframe the paper. Under the reframing, existence would be presented as an agreement prior moderated by image evidence, with the contribution being host attribution and extent selection.
