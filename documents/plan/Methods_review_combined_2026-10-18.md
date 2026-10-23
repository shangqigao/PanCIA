# PanCIA Methods — combined review (18 Oct 2026)

This review covers the Methods of `documents/main.tex` (md5 5e4717…): host reasoning, tumour reasoning, image grounding and EM. It merges two reviews:

- **R1:** the external reviewer's comments.
- **R2:** Claude's review, written after the G0 pilot panel inspection.

Points raised by both are merged; the tag after each heading shows the source. Pilot numbers come from the G0 v2 pilot on 152 TCGA series (`ledger/g0_pilot/`, 17–18 Oct). No code was run for this review.

---

## 0. The overall verdict

Both reviewers agree on the core problem. Host and tumour reasoning are described as producing *priors*, but they make many irreversible decisions before any image evidence is seen. Those rules:

- fix the primary host;
- delete hosts whose organ is missing;
- hard-drop claims by shape or prompt;
- prune flagged outlines;
- order lesions;
- protect primaries.

These decisions determine which explanations remain possible, and often determine the final answer. The image layer then acts as a weak correction:

- **Admitted anchors:** 497 → 476.
- **Decisive evidence:** Λ̂ ≥ 3 was reached by only 2 anchors.
- **Weak evidence:** 58% of host-frame evidence has applicability below 0.5.

Where the image layer does act, in extent choice, it is ill-posed (Sections 4–5). The result is "an anatomical rule system with a probabilistic correction layer".

The anatomical structure itself is valuable and should stay: the knowledge base, the set-based units (claims, anchors, outlines, lesions), the cut organ references and expert-only likelihoods. What needs to change is the separation between three things: generating candidates, assessing uncertain evidence, and applying decision policy.

The same evidence is also reused many times: agreement, organ overlap and touch each feed host choice, existence priors, outline eligibility, lesion class, protections and grounding. Without an explicit dependence model, one observation counts as several corroborating ones. The reasoning masks are themselves derived from the image, so calling their consequences "priors" does not make them independent of the later image evidence. **[R1, R2]**

---

## 1. Host reasoning

**1.1 The definition contradicts the implementation.** [R2]
Host reasoning is defined as R(A_i, A_j), relations between anatomy masks. But the host prior P(h|a) is built from touch and the boundary share, which are tumour–organ relations, i.e. R(T, A). Either redefine the two terms or move P(h|a) under tumour reasoning.

**1.2 Touching the primary forces primary origin.** [R1, R2]
The rule a ∩ E_primary ≠ ∅ ⇒ P(primary|a) = 1 uses a 10 mm envelope, and the value is held fixed in EM. A single voxel within 10 mm therefore forces primary origin, even when most of the candidate lies in a neighbouring organ.

The same touch also decides lesion membership and primary protection, so one uncertain spatial observation becomes three decisions. Knowing the cancer type justifies "a primary tumour exists in this scan". It does not justify "every nearby candidate is the primary".

Many of the bad extents in the pilot panels are protected primary anchors (OV, UCEC, CESC, BLCA) sitting next to bowel or vessels.
*Change:* use touch only to generate host candidates. Score hosts on graded features (overlap fraction, boundary contact, relative position, anatomical uncertainty) and keep neighbouring hosts available. Model the existence of the primary at scan level, including an explicit "primary not detected" option.

**1.3 Missing anatomy is treated as impossible origin.** [R1]
A non-primary host with no feasible mask gets prior 0. Only the primary falls back to a knowledge-base region. But an organ can be missing for several reasons: coverage, agent failure, distortion, or replacement by tumour. This asymmetry means a large metastasis that has replaced its host loses its host hypothesis altogether.
*Change:* distinguish three cases.

- **Confidently outside the field of view:** exclude the host.
- **Expected but not segmented:** keep a low-confidence spatial hypothesis, such as a knowledge-base region.
- **Segmented unreliably:** also keep a low-confidence hypothesis.

**1.4 The organ reference hides disagreement.** [R1, R2]
Ĥ_o is the union of feasible masks, so it carries the leakage of every agent. If an organ agent includes the tumour in its organ mask, the "cut" freezes tumour into the normal reference. Removing the contested overlap also discards tissue that is informative about invasion and origin.

"Cut" is also overclaimed. Through q → N_h, tumour extents reshape the normal model on every iteration. Only the organ masks are fixed.
*Change:* represent three zones separately: confident organ interior (agents agree), uncertain boundary, and disputed territory. Fixed anatomy is fine; treating it as error-free is not. State precisely what is cut.

**1.5 Normal-anatomy checks can reject pathological anatomy.** [R1, R2]
Several checks are asymmetric or destructive:

- An oversized primary is protected, but an oversized secondary host is not.
- A part found outside its whole is clipped, even though the whole may be the wrong mask.
- Organs that should abut can be pushed apart by a mass.
- Feasibility is all-or-nothing per mask: a mask leaking by 40% but under 2·v_max passes whole.

Expert-data validation (0.11% CT / 0.41% MR infeasible) measures specificity only, not sensitivity to realistic errors.
*Change:* reserve hard exclusion for verified geometric impossibility or corrupted output. Otherwise lower the confidence in the mask. Measure sensitivity with injected leaks and merges.

**1.6 Boundary-share attribution is biased, and its weights are not probabilities.** [R1, R2]
The share s_h(a) of the 3 mm boundary band favours a large surrounding organ over the true organ of origin. It also penalises exophytic tumours, whose boundary lies mostly outside their host.

The role weights w_h (1 / 0.3 / 0.15) are preference scores, not calibrated probabilities. Touch is binary: one voxel inside 10 mm makes a candidate.
*Change:* call these anatomical compatibility scores unless they are calibrated. Evaluate host attribution separately on exophytic, invasive and boundary-straddling expert lesions.

---

## 2. Tumour reasoning

**2.1 Agreement is over-trusted.** [R1, R2]
Counting agents rather than prompts is a good safeguard, but BiomedParse and VoxTell are not independent witnesses. They share errors driven by anatomy, artefacts and semantic ambiguity. VoxTell also supplies both tumour and organ masks, so some "anatomical confirmation" comes from the agent that proposed the tumour.

The voxel prior of 0.9 for agreement (and 0.75 and 0.5 elsewhere) is unvalidated, so π̄ and p_a are not probabilities. That is why the posterior sat on a knife-edge at 0.5, and the "decisive evidence" rule is a patch for it. Disagreement may reflect complementary sensitivity rather than unreliability.
*Change:* treat agreement as a feature with provenance, and flag self-confirmation. Calibrate any prior on expert or held-out data, or call it a score.

**2.2 Silence is not support.** [R1]
The 0.75 "silent other agent" prior near the primary turns another agent's miss into positive evidence for a candidate. The fact that a primary exists somewhere does not make *this* candidate likely to be it, so a false positive gains support simply because the other agent sees nothing.
*Change:* silence should only reduce the penalty for missing confirmation. Model the existence of the primary at scan level.

**2.3 Anchors hide unsupported extensions.** [R1]
A single connected claim that overlaps a two-agent core becomes an outline of that anchor. Its unsupported territory then borrows the core's credibility. A thin bridge to a distant false region inherits the core's existence. Conversely, a real separate lesion connected through erroneous tissue never gets its own assessment.
*Change:* separate shared cores, unsupported extensions and additional lobes. Existence of a core must not transfer to everything connected to it.

**2.4 Hard shape and prompt gates can remove real lesions.** [R1]
The slab rule (flat ⇒ drop) can remove thin lesions and surface-spreading disease, and can be triggered by partial coverage or thick slices. The prompt gate (< 50% inside the prompted host envelope ⇒ drop) can remove exophytic or invasive tumours, especially when the host mask is incomplete.
*Change:* make both soft penalties by default. Hard rejection needs more than shape or prompt mismatch.

**2.5 The leak and organ-overlap tests miss the commonest overgrowth.** [R2]
An outline counts as a leak only if more than half of its volume beyond the anchor lies in *non-host organs*. Growth into the host's own normal parenchyma, or into unlabelled fat or bowel, is never a leak, and that is exactly what the orange extents do.

The "is an organ" flag needs three conjunctive conditions, so partial-organ masks pass. E2's 1471 ml VT:liver outline was not flagged. VoxTell host-prompt outlines are often organ-shaped by construction.
*Change:* define overgrowth relative to the host's normal tissue. Treat outlines that are mostly organ (for example ≥ 50% inside one organ reference) as at least heavily penalised extents.

**2.6 Candidate pruning already decides disputed boundaries.** [R1; conflicts with the 17 Oct decision]
Since 17 Oct, flagged and leaking outlines are excluded from the candidate extents (the user's decision after the panel review). R1 points out that this means grounding can no longer rescue an apparent leak that is really invasion, or an organ-overlap flag caused by tumour replacement. That contradicts "flags are soft".

The pilot shows the tension from both sides:

- Keeping those outlines let leaks win by default (29 of 181 changes).
- Dropping them exposed anchors whose survival depended on a neutral leak (8 new demotions).

The real fault is the extent model (Section 4), not the candidate list.
*Change:* keep suspicious-but-plausible outlines with explicit penalties and exclude only demonstrably invalid ones. Otherwise state that the decision is made at pruning time, not by inference.

**2.7 Classification is order-dependent.** [R1, R2]
"The first unflagged primary lesion by class and size" becomes the reference, and local invasion is then judged against it. A wrong first pick distorts everything downstream. The fixed nodule criteria (10 mm, 3 slices) depend on slice thickness and can reject true lesions.
*Change:* score all lesions before committing to an index primary, and separate four decisions:

1. Is the lesion a tumour?
2. What is its host?
3. How does it relate to the known cancer?
4. Which lesion is selected for measurement?

**2.8 Existence and cancer attribution are conflated.** [R1]
An anchor that "lies in an unrelated organ" is soft-rejected as normal, but there are three possibilities: a normal structure, a tumour of the known cancer, or another tumour or indeterminate abnormality. An incomplete list of spread sites should not turn an abnormal lesion into normal tissue by definition.
*Change:* add an "other / indeterminate lesion" category, and separate "outside task scope" from "not tumour".

**2.9 Lesion grouping and spread labels are too deterministic.** [R1]
Chaining can merge distinct lesions through intermediate anchors. The grouping distance τ depends on slice spacing, so lesion identity can change between scans. Proximity does not establish invasion, and a host prompt plus agreement does not establish metastatic origin.
*Change:* separate geometric grouping from biological interpretation. Use labels such as "possible contiguous extension" and "secondary-site tumour candidate".

**2.10 Constants were adjusted on the TCGA pilot.** [R2]
Several rules changed after looking at TCGA pilot outputs, while the text says "fixed a priori, no constant tuned":

- small mask → flag;
- extra components → flag;
- side check only when reliable;
- large primary → flag;
- breast degrade rule;
- organ_replaced conditions.

Reviewers will treat this as tuning on the evaluation cohort.
*Change:* declare a development subset, freeze the rules, and evaluate on a disjoint TCGA set. Add a provenance table for every constant (expert data / knowledge base / design choice).

---

## 3. Shared weakness: repeated use of the same evidence

These two weaknesses run through every component above. [R1, R2]

- **Dependent reuse:** agreement, touch and organ overlap each enter host choice, existence priors, outline eligibility, lesion class, protections, and organ overlap again in grounding as S5.
- **Image-derived priors:** the masks are derived from the image, so their consequences are not independent of the image evidence used later.

*Change:* make each observation enter once, through an explicit model, or document the dependence and test its effect by ablation.

---

## 4. Image evidence and likelihoods

**4.1 Extents are compared on different data.** [R1, R2] This is the main cause of the bad orange extents.
E_a(ξ, h) is computed on each extent itself, so p(E(ξ₁)|T) and p(E(ξ₂)|T) are probabilities of different events. Ranking extents by w·Λ answers "which region, as a whole, looks most like an expert tumour", not "which boundary is right". A large region containing a tumour plus normal tissue can still be "closed" or "patchy".
*Change:* score every extent on a common observation (for example U_a plus a margin). Voxels inside ξ follow the tumour model and the rest follow the host's normal model, so growth into normal parenchyma costs evidence.

**4.2 The tumour table rewards overgrowth by construction.** [R2]
p(E|T) includes partial, dilated, three-slice and leaking variants of expert tumours, all labelled as tumour. That is reasonable for the existence question (z). It is wrong for the extent question (ξ), where those variants are exactly the alternatives to reject.
*Change:* use separate evidence models for existence and extent. Learn an extent model, including a size or parsimony prior, from expert masks against their variants.

**4.3 There is no parsimony in the extent prior.** [R2]
w0 ∝ π̄(ξ). For single-agent anchors every extent ties at 0.5. For two-agent anchors, a 700× union is penalised only by about 2×.
*Change:* add a learned penalty on the volume ratio to the anchor.

**4.4 Evidence is coarse and heavily patched.** [R2]
There are 6 evidence states and Λ ≤ 6.9, so many extents tie. On top of that sit many corrections:

- clipping;
- the 1/3 floor below 3 ml;
- a cap at 1 outside the host;
- shrinkage by n/(n + 10);
- 10% pooled mixing;
- stratum fallback.

The result is a designed score, not a likelihood, and none of these corrections is ablated.

**4.5 The normal table depends on how normals were sampled.** [R2]
p(E|N, h) is a frequency over a designer-chosen mix of normal structure types. It is only a likelihood relative to the distribution of what agents actually mistake for tumour, and that distribution is not estimated.

**4.6 Harmonising to 5 mm erases small lesions.** [R2]
D19, a subtle 1.6 ml hypodense liver lesion, became "organ-like" and was demoted. The cost of harmonisation for lesions under 3 ml is not reported.

---

## 5. EM and inference

**5.1 The E-step uses a different model from the stated posterior.** [R1, R2]
Eq. case_posterior evaluates z = 0 on the anchor under p(E|N, h), and z = 1 on each ξ under p(E|T). The E-step text uses r₁ ∝ P·π·w·Λ and r₀ ∝ P·(1 − π). The normal likelihood varies with host and extent, so it cannot be cancelled as a constant, and the two branches evaluate different regions.
*Change:* adopt one coherent model with explicit likelihoods of the same observation under every hypothesis. Otherwise present the likelihood-ratio form as a conditional scoring model and drop the claim of equivalence.

**5.2 There is no objective, so this is not EM.** [R1, R2]
The iteration mixes responsibilities, per-anchor weight updates, volume-weighted host pooling, a thresholded normal reference, recomputed features, damping and hard protections. No single objective yields all of these. The boxed equation p(T, N) = p(T|N) p(N) is just the chain rule, and damping does not establish convergence.

Also, ρ(ξ) is normalised within z = 1, so it drops the existence responsibility. An almost-certainly false candidate updates its extent weights as strongly as a credible tumour.
*Change:* derive the objective, or rename the procedure "damped alternating refinement". Weight extent updates by p(z = 1).

**5.3 Feeding the posterior back as the prior double-counts evidence.** [R1, R2]
With fixed evidence, ρ₁/ρ₂ = (w₁/w₂)(Λ₁/Λ₂), and feeding ρ back into w applies Λ again on every iteration. The pseudo-count α limits this but does not remove it, so the final weights are not posteriors under the original prior. Estimating per-anchor mixture weights from a single case is not justified.
*Change:* keep the anatomical, existence and extent priors fixed. Let posteriors change only because the evidence or reference changes.

**5.4 The normal reference can confirm itself.** [R1, R2]
With q < 0.1 ⇒ normal reference, the loop can run in either direction:

- A real tumour with a low prior enters the reference, shifts it towards its own appearance, looks less abnormal, and loses more probability.
- A false positive stays excluded, so the reference never challenges it.

Large outlines also delete their host from its own reference at t = 0, giving unknown evidence and Λ = 1. The hard threshold makes the evidence jump discontinuously: the LIHC recovery fell from Λ 6.9 to 1.46 when other anchors' outlines changed. "Cannot drift into normal tissue" is too strong; the iteration can select a leaky candidate or progressively misclassify proposed tissue.
*Change:* start from a frozen, conservative normal reference that excludes all candidates. Test adaptive references separately, with soft weighting and a stable component.

**5.5 The reported host and extent are not the joint posterior.** [R1]
The host comes from the updated P(h|a) and the extent from w(ξ); both are weights, not responsibilities. γ_a(h) also includes the z = 0 branch, whereas an accepted tumour's host should come from P(h | z = 1, evidence). Choosing host and extent separately can produce an incompatible pair.
*Change:* (ĥ, ξ̂) = argmax over (h, ξ) of r_a(h, 1, ξ), and report marginal uncertainties conditioned on z = 1.

**5.6 Lesion host sharing is only approximated.** [R1]
Volume-weighted pooling of responsibilities encourages agreement but does not enforce a shared host. Lesion groups are themselves host-dependent, and a large erroneous anchor dominates the pool.
*Change:* add an explicit lesion-level host variable, or call the pooling a consistency regulariser.

**5.7 Convergence and q are loosely defined.** [R1]
The stopping rule ignores p_a, the normal reference and the evidence states. q(v) = max_a P(v ∈ a) is an occupancy score, not the probability that any tumour occupies v.

---

## 6. Decision policy and claims

**6.1 Policy is mixed with inference.** [R1, R2]
The decisive-evidence rule and the "largest primary never demoted" rule impose outcomes by policy. Knowing that a primary exists does not prove that the largest admitted candidate is that primary. Combined with Λ̂ ≥ 3 occurring for only 2 anchors, the image term is nearly inert, yet the Introduction and Fig. 1 claim grounding "infers, for every anchor, whether it is tumour".
*Change:* report safeguards and overrides separately from the probabilistic output, with override flags. Align the claims with what the pilot shows.

**6.2 Validation lags behind the claims.** [R2]
The R2Seg check of extent choice with simulated outlines was planned but never run. It would have exposed the extent problems on expert data.

---

## 7. Presentation

- Notation is overloaded: E_h (envelope) vs E (evidence); A (anatomy) vs A_{k,o} (masks); P(h|a) used for both prior and update; q, π, π̄, w, w0, ρ and γ have overlapping meanings. [R2]
- About 60 constants appear in prose with no provenance table. [R2]
- "Conditional probabilistic" is used loosely: N is a deterministic reference plus a threshold, and p(N|·) is never computed. [R2]

---

## 8. Recommended restructuring

### Three layers [R1]

1. **Candidate generation.** Keep plausible hosts, lesions and extents, and prune conservatively. Hard exclusions only for verified impossibility or corrupted output. Touch generates host candidates, missing organs keep low-confidence hypotheses, and slab and prompt mismatches become penalties.
2. **Evidence assessment.** Record graded anatomical compatibility, model support with provenance (agreement as a feature, flagged self-confirmation), and image evidence computed on a common observation per anchor with explicit likelihoods for host, existence and extent. Record uncertainty, and count each observation once.
3. **Decision policy.** Apply explicit acceptance thresholds and clinical safeguards, such as scan-level primary existence and an index-lesion choice, after the probability estimates, and flag every override.

### Simplify inference before elaborating it [R1, R2]

1. Fix all priors: no posterior-to-prior feedback.
2. Use a frozen, conservative normal reference.
3. Compute one explicitly defined conditional distribution over (h, z, ξ) on a common observation.
4. Choose host and extent jointly, conditioned on z = 1.
5. Apply safeguards afterwards, with override flags.
6. Add adaptive references or iteration only if they beat this baseline on held-out data.

Compare three variants (fixed reference, reference-only iteration, current full iteration) on calibration, false rejection, extent quality and sensitivity to initialisation.

### Priority order

1. Remove the forced primary host (1.2) and model primary existence at scan level (2.2).
2. Rebuild extent inference on a common observation, with a parsimony prior and a separate extent model rather than the augmented tumour table (4.1–4.3, 5.1). This fixes the orange extents.
3. Fix the priors and freeze the normal reference (5.3, 5.4). Select (h, ξ) jointly (5.5).
4. Missing anatomy keeps low-confidence hosts (1.3). Slab and prompt gates become soft (2.4).
5. Separate existence from attribution, adding an "other / indeterminate" category (2.8). Score lesions before choosing the index primary (2.7).
6. Write a provenance table for the constants, and declare a development cohort vs a frozen evaluation cohort (2.10, 7).
7. Run the R2Seg checks of extent and host choice before any further TCGA runs (6.2).

**Bottom line.** Joint reasoning over existence, host and extent is a strong idea, and the anatomical scaffolding should stay. For it to resolve disagreement, the earlier rules must leave credible alternatives open. The inference also has to compare hypotheses on the same data, with fixed priors. Until then, it should not be presented as calibrated Bayesian EM.
