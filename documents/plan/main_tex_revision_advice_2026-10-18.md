# main.tex revision advice (18 Oct 2026, v2)

This advice is based on `Methods_review_combined_2026-10-18.md` (R1 and R2).

The framework is unchanged: **pan-cancer multi-agent tumour segmentation via conditional probabilistic anatomical reasoning**, with p(T, N | c_a, X) = p(T | N, c_a, X) p(N | c_a, X) as its core.

Figure 2 (PGM) is **kept as the overview**:
- tumour and host require reasoning;
- both are grounded to the image;
- they are linked by joint inference.

Every revision below is a refinement *within* the existing nodes. Nothing is added to the overview.

TCGA has no tumour annotations and serves for visual checks and rating only. Expert data (R2Seg, TotalSegmentator) is the only source of tables and learned priors, and the only quantitative validation.

---

## 1. Figures

**Fig. 2 (PGM): keep as is.**
- Replace the caption placeholder "[to be redrawn …]" with a caption that maps the overview onto the method.
- **Host node:** host reasoning gives the anatomy; tumour–organ relations give the host prior.
- **Tumour node:** tumour reasoning gives existence and the extent candidates.
- **Grounding:** image grounding supplies the evidence for both.
- **Joint inference:** combines them conditionally (N first, then T given N).

**Fig. 1 (pipeline):** update only the caption wording ("infers…" → see §10).

## 2. Conditional formulation (§2.1, Eqs. conditional_joint and case_posterior)

The boxed factorisation stays; its two factors are made concrete and computed on the same data.

- **N = (h_a, 𝒩_h):** the host and its frozen normal-tissue reference.
- **T = (z_a, ξ_a):** existence and extent.
- **Attribution y_a:** for tumours only, y_a ∈ {cancer C, other/indeterminate} is part of T. It is not a new node.

Evidence per anchor, on regions that are the same under every hypothesis:
- **Existence evidence E^ex:** on the anchor core a.
- **Extent evidence Y_{R_a}:** on a common region R_a, the union of the candidate extents plus a margin.

Posterior:
- p(h, z=1, ξ | ·) ∝ P(h|a) π(a,h) w⁰(ξ) p(E^ex | T, h) p(Y_{R_a} | ξ, h)
- p(h, z=0 | ·) ∝ P(h|a) (1 − π(a,h)) p(E^ex | N, h) p(Y_{R_a} | N, h)

With this form, the likelihood-ratio version is exact. It removes the inconsistency with the E-step text (l. 529–531).

## 3. Anatomy (§2.1 Eq. organ_reference; §2.3 Host reasoning)

**Three zones** replace the union Ĥ_o:
- confident core (feasible masks agree);
- uncertain boundary (union minus agreement);
- contested territory C_{o,o'}.

**Organ states:**
- **Present.**
- **Expected but unsegmented:** keeps a low-confidence knowledge-base region, for every host.
- **Out of view:** the only state that removes a host.

**State precisely what is cut:**
- Organ masks are never reshaped.
- The normal reference is computed once (core minus all agents' claims) and not updated.
- Delete "cannot drift into normal tissue".

**Unary and pairwise checks lower mask confidence.** Hard infeasibility is reserved for corrupt output or a geometric impossibility under a reliable header. This removes the primary/secondary asymmetry for oversized organs. A part outside its whole is not clipped.

## 4. Host and tumour reasoning definitions (§2.1)

Keep the user's definitions and make the text consistent with them.

- **Host reasoning, R(A_i, A_j):** gives the anatomy only (feasibility, organ states, zones, mask confidence).
- **Tumour reasoning, R(T_i, A_j):** gives everything else, including the host prior P(h|a), because touch and overlap are tumour–organ relations.

## 5. "Priors, not decisions" (l. 174–182)

Earlier steps generate candidates and graded evidence. They decide only what is impossible.

- **Hard rules (prior 0):** corrupt agent output, and hosts out of view.
- **Slab rule and prompt gate:** become soft penalties.
- **Remove "P(primary | a) = 1 when touching".** Replace it with a scan-level statement, "primary present among the candidates", whose posterior is reported. Touch only generates host candidates.
- **Silence of another agent:** removes the penalty for missing confirmation, but adds no support.

## 6. Tumour reasoning details (§2.3)

**Host prior → anatomical compatibility.**
- Graded features: overlap with the core and the territory, boundary contact, relative position (inside / straddling / exophytic), and mask confidence.
- Fitted on expert data, where the organ of origin is known.
- w_h is kept as a knowledge-base preference, labelled as a design prior and used once.

**Units.**
- Outlines are decomposed into: shared core, unsupported extensions, and lobes joined by a thin bridge.
- A lobe becomes its own anchor.

**Support.**
- Agreement is a declared support level, with provenance recorded (for example, VT's tumour mask confirmed only by VT's own organ mask).
- It cannot be calibrated on expert data because agents are not run there; sensitivity is reported instead.

**Lesions.**
- Grouping is geometric only: a fixed distance, and no chaining through single-agent bridges.
- Roles are descriptive labels ("possible contiguous extension", "secondary-site candidate").
- The order-dependent "first unflagged primary" pass is removed. All lesions are scored first, and the index lesion is chosen in the decision policy.
- An anchor in an organ outside the spread set gets attribution "other/indeterminate", not "normal".

**Extent candidates (Eq. extents).**
- Flagged and leaking outlines stay as candidates, penalised through w⁰ (reverses the 17 Oct exclusion; user decision).
- Only invalid outlines are excluded, for example the whole organ.

**Extent prior.**
- w⁰(ξ) gains a parsimony term, learned on expert tumours against their partial and overgrown variants.

## 7. Image grounding (§2.3)

**Existence term.**
- Keep the sign tree and the tables, computed on the core.
- Rebuild p(E|T) with partial variants only. Dilated and leaking variants belong to the extent question.
- Remove S5 from E, so organ overlap is used once (as an extent penalty).

**Extent term (new paragraph).**
- The extension voxels (ξ minus the core) are compared under two references: the core's appearance, and the host's frozen normal reference.
- Combined with the parsimony prior.
- An iso-intense extension is neutral, so the prior keeps the extent near the core.
- "Why choose, not redraw" stays.

**Resolution.** Small lesions on thin-slice scans use native-resolution tables (the D19 case).

**Patches.** Keep only those with a stated reason (clip, small-band floor, in-host condition, shrinkage), listed in a supplementary table.

## 8. Joint inference (§2.1 "Joint inference"; §2.3 "Expectation–maximization")

**One-pass conditional inference**, which reads the boxed equation literally:
1. **N:** frozen references and the host posterior, with the lesion host shared by enumeration over the lesion.
2. **T given N:** enumerate (z, ξ, y).

**No feedback.** Priors are fixed: there is no posterior-to-prior feedback and no damping.

**Joint choice.** (ĥ, ξ̂) = argmax r(h, z=1, ξ), with marginals conditioned on z = 1.

**Refinement as an ablation only.** An optional N-given-T update of the normal reference is an ablation. Drop "EM" unless an objective is derived.

## 9. Decision policy (replaces "Decision", l. 550–557)

- Acceptance is a threshold on the posterior (3:1 odds convention if calibration on expert data is not shown).
- Safeguards come after inference and are flagged, for example "primary not detected", optionally admitting the best primary candidate as a flagged override.
- Probabilistic outputs and overrides are reported separately.

## 10. Experiments and claims (§3, Introduction)

**Data roles.**
- **Expert data:** the source of tables and learned priors, and the testbed of simulated agent-like candidates with known truth. It measures host attribution (exophytic / invasive / intraparenchymal), existence calibration, extent choice and small lesions.
- **TCGA:** visual checks, rubric ratings and reporting only.

**Disclosure.** Some anatomy rules were softened after visual inspection of TCGA pilot outputs: small-mask and component flags, the side check, and breast degrade. Say so in one sentence.

**Supplementary constants table.** The source of each constant: expert data / knowledge base / design choice.

**Ablations.**
- one-pass vs refinement;
- with and without the parsimony prior;
- flagged outlines kept vs pruned;
- compatibility vs touch.

**Notation.**
- Envelope E_h → 𝓔_h.
- P⁰ is the prior, P the posterior.
- Stop reusing q, π and w.

**Claims.** Soften "infers for every anchor…" in the Introduction and the Fig. 1 caption until the expert testbed shows that the image term matters.

---

**Decisions needed from the user:**
- (i) Keep flagged and leaking outlines as penalised extent candidates?
- (ii) Are "other/indeterminate" lesions reported only, or also segmented?
