# PanCIA v8 plan: agreed changes (19 Oct 2026)

This plan supersedes Section 4 of `claude/methods_review_v7_2026-10-18.md` and the earlier draft of this document. The user agreed these changes on 19 Oct.

`documents/main.tex` has been updated to match (md5 3d98e2; backup `documents/bak/main_before_v7_alpha.tex`). The code has not been changed yet.

**Principle.** Every number is either estimated on R2Seg or TotalSegmentator, or is a stated design choice: the error level α, ordinal assumptions, or geometry. No numeric prior is attached to agent agreement or to the cancer type.

**Framing.** We keep p(T,N|c,X) = p(T|N) p(N) and Fig. 2.
- N (the host and its normal tissue) is inferred probabilistically from anatomy.
- T is tested against expert likelihoods given N.
- The tumour output is not called a posterior.

## Host (normal model)

1. **Equal host weights.** The weights 1 / 0.3 / 0.15 are removed. H_C decides only which organs are candidates. The roles primary, local invasion and distant are attached after inference. The planner keeps its role ranking only to order prompts under the budget.
2. **No special rule for omentum or space structures.** In run 2 with equal weights, 89 lesions moved to omentum: 57 from an unsegmented ovary or uterus, and about two thirds with a margin below 0.1. These cases are ambiguous, and are reported as such.
3. **Host credible set.** The host is reported as the smallest set of candidates with probability ≥ 1 − α. This mirrors practice, where difficult sites are settled by pathology.
4. **Choice-model fit.** Compatibility is refitted as a choice model (softmax over each case's candidates, conditional likelihood) on the testbed, replacing the separate logistic score and weights.
5. **No-host score.** Fixed at the α-quantile of the true-host scores of expert tumours.
6. **Override.**
   - Fix the X42 host-class label bug.
   - Report the inferred host next to the imposed host.
   - Override lesion: the one whose credible set contains the primary, with the highest support level, then the largest volume.
   - Override only that lesion's non-rejected anchors.
7. **Drop the "possible contiguous extension" role.** Anchors within τ share one host. This is listed as a limitation.

## Tumour (existence and extent)

8. **Remove** π_two, π_single, ρ and the 0.75 / 0.25 thresholds, together with p_a and p_prim.
9. **Ordinal support level ℓ(h).**
   - 2 for a two-agent anchor.
   - 1 for a single-agent or lobe anchor.
   - Minus 1 for each warning signal under h, with a minimum of 0.
10. **Existence test at α**, per stratum, from the expert tables.
    - **Rules out:** p(o|T) ≤ α and Λ < 1.
    - **Confirms:** p(o|N) ≤ α and Λ > 1. A confirmation counts only for an anchor inside its host.
    - **Neutral:** everything else, including degraded scans.
    - Use α = 0.05 by default; report 0.01 and 0.10.
    - The small-lesion cap (1/3) and clipping are removed. The smoothing constants are chosen by held-out likelihood.
11. **Decision table.** A conflict between support and image gives uncertain.

    | ℓ | Confirms | Neutral | Rules out |
    |---|---|---|---|
    | 2 | accept | accept | uncertain |
    | 1 | accept | uncertain | reject |
    | 0 | uncertain | reject | reject |

12. **Several hosts.** The decision must agree under every host in the credible set; otherwise the anchor is uncertain.
13. **Extent prior w0 ∝ ψ · ∏ η_f.**
    - The voxel support π̄ (0.9 / 0.5) is dropped.
    - The flag factors η_f are learned as density ratios jointly with ψ, holding out one collection at a time.
    - Extent evidence is not used; it appears only as an ablation.

## Run-2 preview (recomputed from saved outputs; approximate strata)

| Scheme | Accepted | Uncertain | Rejected | Series with an accepted tumour |
|---|---|---|---|---|
| Image only, equal treatment | 12 | 2,974 | 234 | 11 |
| Ordinal levels + test | 368 | 701 | 2,151 | 120 |
| Run 2 (numeric priors) | 226 | 1,152 | 1,732 | 100 |

The ordinal scheme agrees with run 2 on 202 of run 2's 226 accepted anchors.

## Implementation steps

1. `grounding_g1.py` → `grounding_g2.py`:
   - existence test tables, with rule-out and confirm sets per stratum for each α;
   - decision table;
   - host credible set;
   - choice-model host posterior.
2. `tb_fit.py`:
   - choice-model fit of β, the no-host α-quantile and credible-set coverage;
   - joint fit of ψ and η_f;
   - held-out false rule-out and false confirmation rates for each α.
3. `ledger_v7.py`:
   - support level for each host;
   - drop the 0.15 weight for organs outside H_C;
   - drop π̄.
4. `g1_run.py` → `g2_run.py`:
   - new outputs: host set, ℓ, outcome, test result and Λ for each host in the set;
   - override fixes.
5. **Testbed first:** refit and report the held-out results.
6. **Then the TCGA pilot (bug check only)**, with panels showing the host credible set.
7. **Fill main.tex:** the refitted β and ψ values in Table S1, the testbed numbers, and the sensitivity analysis.

## Sensitivity and ablation

**Sensitivity:** α ∈ {0.01, 0.05, 0.10}; τ and the 10 mm envelope; QC thresholds.

**Ablations:**
- the numeric priors (π_two 0.75 or 0.9, ρ 1/3, thresholds);
- the KB host weights instead of equal weights;
- each evidence component removed in turn;
- no ψ or η_f;
- extent evidence added;
- flagged outlines removed instead of penalised;
- binary touch instead of compatibility;
- the conflict rule: a ruled-out ℓ = 2 anchor rejected instead of uncertain.

**Supplement:** one-at-a-time checks of the lobe, claim, flatness, leak and overlap thresholds.

## Open

- Extend the host testbed with bladder, breast, colon and uterus from R2Seg, if they have organ labels, plus synthetic invasion cases.
- The stratum lookup in the run-2 preview is approximate (small = < 3 ml; CT); the real implementation uses the full stratum key.
