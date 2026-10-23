# R2 v6 image grounding: normative ("what should be normal here") plan and outcome

Status: plan approved 12 Oct 2026; executed overnight. Detailed results: PanCIA project note claude/r2_normative_results_2026-10-12.md; outputs in clinical/PanCIA_outputs/ledger/v6/norm/.

## Plan (as approved)
- Organ-specific normal models (kidney, liver, pancreas, lung; CT first) from expert normal anatomy only (TotalSegmentator no_pathology CT; R2Seg tumour-free organs as a held-out check). One pan-cancer decision rule.
- Channels relative to the same organ: intensity, texture, local discontinuity, organ shape (occupancy atlas). OR logic (Sidak min-p).
- Classes: tumour-like p < 0.05, organ-like p > 0.5, otherwise uncertain.
- Success: GT tumours organ-like <= 5%, tumour-like >= 70% per organ, FA <= 7%; then TCGA.

## Outcome
| Question | Answer |
|---|---|
| Does the normative OR detect GT tumours in normal tissue? | Yes. GT abnormal: kidney 80%, liver 95%, lung 100%, pancreas 59%; quiet <= 7%; FA 2-4% on held-out normals (5-13% on other-domain normals) |
| Does it discriminate agent candidates (TCGA)? | No. Selection: agents return regions that already differ from parenchyma |
| Which evidence transfers in-domain (GT vs agent-like normal structures) and out-of-domain? | Region elongation everywhere; organ deformation for kidney CT; intensity does not |
| Abnormality gate + geometry rule (N5) | In-domain: GT organ-like 0-2% except lung 16%. TCGA CT: coverage 0.50, confident error 0.27. TCGA MR: 0.43, 0.39. Not meeting criteria |

## Next steps (for discussion)
1. The normal model must include normal sub-structures (vessels, fissures, renal sinus, partial-volume edges). A learned normative model on normal TS scans, conditioned on organ and location.
2. In-scan matching against the scan's own salient normal structures.
3. A local organ-contour deformation term for paired solid organs.
