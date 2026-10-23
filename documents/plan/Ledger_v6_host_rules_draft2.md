# Ledger v6 host rules, draft 2 (13 Oct 2026; for discussion, not yet implemented)

Agreed with the user:
- **Protect the primary.** In these pre-surgical scans with confirmed cancer, a primary tumour exists unless its organ is out of view. Lesions near the primary organ are therefore primary first. Only after primaries are admitted are other claims considered as invasion or distant disease.
- **Contact tolerance is 10 mm** everywhere contact is tested: v3's own primary envelope; it covers TotalSegmentator's 3 mm grid plus a thick slice. Merging claims into lesions stays at v3's 5 mm.
- **Organ-likeness near the primary flags, never rejects.**

## Agent fragmentation check (read-only; 400 random series per agent)
| | VoxTell first prompt | BiomedParse |
|---|---|---|
| Empty | 83 / 387 | 2 / 378 |
| Components per series, median (q90) | 5 (32) | 15 (41) |
| Series with ≥ 1 piece < 0.5 ml | 76% | 86% |
| Volume in pieces < 0.5 ml, median (q90) | 0.1% (4.2%) | 0.7% (6.6%) |
| Mask made only of pieces < 0.5 ml | 7 / 304 | 2 / 376 |
| Total ml / largest piece ml, median | 50 / 40 | 107 / 38 |

Small pieces are common but carry almost no volume. BiomedParse's problem is different: many separate pieces of 0.5 ml or more spread across the scan. Its median total is 107 ml against 38 ml for its largest piece.

## Claim level
- **C0 (v3).** A claim is a connected component of at least 0.5 ml from one agent output. VoxTell crumbs within 10 mm of the primary organ are grouped (≤ 5 mm apart), with a 0.2 ml floor for the group.
- **C1 (changed).** If the primary is in view and a claim lies within 10 mm of a primary organ mask, its reference host is the primary. Otherwise, as v3: the prompt host, or the host whose envelope holds most of the claim.
- **C2 (v3).** Prompt gate: a prompted claim with less than 50% inside its host envelope is dropped.
- **C3 (v3).** Slab: a flat claim is dropped unless the other agent claims the same voxels.
- **C4 (changed).** Organ-shaped:
  - reference host is the primary: kept, flagged `organ_like` (×0.3, sent to image-grounding trim), never dropped;
  - other hosts: v3 unchanged.

## Lesion building (v3, one change)
- **v3.** Lesions are VoxTell-anchored: VoxTell ∪ BiomedParse within 5 mm ∪ BiomedParse 2-D pieces touching VoxTell on its slices, grouped by 5 mm. The remaining BiomedParse forms separate lesions.
- **Changed.** "BiomedParse never joins inside another TotalSegmentator organ" no longer applies within 10 mm of the primary organ. A tumour absorbed into a neighbour's label can then still join.

## Lesion host
- **V0.** If the primary is out of view, all lesions go to V4.
- **V1.** A lesion within 10 mm of a primary organ mask is a primary candidate. There are no overlap conditions: v3's "≥ 50% inside" and "< 50% inside other organs" are removed.
- **V2.** Organ-likeness of a primary candidate is a flag only.
- **V3.** The host of a primary candidate is the primary, with certainty.
  - Paired organ with unresolved side: the side within 10 mm; if both, the side with the larger boundary-band share.
  - Other spread-set organs within 10 mm are recorded as involved organs, with band shares (`extends_into`). This is a multi-organ-contact feature.
- **V4.** Every other lesion has an uncertain host.
  - Possible hosts: secondary hosts within 10 mm of their mask, plus hosts without a mask at the floor ε (flagged, never admitted from).
  - P(h) ∝ w_h · L(3 mm band share), with w = 0.3 for local invasion and 0.15 for distant; "none" = 0.25 · ε.
- **V5.** Tumour rules (v3, in order):
  - primaries: largest confirmed first, then multifocal;
  - local invasion: within 10 mm of an admitted primary;
  - distant: BiomedParse ∩ that site's VoxTell prompt, plus prompt coherence;
  - V4 lesions: rules evaluated under each possible host and averaged by P(h).

## Open
1. The 80% "inside the host" condition for single-model extra primary lesions: keep, or drop as a host-overlap rule? Dropping it added 49 lesions in pilot 1.
2. Expected risk of V1: within 10 mm, agent claims on adjacent bowel loops or vessels become primary candidates. They are checked by the tumour rules and later by image grounding (elongation, closedness).
