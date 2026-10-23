"""Reasoning-layer vertebral spans (20 Oct 2026; option 1 agreed with the user). The planner keeps the frozen KB spans (VoxTell was
run with them); ledger v8 and anatomy QC v8 read the spans below: measured on TotalSegmentator CT expert anatomy where possible
(knowledge_base/reasoning_spans_v1.json, scripts/spans_measure.py), else the KB span with its variability (1 level, design value).
The in-view share uses the raw span against the scan frame widened by (variability + 0.5) levels on each side, the same margin as
the knowledge-base region, so a slightly misplaced span or a frame cut near the organ does not remove the organ as a host."""
import json, os
_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'knowledge_base', 'reasoning_spans_v1.json')
_S = json.load(open(_P))['spans'] if os.path.exists(_P) else {}
MARGIN = 0.5


def span(kb, ent):
    if ent in _S:
        return _S[ent]['span']
    return ((kb.entities.get(ent, {}) or {}).get('anchor') or {}).get('span')


def var(kb, ent):
    if ent in _S:
        return float(_S[ent]['span_var'])
    return float(((kb.entities.get(ent, {}) or {}).get('anchor') or {}).get('span_var', 1))


def inside(kb, ent, frame):
    """share of the organ's span (levels) inside the scan frame widened by var + 0.5 levels on each side; None if unknown"""
    sp = span(kb, ent)
    if not sp or not frame or not frame[0] or not frame[1]:
        return None
    try:
        a, b = kb.level_index(sp[0]), kb.level_index(sp[1]); f0, f1 = kb.level_index(frame[0]), kb.level_index(frame[1])
    except ValueError:
        return None
    m = var(kb, ent) + MARGIN
    return float(min(1.0, max(0.0, min(b, f1 + m) - max(a, f0 - m) + 1) / (b - a + 1)))


def source(ent):
    return 'measured' if ent in _S else 'kb'
