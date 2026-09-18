"""pancia_kb — anchor-centric anatomical knowledge base + deterministic prompt planner.

    from pancia_kb import KnowledgeBase, Planner
    kb = KnowledgeBase.load('knowledge_base/')
    kb.validate()
    plan = Planner(kb).plan(ts_output, modality='MR', cancer_type='TCGA-CESC', sex='female')
    plan.prompts   -> list of PromptJob (term, aliases, gate, wave, reason chain)
    plan.log       -> per-decision audit trail (json-serialisable)
"""
from .kb import KnowledgeBase
from .planner import Planner, TSOutput, TSStructure, Plan, PromptJob

__all__ = ['KnowledgeBase', 'Planner', 'TSOutput', 'TSStructure', 'Plan', 'PromptJob']
__version__ = '0.1.0'
