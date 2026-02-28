"""
Model XP — An Architecture for Deterministic Artificial Intelligence
Authored by Nicholas Michael Grossi

A logic-based AI engine that operates without neural networks,
without probabilistic sampling, and without GPU requirements.
Runs on any hardware. Every output is deterministic and auditable.
"""

from .fsm_controller import FSMController
from .persona_loader import PersonaLoader, Persona
from .knowledge_base import KnowledgeBase
from .inference_engine import InferenceEngine
from .governance import GovernanceLayer

__version__ = "1.0.0"
__author__ = "Nicholas Michael Grossi"
__all__ = [
    "FSMController",
    "PersonaLoader",
    "Persona",
    "KnowledgeBase",
    "InferenceEngine",
    "GovernanceLayer",
]
