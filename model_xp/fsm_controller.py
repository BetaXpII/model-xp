"""
Model XP — Finite State Machine (FSM) Controller
Authored by Nicholas Michael Grossi

Manages the operational lifecycle of Model XP through a sequence of
discrete, defined states. Prevents non-deterministic behavior and ensures
every action is part of a controlled, auditable process.

States:
  IDLE → LOAD_PERSONA → VALIDATE_INPUT → INFERENCE → GOVERNANCE_CHECK → OUTPUT
                                                                        ↓
                                                                   HALT/ERROR
"""

import datetime
from enum import Enum, auto
from typing import Optional

from .persona_loader import PersonaLoader, Persona, PersonaLoadError
from .knowledge_base import KnowledgeBase
from .inference_engine import InferenceEngine
from .governance import GovernanceLayer


class State(Enum):
    IDLE = auto()
    LOAD_PERSONA = auto()
    VALIDATE_INPUT = auto()
    INFERENCE = auto()
    GOVERNANCE_CHECK = auto()
    OUTPUT = auto()
    HALT = auto()


class FSMController:
    """
    The Finite State Machine Controller for Model XP.

    Orchestrates the full processing pipeline from input receipt to output delivery.
    Each state transition is logged. The system cannot skip states or go backward.
    """

    def __init__(
        self,
        personas_dir: str,
        knowledge_dir: str,
        constitution_path: Optional[str] = None,
        default_persona_id: str = "default"
    ):
        self.personas_dir = personas_dir
        self.knowledge_dir = knowledge_dir
        self.default_persona_id = default_persona_id

        self._loader = PersonaLoader(personas_dir, constitution_path)
        self._kb = KnowledgeBase()
        self._engine = InferenceEngine(self._kb)
        self._governance = GovernanceLayer()

        self._state: State = State.IDLE
        self._active_persona: Optional[Persona] = None
        self._state_log: list = []

        # Load the default persona on initialization
        self._load_persona(default_persona_id)

    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------

    def process(self, user_input: str) -> dict:
        """
        Process a user input through the full FSM pipeline.
        Returns a structured response dict with state, answer, proof, and audit log.
        """
        self._transition(State.LOAD_PERSONA)

        # Check for persona-switch command: /persona <id>
        if user_input.strip().startswith("/persona "):
            persona_id = user_input.strip()[9:].strip()
            return self._handle_persona_switch(persona_id)

        # Check for system commands
        if user_input.strip().startswith("/"):
            return self._handle_system_command(user_input.strip())

        self._transition(State.VALIDATE_INPUT)
        validation = self._validate_input(user_input)
        if not validation["valid"]:
            return self._halt(validation["reason"])

        self._transition(State.INFERENCE)
        inference_result = self._engine.process(user_input, self._active_persona)

        self._transition(State.GOVERNANCE_CHECK)
        governance_result = self._governance.check(inference_result, self._active_persona)

        if not inference_result.success:
            self._transition(State.HALT)
            return {
                "state": "HALT",
                "persona": self._active_persona.name,
                "answer": None,
                "halt_reason": inference_result.halt_reason,
                "proof": inference_result.proof,
                "audit_log": governance_result.audit_log,
                "timestamp": self._now()
            }

        if not governance_result.passed:
            self._transition(State.HALT)
            return {
                "state": "HALT",
                "persona": self._active_persona.name,
                "answer": None,
                "halt_reason": f"GOVERNANCE VIOLATION: {'; '.join(governance_result.violations)}",
                "proof": inference_result.proof,
                "audit_log": governance_result.audit_log,
                "timestamp": self._now()
            }

        self._transition(State.OUTPUT)
        self._transition(State.IDLE)

        return {
            "state": "OUTPUT",
            "persona": self._active_persona.name,
            "answer": governance_result.response,
            "halt_reason": None,
            "proof": inference_result.proof,
            "audit_log": governance_result.audit_log,
            "timestamp": self._now()
        }

    def get_active_persona(self) -> Optional[Persona]:
        """Return the currently active persona."""
        return self._active_persona

    def get_state(self) -> State:
        """Return the current FSM state."""
        return self._state

    def get_state_log(self) -> list:
        """Return the full state transition log."""
        return list(self._state_log)

    def list_personas(self) -> list:
        """Return a list of available persona IDs."""
        return self._loader.list_available()

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _load_persona(self, persona_id: str) -> bool:
        """Load a persona by ID. Returns True on success, False on failure."""
        try:
            persona = self._loader.load(persona_id)
            self._active_persona = persona
            # Load knowledge domains for this persona
            self._kb = KnowledgeBase()
            for domain in persona.domains:
                domain_file = domain.replace(".", "_") + ".json"
                domain_path = f"{self.knowledge_dir}/{domain_file}"
                self._kb.load_domain(domain_path, domain)
            self._engine = InferenceEngine(self._kb)
            return True
        except PersonaLoadError:
            # Fall back to a minimal built-in default persona
            if persona_id != "default":
                return False
            self._active_persona = self._create_fallback_persona()
            return True

    def _create_fallback_persona(self) -> Persona:
        """Create a minimal built-in fallback persona when no file is found."""
        from .persona_loader import Persona
        data = {
            "schemaVersion": "1.0",
            "personaId": "default",
            "identity": {
                "name": "Model XP",
                "archetype": "General Assistant",
                "description": "The default Model XP persona.",
                "voice": {"tone": "formal", "formality": 8, "style": "concise", "avoid": []}
            },
            "knowledge": {
                "domains": [],
                "accessLevel": "public_data_only",
                "allowInference": True,
                "inferenceDepth": 3
            },
            "skills": {
                "enabled": ["data_query", "report_generation"],
                "disabled": []
            },
            "constraints": {
                "outputFormat": "text/plain",
                "maxResponseTokens": 4096,
                "disallowedActions": [],
                "ethicalGuardrails": {"noPII": True}
            },
            "evolution": {"enabled": False}
        }
        constitution = (
            "CONSTITUTION OF MODEL XP\n"
            "1. Primacy of Human Control.\n"
            "2. Truthful and Verifiable Communication.\n"
            "3. Deterministic Execution.\n"
            "4. Operational Transparency.\n"
        )
        return Persona(data, constitution)

    def _validate_input(self, user_input: str) -> dict:
        """Validate user input before inference."""
        if not user_input or not user_input.strip():
            return {"valid": False, "reason": "EMPTY INPUT: No query was provided."}
        if len(user_input) > 10000:
            return {"valid": False, "reason": "INPUT TOO LONG: Query exceeds maximum permitted length of 10,000 characters."}
        return {"valid": True, "reason": None}

    def _handle_persona_switch(self, persona_id: str) -> dict:
        """Handle a persona-switch command."""
        success = self._load_persona(persona_id)
        if success and self._active_persona.persona_id == persona_id:
            self._transition(State.IDLE)
            return {
                "state": "OUTPUT",
                "persona": self._active_persona.name,
                "answer": (
                    f"Persona switched successfully.\n"
                    f"Active Persona: {self._active_persona.name} ({self._active_persona.archetype})\n"
                    f"Persona ID: {self._active_persona.persona_id}\n"
                    f"Authorized Domains: {', '.join(self._active_persona.domains) or 'General'}"
                ),
                "halt_reason": None,
                "proof": ["Persona loaded from JSON configuration file."],
                "audit_log": [],
                "timestamp": self._now()
            }
        self._transition(State.HALT)
        available = self.list_personas()
        return self._halt(
            f"PERSONA NOT FOUND: '{persona_id}' does not exist. "
            f"Available personas: {available}"
        )

    def _handle_system_command(self, command: str) -> dict:
        """Handle system commands."""
        if command == "/help":
            help_text = (
                "Model XP — Available Commands\n"
                "─────────────────────────────\n"
                "/help                  Display this help message.\n"
                "/persona <id>          Switch to a different persona.\n"
                "/personas              List all available personas.\n"
                "/status                Display current system status.\n"
                "/audit                 Display the full audit log.\n"
                "/state                 Display the current FSM state.\n"
                "─────────────────────────────\n"
                "Any other input is processed as a query."
            )
            self._transition(State.IDLE)
            return {"state": "OUTPUT", "persona": self._active_persona.name,
                    "answer": help_text, "halt_reason": None,
                    "proof": [], "audit_log": [], "timestamp": self._now()}

        if command == "/personas":
            available = self.list_personas()
            self._transition(State.IDLE)
            return {"state": "OUTPUT", "persona": self._active_persona.name,
                    "answer": f"Available Personas: {', '.join(available) if available else 'None found.'}",
                    "halt_reason": None, "proof": [], "audit_log": [], "timestamp": self._now()}

        if command == "/status":
            p = self._active_persona
            status = (
                f"Active Persona:    {p.name} ({p.archetype})\n"
                f"Persona ID:        {p.persona_id}\n"
                f"FSM State:         {self._state.name}\n"
                f"Knowledge Domains: {', '.join(self.kb_domains()) or 'None loaded'}\n"
                f"Facts Loaded:      {self._kb.fact_count()}\n"
                f"Rules Loaded:      {self._kb.rule_count()}\n"
                f"Evolution:         {'Enabled' if p.evolution_enabled else 'Disabled'}"
            )
            self._transition(State.IDLE)
            return {"state": "OUTPUT", "persona": p.name,
                    "answer": status, "halt_reason": None,
                    "proof": [], "audit_log": [], "timestamp": self._now()}

        if command == "/audit":
            log = self._governance.get_audit_log()
            if not log:
                answer = "Audit log is empty."
            else:
                import json
                answer = json.dumps(log, indent=2)
            self._transition(State.IDLE)
            return {"state": "OUTPUT", "persona": self._active_persona.name,
                    "answer": answer, "halt_reason": None,
                    "proof": [], "audit_log": [], "timestamp": self._now()}

        if command == "/state":
            self._transition(State.IDLE)
            return {"state": "OUTPUT", "persona": self._active_persona.name,
                    "answer": f"Current FSM State: {self._state.name}",
                    "halt_reason": None, "proof": [], "audit_log": [], "timestamp": self._now()}

        return self._halt(f"UNKNOWN COMMAND: '{command}'. Type /help for available commands.")

    def kb_domains(self) -> list:
        return self._kb.get_loaded_domains()

    def _transition(self, new_state: State) -> None:
        """Record a state transition."""
        entry = {
            "from": self._state.name,
            "to": new_state.name,
            "timestamp": self._now()
        }
        self._state_log.append(entry)
        self._state = new_state

    def _halt(self, reason: str) -> dict:
        """Transition to HALT state and return a halt response."""
        self._transition(State.HALT)
        self._transition(State.IDLE)
        return {
            "state": "HALT",
            "persona": self._active_persona.name if self._active_persona else "NONE",
            "answer": None,
            "halt_reason": reason,
            "proof": [],
            "audit_log": [],
            "timestamp": self._now()
        }

    def _now(self) -> str:
        return datetime.datetime.utcnow().isoformat() + "Z"
