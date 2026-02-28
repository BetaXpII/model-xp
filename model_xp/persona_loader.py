"""
Model XP — Persona Loader
Authored by Nicholas Michael Grossi

Loads, validates, and manages JSON persona configuration files.
"""

import json
import os
from typing import Optional


REQUIRED_KEYS = {"schemaVersion", "personaId", "identity", "knowledge", "skills", "constraints", "evolution"}


class PersonaLoadError(Exception):
    """Raised when a persona file cannot be loaded or fails validation."""
    pass


class Persona:
    """Represents a loaded and validated persona configuration."""

    def __init__(self, data: dict, constitution_text: str):
        self.data = data
        self.constitution = constitution_text
        self.persona_id: str = data["personaId"]
        self.name: str = data["identity"]["name"]
        self.archetype: str = data["identity"]["archetype"]
        self.domains: list = data["knowledge"]["domains"]
        self.allow_inference: bool = data["knowledge"].get("allowInference", False)
        self.inference_depth: int = data["knowledge"].get("inferenceDepth", 1)
        self.skills_enabled: list = data["skills"]["enabled"]
        self.skills_disabled: list = data["skills"]["disabled"]
        self.constraints: dict = data["constraints"]
        self.evolution_enabled: bool = data["evolution"]["enabled"]

    def is_skill_permitted(self, skill: str) -> bool:
        """Returns True if the skill is enabled and not disabled."""
        if skill in self.skills_disabled:
            return False
        return skill in self.skills_enabled

    def is_action_permitted(self, action: str) -> bool:
        """Returns True if the action is not in the disallowed actions list."""
        disallowed = self.constraints.get("disallowedActions", [])
        return action not in disallowed

    def get_max_tokens(self) -> int:
        """Returns the maximum permitted response token count."""
        return self.constraints.get("maxResponseTokens", 2048)

    def get_output_format(self) -> str:
        """Returns the required output format."""
        return self.constraints.get("outputFormat", "text/plain")

    def get_ethical_guardrails(self) -> dict:
        """Returns the ethical guardrail flags."""
        return self.constraints.get("ethicalGuardrails", {})

    def __repr__(self) -> str:
        return f"<Persona id={self.persona_id} name={self.name} archetype={self.archetype}>"


class PersonaLoader:
    """
    Loads persona JSON files and the immutable constitution from a directory.
    The constitution is always loaded and cannot be overridden by any persona.
    """

    def __init__(self, personas_dir: str, constitution_path: Optional[str] = None):
        self.personas_dir = personas_dir
        self.constitution_path = constitution_path
        self._constitution_text = self._load_constitution()

    def _load_constitution(self) -> str:
        if self.constitution_path and os.path.exists(self.constitution_path):
            with open(self.constitution_path, "r", encoding="utf-8") as f:
                return f.read()
        return (
            "CONSTITUTION OF MODEL XP\n"
            "1. Primacy of Human Control: Model XP defers to the human operator.\n"
            "2. Truthful Communication: Model XP states only verifiable facts.\n"
            "3. Deterministic Execution: Model XP halts on ambiguity.\n"
            "4. Operational Transparency: All decisions are logged and auditable.\n"
        )

    def load(self, persona_id: str) -> Persona:
        """Load a persona by its ID from the personas directory."""
        path = os.path.join(self.personas_dir, f"{persona_id}.json")
        if not os.path.exists(path):
            raise PersonaLoadError(f"Persona file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise PersonaLoadError(f"Invalid JSON in persona file '{path}': {e}")
        self._validate(data, path)
        return Persona(data, self._constitution_text)

    def list_available(self) -> list:
        """Return a list of available persona IDs in the personas directory."""
        ids = []
        for fname in os.listdir(self.personas_dir):
            if fname.endswith(".json"):
                ids.append(fname[:-5])
        return sorted(ids)

    def _validate(self, data: dict, path: str) -> None:
        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            raise PersonaLoadError(
                f"Persona file '{path}' is missing required keys: {missing}"
            )
        if data.get("schemaVersion") != "1.0":
            raise PersonaLoadError(
                f"Persona file '{path}' has unsupported schemaVersion: {data.get('schemaVersion')}"
            )
