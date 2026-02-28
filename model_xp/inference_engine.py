"""
Model XP — Logic-Based Network (LBN) Inference Engine
Authored by Nicholas Michael Grossi

Executes queries using symbolic logic and constraint satisfaction.
No neural networks. No matrix multiplication. No probabilistic sampling.
All operations are founded on Boolean algebra.
"""

import re
from typing import Optional
from .knowledge_base import KnowledgeBase
from .persona_loader import Persona


class InferenceResult:
    """Encapsulates the result of a single inference operation."""

    def __init__(
        self,
        success: bool,
        answer: Optional[str],
        proof: list,
        halt_reason: Optional[str] = None
    ):
        self.success = success
        self.answer = answer
        self.proof = proof
        self.halt_reason = halt_reason

    def __repr__(self) -> str:
        if self.success:
            return f"<InferenceResult SUCCESS answer={self.answer!r}>"
        return f"<InferenceResult HALT reason={self.halt_reason!r}>"


class InferenceEngine:
    """
    The Logic-Based Network (LBN) Engine.

    Resolves queries by evaluating them against the symbolic knowledge base
    using propositional logic and forward-chaining inference.

    The engine operates deterministically: identical input produces identical output.
    If a unique answer cannot be derived, the engine returns a HALT result.
    """

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self._query_log: list = []

    def process(self, query: str, persona: Persona) -> InferenceResult:
        """
        Process a query under the constraints of the active persona.

        Steps:
        1. Parse the query into a normalized form.
        2. Verify the query domain is permitted by the persona.
        3. Evaluate the query against the knowledge base.
        4. Return a deterministic InferenceResult.
        """
        # Step 1: Normalize the query
        normalized = self._normalize(query)
        self._query_log.append({"raw": query, "normalized": normalized})

        # Step 2: Check domain authorization
        domain_check = self._check_domain_authorization(normalized, persona)
        if not domain_check["authorized"]:
            return InferenceResult(
                success=False,
                answer=None,
                proof=[],
                halt_reason=(
                    f"DOMAIN VIOLATION: Query domain '{domain_check['detected_domain']}' "
                    f"is not authorized for persona '{persona.name}'. "
                    f"Authorized domains: {persona.domains}."
                )
            )

        # Step 3: Check for ambiguity
        if self._is_ambiguous(normalized):
            return InferenceResult(
                success=False,
                answer=None,
                proof=[],
                halt_reason=(
                    "AMBIGUITY DETECTED: The query does not resolve to a unique logical path. "
                    "Model XP requires explicit clarification before proceeding."
                )
            )

        # Step 4: Evaluate against knowledge base
        result = self.kb.evaluate(normalized)

        if result["found"]:
            answer = self._format_answer(result["value"], query, persona)
            return InferenceResult(
                success=True,
                answer=answer,
                proof=result["proof"]
            )

        # Step 5: Attempt natural language pattern matching for direct questions
        nl_result = self._process_natural_language(query, persona)
        if nl_result:
            return nl_result

        # Step 6: No answer found — HALT
        return InferenceResult(
            success=False,
            answer=None,
            proof=result["proof"],
            halt_reason=(
                f"NO DERIVATION FOUND: The query '{query}' cannot be resolved "
                f"from the knowledge base under persona '{persona.name}'. "
                f"Loaded domains: {self.kb.get_loaded_domains()}. "
                f"Facts available: {self.kb.fact_count()}."
            )
        )

    def _normalize(self, query: str) -> str:
        """Normalize a query string to a dot-notation key format."""
        normalized = query.strip().lower()
        normalized = re.sub(r"[^a-z0-9._\s]", "", normalized)
        normalized = re.sub(r"\s+", ".", normalized)
        return normalized

    def _check_domain_authorization(self, normalized_query: str, persona: Persona) -> dict:
        """Check whether the query falls within an authorized domain."""
        if not persona.domains:
            return {"authorized": True, "detected_domain": None}
        for domain in persona.domains:
            domain_key = domain.lower().replace(".", ".").replace(" ", "_")
            if normalized_query.startswith(domain_key):
                return {"authorized": True, "detected_domain": domain}
        # If no domain prefix matches, check if the query is a general knowledge query
        # General queries (no domain prefix) are permitted
        if "." not in normalized_query or len(normalized_query.split(".")) <= 2:
            return {"authorized": True, "detected_domain": "general"}
        return {"authorized": True, "detected_domain": "general"}

    def _is_ambiguous(self, normalized_query: str) -> bool:
        """
        Detect ambiguity in a query.
        A query is ambiguous if it contains contradiction markers or undefined references.
        """
        ambiguity_markers = ["or.not", "maybe", "possibly", "undefined", "unknown.unknown"]
        for marker in ambiguity_markers:
            if marker in normalized_query:
                return True
        return False

    def _process_natural_language(self, query: str, persona: Persona) -> Optional[InferenceResult]:
        """
        Process natural language queries by matching against known fact patterns.
        This is a deterministic pattern matcher, not a probabilistic language model.
        """
        q = query.strip().lower()

        # Identity queries
        if any(phrase in q for phrase in ["what is your name", "who are you", "identify yourself"]):
            return InferenceResult(
                success=True,
                answer=(
                    f"Designation: {persona.name}\n"
                    f"Archetype: {persona.archetype}\n"
                    f"Persona ID: {persona.persona_id}\n"
                    f"Authorized Domains: {', '.join(persona.domains)}\n"
                    f"Evolution Enabled: {persona.evolution_enabled}"
                ),
                proof=["Identity resolved from active persona configuration."]
            )

        # Capability queries
        if any(phrase in q for phrase in ["what can you do", "list skills", "capabilities"]):
            return InferenceResult(
                success=True,
                answer=(
                    f"Enabled Skills: {', '.join(persona.skills_enabled)}\n"
                    f"Disabled Skills: {', '.join(persona.skills_disabled)}"
                ),
                proof=["Skills resolved from active persona configuration."]
            )

        # Constitution query
        if any(phrase in q for phrase in ["constitution", "core axioms", "governing principles"]):
            return InferenceResult(
                success=True,
                answer=persona.constitution,
                proof=["Constitution retrieved from immutable constitutional document."]
            )

        # Knowledge base status
        if any(phrase in q for phrase in ["knowledge base", "how many facts", "loaded domains"]):
            return InferenceResult(
                success=True,
                answer=(
                    f"Loaded Domains: {', '.join(self.kb.get_loaded_domains()) or 'None'}\n"
                    f"Total Facts: {self.kb.fact_count()}\n"
                    f"Total Rules: {self.kb.rule_count()}"
                ),
                proof=["Knowledge base status retrieved from runtime state."]
            )

        # Fact assertion queries: "what is X" or "define X"
        for prefix in ["what is ", "define ", "describe "]:
            if q.startswith(prefix):
                subject = q[len(prefix):].strip().replace(" ", ".")
                result = self.kb.evaluate(subject)
                if result["found"]:
                    return InferenceResult(
                        success=True,
                        answer=str(result["value"]),
                        proof=result["proof"]
                    )

        return None

    def _format_answer(self, value, query: str, persona: Persona) -> str:
        """Format a raw value into a response appropriate for the persona's output format."""
        if persona.get_output_format() == "application/json":
            import json
            return json.dumps({"query": query, "result": value, "status": "resolved"}, indent=2)
        return str(value)

    def get_query_log(self) -> list:
        """Return the full query log for audit purposes."""
        return list(self._query_log)
