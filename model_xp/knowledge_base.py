"""
Model XP — Knowledge Base
Authored by Nicholas Michael Grossi

A symbolic knowledge base that stores logical propositions as facts and rules.
All data is binarized into true/false assertions. No vector embeddings. No neural weights.
"""

import json
import os
from typing import Optional


class KnowledgeBase:
    """
    A structured repository of logical propositions.

    Facts are stored as key-value pairs where the key is a subject-predicate string
    and the value is a boolean or a string assertion.

    Rules are stored as conditional mappings: if condition_key is True, then conclusion_key is True.
    """

    def __init__(self):
        self._facts: dict = {}
        self._rules: list = []
        self._domains_loaded: set = set()

    def load_domain(self, domain_path: str, domain_name: str) -> None:
        """Load a knowledge domain from a JSON file into the knowledge base."""
        if not os.path.exists(domain_path):
            return
        with open(domain_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        facts = data.get("facts", {})
        rules = data.get("rules", [])
        self._facts.update(facts)
        self._rules.extend(rules)
        self._domains_loaded.add(domain_name)

    def assert_fact(self, key: str, value) -> None:
        """Assert a fact into the knowledge base."""
        self._facts[key] = value

    def query_fact(self, key: str) -> Optional[object]:
        """Query a fact by key. Returns None if not found."""
        return self._facts.get(key, None)

    def evaluate(self, query: str) -> dict:
        """
        Evaluate a query against the knowledge base using forward chaining.

        The query is a dot-notation key (e.g., "finance.sp500.has_ceo").
        Returns a result dict with 'found', 'value', and 'proof' fields.
        """
        # Direct fact lookup
        if query in self._facts:
            return {
                "found": True,
                "value": self._facts[query],
                "proof": [f"Direct fact assertion: {query} = {self._facts[query]}"]
            }

        # Forward chaining through rules
        proof_chain = []
        derived = dict(self._facts)
        changed = True
        iterations = 0
        max_iterations = 10

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for rule in self._rules:
                condition = rule.get("if")
                conclusion = rule.get("then")
                conclusion_key = rule.get("conclude")
                if condition in derived and derived[condition]:
                    if conclusion_key not in derived:
                        derived[conclusion_key] = conclusion
                        proof_chain.append(
                            f"Rule applied: IF {condition} THEN {conclusion_key} = {conclusion}"
                        )
                        changed = True

        if query in derived:
            return {
                "found": True,
                "value": derived[query],
                "proof": proof_chain + [f"Derived: {query} = {derived[query]}"]
            }

        return {
            "found": False,
            "value": None,
            "proof": [f"Query '{query}' not found in knowledge base after {iterations} inference iterations."]
        }

    def get_loaded_domains(self) -> list:
        """Return the list of loaded domain names."""
        return sorted(self._domains_loaded)

    def fact_count(self) -> int:
        """Return the total number of facts in the knowledge base."""
        return len(self._facts)

    def rule_count(self) -> int:
        """Return the total number of rules in the knowledge base."""
        return len(self._rules)
