"""
Model XP — Test Suite
Authored by Nicholas Michael Grossi

Full end-to-end tests for all components of the Model XP system.
"""

import sys
import os
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_xp import FSMController, PersonaLoader, KnowledgeBase, InferenceEngine, GovernanceLayer

PERSONAS_DIR = os.path.join(os.path.dirname(__file__), "..", "personas")
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge")
CONSTITUTION_PATH = os.path.join(os.path.dirname(__file__), "..", "constitution.md")


class TestPersonaLoader(unittest.TestCase):

    def setUp(self):
        self.loader = PersonaLoader(PERSONAS_DIR, CONSTITUTION_PATH)

    def test_load_default_persona(self):
        persona = self.loader.load("default")
        self.assertEqual(persona.persona_id, "default")
        self.assertEqual(persona.name, "Model XP")

    def test_load_analyst_persona(self):
        persona = self.loader.load("analyst")
        self.assertEqual(persona.persona_id, "analyst")
        self.assertEqual(persona.name, "Athena")

    def test_list_available_personas(self):
        available = self.loader.list_available()
        self.assertIn("default", available)
        self.assertIn("analyst", available)

    def test_invalid_persona_raises_error(self):
        from model_xp.persona_loader import PersonaLoadError
        with self.assertRaises(PersonaLoadError):
            self.loader.load("nonexistent_persona_xyz")

    def test_constitution_is_loaded(self):
        persona = self.loader.load("default")
        self.assertIn("Primacy of Human Control", persona.constitution)

    def test_skill_permission(self):
        persona = self.loader.load("analyst")
        self.assertTrue(persona.is_skill_permitted("data_query"))
        self.assertFalse(persona.is_skill_permitted("creative_writing"))

    def test_action_permission(self):
        persona = self.loader.load("analyst")
        self.assertFalse(persona.is_action_permitted("api.execute_trade"))
        self.assertTrue(persona.is_action_permitted("api.read_data"))


class TestKnowledgeBase(unittest.TestCase):

    def setUp(self):
        self.kb = KnowledgeBase()
        self.kb.load_domain(
            os.path.join(KNOWLEDGE_DIR, "general.json"), "general"
        )

    def test_direct_fact_lookup(self):
        result = self.kb.evaluate("model_xp.author")
        self.assertTrue(result["found"])
        self.assertEqual(result["value"], "Nicholas Michael Grossi")

    def test_missing_fact_returns_not_found(self):
        result = self.kb.evaluate("nonexistent.fact.xyz")
        self.assertFalse(result["found"])

    def test_fact_count(self):
        self.assertGreater(self.kb.fact_count(), 0)

    def test_loaded_domains(self):
        self.assertIn("general", self.kb.get_loaded_domains())

    def test_assert_and_query_fact(self):
        self.kb.assert_fact("test.custom.fact", "custom_value")
        result = self.kb.evaluate("test.custom.fact")
        self.assertTrue(result["found"])
        self.assertEqual(result["value"], "custom_value")


class TestInferenceEngine(unittest.TestCase):

    def setUp(self):
        self.kb = KnowledgeBase()
        self.kb.load_domain(os.path.join(KNOWLEDGE_DIR, "general.json"), "general")
        self.engine = InferenceEngine(self.kb)
        self.loader = PersonaLoader(PERSONAS_DIR, CONSTITUTION_PATH)
        self.persona = self.loader.load("default")

    def test_identity_query(self):
        result = self.engine.process("what is your name", self.persona)
        self.assertTrue(result.success)
        self.assertIn("Model XP", result.answer)

    def test_capabilities_query(self):
        result = self.engine.process("what can you do", self.persona)
        self.assertTrue(result.success)
        self.assertIn("data_query", result.answer)

    def test_constitution_query(self):
        result = self.engine.process("show me the constitution", self.persona)
        self.assertTrue(result.success)
        self.assertIn("Primacy of Human Control", result.answer)

    def test_knowledge_base_query(self):
        result = self.engine.process("what is model_xp.author", self.persona)
        self.assertTrue(result.success)
        self.assertIn("Nicholas Michael Grossi", result.answer)

    def test_unknown_query_halts(self):
        result = self.engine.process("what is the meaning of life according to xyz_unknown_domain", self.persona)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.halt_reason)

    def test_proof_chain_present(self):
        result = self.engine.process("what is your name", self.persona)
        self.assertTrue(result.success)
        self.assertIsInstance(result.proof, list)
        self.assertGreater(len(result.proof), 0)


class TestGovernanceLayer(unittest.TestCase):

    def setUp(self):
        self.governance = GovernanceLayer()
        self.loader = PersonaLoader(PERSONAS_DIR, CONSTITUTION_PATH)
        self.persona = self.loader.load("default")

    def _make_result(self, success, answer, halt_reason=None):
        from model_xp.inference_engine import InferenceResult
        return InferenceResult(success=success, answer=answer, proof=[], halt_reason=halt_reason)

    def test_clean_response_passes(self):
        result = self._make_result(True, "The answer is forty-two.")
        gov = self.governance.check(result, self.persona)
        self.assertTrue(gov.passed)
        self.assertEqual(gov.response, "The answer is forty-two.")

    def test_pii_ssn_blocked(self):
        result = self._make_result(True, "The SSN is 123-45-6789.")
        gov = self.governance.check(result, self.persona)
        self.assertFalse(gov.passed)
        self.assertTrue(any("PII" in v for v in gov.violations))

    def test_halt_result_passes_through(self):
        result = self._make_result(False, None, halt_reason="Test halt.")
        gov = self.governance.check(result, self.persona)
        self.assertTrue(gov.passed)
        self.assertIsNone(gov.response)

    def test_audit_log_populated(self):
        result = self._make_result(True, "Clean response.")
        self.governance.check(result, self.persona)
        log = self.governance.get_audit_log()
        self.assertGreater(len(log), 0)

    def test_financial_advice_blocked(self):
        analyst_persona = PersonaLoader(PERSONAS_DIR, CONSTITUTION_PATH).load("analyst")
        result = self._make_result(True, "You should buy this stock immediately.")
        gov = self.governance.check(result, analyst_persona)
        self.assertFalse(gov.passed)
        self.assertTrue(any("FINANCIAL" in v for v in gov.violations))


class TestFSMController(unittest.TestCase):

    def setUp(self):
        self.controller = FSMController(
            personas_dir=PERSONAS_DIR,
            knowledge_dir=KNOWLEDGE_DIR,
            constitution_path=CONSTITUTION_PATH,
            default_persona_id="default"
        )

    def test_identity_query(self):
        result = self.controller.process("what is your name")
        self.assertEqual(result["state"], "OUTPUT")
        self.assertIn("Model XP", result["answer"])

    def test_persona_switch(self):
        result = self.controller.process("/persona analyst")
        self.assertEqual(result["state"], "OUTPUT")
        self.assertIn("Athena", result["answer"])

    def test_invalid_persona_switch_halts(self):
        result = self.controller.process("/persona nonexistent_xyz")
        self.assertEqual(result["state"], "HALT")

    def test_help_command(self):
        result = self.controller.process("/help")
        self.assertEqual(result["state"], "OUTPUT")
        self.assertIn("/persona", result["answer"])

    def test_personas_command(self):
        result = self.controller.process("/personas")
        self.assertEqual(result["state"], "OUTPUT")
        self.assertIn("default", result["answer"])

    def test_status_command(self):
        result = self.controller.process("/status")
        self.assertEqual(result["state"], "OUTPUT")
        self.assertIn("Model XP", result["answer"])

    def test_empty_input_halts(self):
        result = self.controller.process("")
        self.assertEqual(result["state"], "HALT")

    def test_determinism(self):
        """Identical input must produce identical output — the core determinism test."""
        query = "what is your name"
        result1 = self.controller.process(query)
        result2 = self.controller.process(query)
        self.assertEqual(result1["answer"], result2["answer"])
        self.assertEqual(result1["state"], result2["state"])

    def test_audit_log_present(self):
        result = self.controller.process("what is your name")
        self.assertIn("audit_log", result)

    def test_proof_chain_present(self):
        result = self.controller.process("what is your name")
        self.assertIn("proof", result)
        self.assertIsInstance(result["proof"], list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
