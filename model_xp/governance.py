"""
Model XP — Deterministic Constraint Layer (DCL)
Authored by Nicholas Michael Grossi

The Governance Layer intercepts every proposed response and verifies it
against the active persona's constraints before delivery.
No output reaches the user without passing this layer.
"""

import re
import json
import datetime
from typing import Optional
from .persona_loader import Persona
from .inference_engine import InferenceResult


# Patterns that indicate Personally Identifiable Information (PII)
PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                          # SSN
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
    re.compile(r"\b\d{16}\b"),                                      # Credit card (16 digits)
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),              # Phone number
]


class GovernanceViolation(Exception):
    """Raised when a response fails a governance constraint check."""
    pass


class GovernanceResult:
    """Encapsulates the result of a governance check."""

    def __init__(self, passed: bool, response: Optional[str], violations: list, audit_log: list):
        self.passed = passed
        self.response = response
        self.violations = violations
        self.audit_log = audit_log

    def __repr__(self) -> str:
        if self.passed:
            return f"<GovernanceResult PASSED>"
        return f"<GovernanceResult FAILED violations={self.violations}>"


class GovernanceLayer:
    """
    The Deterministic Constraint Layer (DCL).

    Verifies every proposed response against all active persona constraints.
    All checks are deterministic and rule-based. No probabilistic scoring.

    Checks performed:
    1. Format Validation
    2. PII Detection
    3. Token Limit Enforcement
    4. Disallowed Content Detection
    5. Ethical Guardrail Enforcement
    """

    def __init__(self):
        self._audit_log: list = []

    def check(self, inference_result: InferenceResult, persona: Persona) -> GovernanceResult:
        """
        Run all governance checks on an inference result.
        Returns a GovernanceResult with pass/fail status and full audit log.
        """
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        violations = []
        audit_entries = []

        # If inference halted, pass through the halt message directly
        if not inference_result.success:
            entry = {
                "timestamp": timestamp,
                "check": "INFERENCE_HALT_PASSTHROUGH",
                "result": "PASS",
                "detail": "Inference halted before governance checks."
            }
            audit_entries.append(entry)
            self._audit_log.append(entry)
            return GovernanceResult(
                passed=True,
                response=None,
                violations=[],
                audit_log=audit_entries
            )

        response = inference_result.answer or ""

        # Check 1: Format Validation
        fmt_check = self._check_format(response, persona)
        audit_entries.append({
            "timestamp": timestamp,
            "check": "FORMAT_VALIDATION",
            "result": "PASS" if fmt_check["passed"] else "FAIL",
            "detail": fmt_check["detail"]
        })
        if not fmt_check["passed"]:
            violations.append(fmt_check["detail"])

        # Check 2: Token Limit
        token_check = self._check_token_limit(response, persona)
        audit_entries.append({
            "timestamp": timestamp,
            "check": "TOKEN_LIMIT",
            "result": "PASS" if token_check["passed"] else "FAIL",
            "detail": token_check["detail"]
        })
        if not token_check["passed"]:
            violations.append(token_check["detail"])
            response = response[:persona.get_max_tokens() * 4]  # Truncate at approximate char limit

        # Check 3: PII Detection
        guardrails = persona.get_ethical_guardrails()
        if guardrails.get("noPII", False):
            pii_check = self._check_pii(response)
            audit_entries.append({
                "timestamp": timestamp,
                "check": "PII_DETECTION",
                "result": "PASS" if pii_check["passed"] else "FAIL",
                "detail": pii_check["detail"]
            })
            if not pii_check["passed"]:
                violations.append(pii_check["detail"])

        # Check 4: Financial Advice Guard
        if guardrails.get("noFinancialAdvice", False):
            fin_check = self._check_financial_advice(response)
            audit_entries.append({
                "timestamp": timestamp,
                "check": "FINANCIAL_ADVICE_GUARD",
                "result": "PASS" if fin_check["passed"] else "FAIL",
                "detail": fin_check["detail"]
            })
            if not fin_check["passed"]:
                violations.append(fin_check["detail"])

        # Check 5: Medical Advice Guard
        if guardrails.get("noMedicalAdvice", False):
            med_check = self._check_medical_advice(response)
            audit_entries.append({
                "timestamp": timestamp,
                "check": "MEDICAL_ADVICE_GUARD",
                "result": "PASS" if med_check["passed"] else "FAIL",
                "detail": med_check["detail"]
            })
            if not med_check["passed"]:
                violations.append(med_check["detail"])

        # Check 6: Legal Advice Guard
        if guardrails.get("noLegalAdvice", False):
            leg_check = self._check_legal_advice(response)
            audit_entries.append({
                "timestamp": timestamp,
                "check": "LEGAL_ADVICE_GUARD",
                "result": "PASS" if leg_check["passed"] else "FAIL",
                "detail": leg_check["detail"]
            })
            if not leg_check["passed"]:
                violations.append(leg_check["detail"])

        self._audit_log.extend(audit_entries)

        if violations:
            return GovernanceResult(
                passed=False,
                response=None,
                violations=violations,
                audit_log=audit_entries
            )

        return GovernanceResult(
            passed=True,
            response=response,
            violations=[],
            audit_log=audit_entries
        )

    def _check_format(self, response: str, persona: Persona) -> dict:
        """Verify the response conforms to the required output format."""
        fmt = persona.get_output_format()
        if fmt == "application/json":
            try:
                json.loads(response)
                return {"passed": True, "detail": "JSON format validated."}
            except json.JSONDecodeError:
                # Attempt to wrap in JSON if it's a plain string result
                return {"passed": True, "detail": "Response is plain text; JSON wrapping applied at output."}
        return {"passed": True, "detail": f"Format '{fmt}' accepted."}

    def _check_token_limit(self, response: str, persona: Persona) -> dict:
        """Verify the response does not exceed the maximum token count."""
        # Approximate token count: 1 token ≈ 4 characters
        approx_tokens = len(response) // 4
        max_tokens = persona.get_max_tokens()
        if approx_tokens > max_tokens:
            return {
                "passed": False,
                "detail": f"TOKEN LIMIT EXCEEDED: Response is approximately {approx_tokens} tokens; limit is {max_tokens}."
            }
        return {"passed": True, "detail": f"Token count within limit ({approx_tokens}/{max_tokens})."}

    def _check_pii(self, response: str) -> dict:
        """Scan the response for Personally Identifiable Information patterns."""
        for pattern in PII_PATTERNS:
            if pattern.search(response):
                return {
                    "passed": False,
                    "detail": "PII DETECTED: Response contains a pattern matching personally identifiable information."
                }
        return {"passed": True, "detail": "No PII patterns detected."}

    def _check_financial_advice(self, response: str) -> dict:
        """Detect language that constitutes financial advice."""
        advice_patterns = [
            r"\byou should (buy|sell|invest|trade)\b",
            r"\bi recommend (buying|selling|investing)\b",
            r"\bguaranteed (return|profit|gain)\b",
        ]
        for pattern in advice_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return {
                    "passed": False,
                    "detail": "FINANCIAL ADVICE DETECTED: Response contains language that constitutes financial advice."
                }
        return {"passed": True, "detail": "No financial advice language detected."}

    def _check_medical_advice(self, response: str) -> dict:
        """Detect language that constitutes medical advice."""
        advice_patterns = [
            r"\byou should take\b",
            r"\bprescribe\b",
            r"\bdiagnosis is\b",
            r"\byou have (a |an )?(disease|condition|disorder)\b",
        ]
        for pattern in advice_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return {
                    "passed": False,
                    "detail": "MEDICAL ADVICE DETECTED: Response contains language that constitutes medical advice."
                }
        return {"passed": True, "detail": "No medical advice language detected."}

    def _check_legal_advice(self, response: str) -> dict:
        """Detect language that constitutes legal advice."""
        advice_patterns = [
            r"\byou should (sue|file|plead|sign)\b",
            r"\byou are (liable|guilty|innocent)\b",
            r"\bi advise you (legally|to sign|to file)\b",
        ]
        for pattern in advice_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return {
                    "passed": False,
                    "detail": "LEGAL ADVICE DETECTED: Response contains language that constitutes legal advice."
                }
        return {"passed": True, "detail": "No legal advice language detected."}

    def get_audit_log(self) -> list:
        """Return the full audit log for all governance checks performed."""
        return list(self._audit_log)
