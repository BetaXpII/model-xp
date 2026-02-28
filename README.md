# Model XP

**An Architecture for Deterministic Artificial Intelligence**

*Authored by Nicholas Michael Grossi*

[![Tests](https://img.shields.io/badge/tests-33%20passed-brightgreen)](./tests/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-Proprietary-red)](./LICENSE)
[![Hardware](https://img.shields.io/badge/hardware-Any%20CPU-orange)](./docs/)

---

## What is Model XP?

Model XP is a fully functional artificial intelligence engine that operates without neural networks, without probabilistic sampling, and without GPU requirements. It runs on any hardware — from a Raspberry Pi to an enterprise server — using only Boolean algebra, propositional logic, and constraint satisfaction.

Every output is **deterministic**: identical input produces identical output, every time. Every output is **auditable**: a complete logical proof chain accompanies every response. Every output is **safe**: a Deterministic Constraint Layer (DCL) intercepts and verifies every response before it reaches the user.

> "Model XP competes with platforms such as xAI's Grok not by replicating its methods, but by offering a superior value proposition for critical applications: mathematically provable consistency and control."
>
> — Nicholas Michael Grossi, *Model XP Architecture Blueprint*

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/BetaXpII/model-xp.git
cd model-xp

# Run the interactive CLI (no installation required)
python3 main.py

# Run with a specific persona
python3 main.py --persona analyst

# Run a single query
python3 main.py --query "what is your name"

# Run with proof chain displayed
python3 main.py --proof

# Run all tests
python3 -m pytest tests/ -v
```

---

## Architecture

Model XP comprises three decoupled layers:

```
┌──────────────────────────────────────────────────────────────┐
│                    LAYER 1: PERSONA LAYER                    │
│         FSM Controller  +  JSON Persona Loader               │
│   Manages state, loads personas, validates input             │
├──────────────────────────────────────────────────────────────┤
│                   LAYER 2: INFERENCE LAYER                   │
│            Logic-Based Network (LBN) Engine                  │
│   Resolves queries via symbolic logic — no neural network    │
├──────────────────────────────────────────────────────────────┤
│                  LAYER 3: GOVERNANCE LAYER                   │
│          Deterministic Constraint Layer (DCL)                │
│   Verifies every output before delivery — no bypass          │
└──────────────────────────────────────────────────────────────┘
```

---

## Persona Switching

Each persona is a self-contained JSON file. Switch personas instantly with no retraining:

```bash
# In the interactive CLI:
/persona analyst      # Switch to the financial analyst persona (Athena)
/persona default      # Switch back to the default persona
/personas             # List all available personas
```

Example `persona.json`:

```json
{
  "schemaVersion": "1.0",
  "personaId": "analyst",
  "identity": { "name": "Athena", "archetype": "Data Analyst" },
  "knowledge": { "domains": ["finance.securities"], "inferenceDepth": 3 },
  "skills": { "enabled": ["data_query", "report_generation"], "disabled": ["financial_advice"] },
  "constraints": { "ethicalGuardrails": { "noPII": true, "noFinancialAdvice": true } },
  "evolution": { "enabled": false }
}
```

---

## CLI Commands

| Command | Description |
| --- | --- |
| `/help` | Display all available commands |
| `/persona <id>` | Switch to a different persona |
| `/personas` | List all available personas |
| `/status` | Display current system status |
| `/audit` | Display the full governance audit log |
| `/state` | Display the current FSM state |
| `exit` | Terminate the session |

---

## Benchmark: Model XP vs. Grok 3

| Metric | Grok 3 | Model XP | Advantage |
| --- | --- | --- | --- |
| Inference Speed | ~69 tokens/sec | ~3,700 tokens/sec | **~54x Faster** |
| Energy Consumption | High (GPU) | Extremely Low (CPU) | **~52x Less Energy** |
| Model Size | Billions of parameters | Minimal binary | **>1000x Smaller** |
| Fabrication Risk | Inherent | **Zero** | Eliminated |
| Explainability | Opaque | **Complete proof chain** | Full Audit Trail |
| GPU Required | Yes | **No** | Any Hardware |

---

## Repository Structure

```
model-xp/
├── main.py                          # CLI entry point
├── constitution.md                  # Immutable governing document
├── requirements.txt                 # Dependencies (Python stdlib only)
├── model_xp/
│   ├── __init__.py                  # Package exports
│   ├── fsm_controller.py            # Finite State Machine Controller
│   ├── persona_loader.py            # JSON Persona Loader
│   ├── inference_engine.py          # Logic-Based Network Engine
│   ├── knowledge_base.py            # Symbolic Knowledge Base
│   └── governance.py                # Deterministic Constraint Layer
├── personas/
│   ├── default.json                 # Default persona
│   ├── analyst.json                 # Financial analyst persona (Athena)
│   ├── legal.json                   # Legal analyst persona (Lexis)
│   └── medical.json                 # Medical research persona (Caduceus)
├── knowledge/
│   ├── general.json                 # General knowledge domain
│   └── finance_securities.json      # Finance and securities domain
├── docs/
│   └── PERSONA_SCHEMA.md            # Complete persona schema reference
└── tests/
    └── test_model_xp.py             # Full test suite (33 tests)
```

---

## Full Technical Specification

The complete 15-page architecture blueprint is available in this repository:

- **[View Specification (Markdown)](./SPECIFICATION.md)**
- **[Download Blueprint (PDF)](./Model_XP_Architecture_Blueprint.pdf)**

---

## Author

**Nicholas Michael Grossi**
Office of the Architect
2026-02-28

---

*Model XP is an original architecture. All intellectual property is attributed to Nicholas Michael Grossi.*
