#!/usr/bin/env python3
"""
Model XP — Command Line Interface
Authored by Nicholas Michael Grossi

Run Model XP interactively from the command line.

Usage:
    python3 main.py
    python3 main.py --persona analyst
    python3 main.py --query "what is your name"
"""

import argparse
import json
import os
import sys

# Ensure the package is importable from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_xp import FSMController

PERSONAS_DIR = os.path.join(os.path.dirname(__file__), "personas")
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
CONSTITUTION_PATH = os.path.join(os.path.dirname(__file__), "constitution.md")

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║                         MODEL XP  v1.0.0                        ║
║      An Architecture for Deterministic Artificial Intelligence   ║
║                  Authored by Nicholas Michael Grossi             ║
╠══════════════════════════════════════════════════════════════════╣
║  Type /help for commands.  Type /personas to list personas.      ║
║  Type /persona <id> to switch personas.  Type 'exit' to quit.   ║
╚══════════════════════════════════════════════════════════════════╝
"""


def format_response(result: dict, show_proof: bool = False, show_audit: bool = False) -> str:
    """Format a Model XP response for display."""
    lines = []
    state = result.get("state", "UNKNOWN")
    persona = result.get("persona", "UNKNOWN")

    lines.append(f"\n[{state}] Persona: {persona}")
    lines.append("─" * 60)

    if state == "HALT":
        lines.append(f"HALT: {result.get('halt_reason', 'Unknown halt reason.')}")
    else:
        answer = result.get("answer", "")
        lines.append(answer)

    if show_proof and result.get("proof"):
        lines.append("\n── Proof Chain ──")
        for step in result["proof"]:
            lines.append(f"  {step}")

    if show_audit and result.get("audit_log"):
        lines.append("\n── Audit Log ──")
        for entry in result["audit_log"]:
            lines.append(f"  [{entry['check']}] {entry['result']}: {entry['detail']}")

    lines.append("─" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Model XP — Deterministic AI Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--persona", "-p",
        default="default",
        help="Persona ID to load on startup (default: 'default')"
    )
    parser.add_argument(
        "--query", "-q",
        default=None,
        help="Run a single query and exit (non-interactive mode)"
    )
    parser.add_argument(
        "--proof",
        action="store_true",
        help="Display the logical proof chain for each response"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Display the governance audit log for each response"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output raw JSON responses"
    )
    args = parser.parse_args()

    # Initialize the FSM Controller
    try:
        controller = FSMController(
            personas_dir=PERSONAS_DIR,
            knowledge_dir=KNOWLEDGE_DIR,
            constitution_path=CONSTITUTION_PATH,
            default_persona_id=args.persona
        )
    except Exception as e:
        print(f"FATAL: Model XP failed to initialize: {e}", file=sys.stderr)
        sys.exit(1)

    # Non-interactive single query mode
    if args.query:
        result = controller.process(args.query)
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print(format_response(result, show_proof=args.proof, show_audit=args.audit))
        sys.exit(0 if result["state"] == "OUTPUT" else 1)

    # Interactive mode
    print(BANNER)
    active = controller.get_active_persona()
    if active:
        print(f"Active Persona: {active.name} ({active.archetype})")
        print(f"Persona ID:     {active.persona_id}")
        print(f"Domains:        {', '.join(active.domains) or 'General'}\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nModel XP session terminated.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Model XP session terminated.")
            break

        result = controller.process(user_input)

        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print(format_response(result, show_proof=args.proof, show_audit=args.audit))


if __name__ == "__main__":
    main()
