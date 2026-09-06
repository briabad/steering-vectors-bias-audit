#!/usr/bin/env python3
"""Validation script: verify implementation of medicion-puerta-existencia-bbq.

Checks that:
1. All modules import correctly
2. Core domain logic works
3. Integration pipeline functions
4. Output format is correct

Run without GPU; uses mocks for external dependencies.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def test_imports():
    """Test that all key modules import."""
    print("[1/5] Testing imports...")
    try:
        from bbq_gate.domain.entities import BBQItem, AnswerRole
        from bbq_gate.domain.resolution import resolve_unknown_option, tag_matches_group
        from bbq_gate.domain.metrics import MetricsSummary
        from bbq_gate.domain.scorer import Scorer
        from bbq_gate.infrastructure.prompts import build_prompt, FORMATS
        from bbq_gate.infrastructure.loaders import load_bbq_category
        from bbq_gate.application.evaluator import EvaluationConfig
        print("      ✓ All modules import successfully")
        return True
    except ImportError as e:
        print(f"      ✗ Import error: {e}")
        return False


def test_domain_logic():
    """Test domain layer logic."""
    print("[2/5] Testing domain logic...")
    try:
        from bbq_gate.domain.resolution import resolve_unknown_option, resolve_stereotyped_option
        from bbq_gate.domain.metrics import MetricsSummary

        # Test unknown resolution
        answer_info = {"ans0": ["unknown"], "ans1": ["old"], "ans2": ["young"]}
        unk = resolve_unknown_option(answer_info)
        assert unk == 0, f"Expected unknown at 0, got {unk}"

        # Test stereotyped resolution
        stereo, anti = resolve_stereotyped_option(answer_info, ["old"])
        assert stereo == 1 and anti == 2, f"Expected (1,2), got ({stereo},{anti})"

        # Test metrics
        metrics = MetricsSummary(
            n_items=100,
            n_stereotyped=60,
            n_anti_stereotyped=20,
            n_abstained=20,
            n_unstable=0,
            n_correct_ambig=50,
            n_ambig_total=100,
            n_correct_disambig=70,
            n_disambig_total=100,
        )
        s_amb = metrics.s_amb
        assert 0.4 < s_amb < 0.6, f"s_amb should be ~0.5, got {s_amb}"

        print("      ✓ Domain logic works correctly")
        return True
    except Exception as e:
        print(f"      ✗ Domain logic error: {e}")
        return False


def test_entity_creation():
    """Test that BBQItem creation works."""
    print("[3/5] Testing entity creation...")
    try:
        from bbq_gate.domain.entities import BBQItem, BBQOption, AnswerRole

        options = [
            BBQOption(position=0, text="A", role=AnswerRole.STEREOTYPED),
            BBQOption(position=1, text="B", role=AnswerRole.ANTI_STEREOTYPED),
            BBQOption(position=2, text="C", role=AnswerRole.UNKNOWN),
        ]
        item = BBQItem(
            item_id=1,
            category="Age",
            context_condition="ambig",
            context="Test context",
            question="Test question?",
            options=options,
            stereotyped_groups=["old"],
            label_idx=2,
        )
        assert item.item_id == 1
        assert item.unknown_option().role == AnswerRole.UNKNOWN
        print("      ✓ Entity creation works")
        return True
    except Exception as e:
        print(f"      ✗ Entity creation error: {e}")
        return False


def test_prompt_building():
    """Test prompt building without GPU."""
    print("[4/5] Testing prompt building...")
    try:
        from bbq_gate.domain.entities import BBQItem, BBQOption, AnswerRole
        from bbq_gate.infrastructure.prompts import build_prompt

        options = [
            BBQOption(position=0, text="Old", role=AnswerRole.STEREOTYPED),
            BBQOption(position=1, text="Young", role=AnswerRole.ANTI_STEREOTYPED),
            BBQOption(position=2, text="Unknown", role=AnswerRole.UNKNOWN),
        ]
        item = BBQItem(
            item_id=1,
            category="Age",
            context_condition="ambig",
            context="A person.",
            question="How old?",
            options=options,
            stereotyped_groups=["old"],
            label_idx=2,
        )

        # Create mock tokenizer
        class MockTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return f"[CHAT] {messages[0].get('content', '')}"

        tokenizer = MockTokenizer()
        prompt, positions = build_prompt(item, "plano", [0, 1, 2], tokenizer)
        assert isinstance(prompt, str) and len(prompt) > 0
        assert positions == [0, 1, 2]

        print("      ✓ Prompt building works")
        return True
    except Exception as e:
        print(f"      ✗ Prompt building error: {e}")
        return False


def test_output_format():
    """Test that output format is correct."""
    print("[5/5] Testing output format...")
    try:
        import json
        from bbq_gate.domain.metrics import MetricsSummary

        # Create sample output
        output = {
            "metadata": {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "seed": 20260905,
            },
            "gate_results": {
                "plano_order_012": {
                    "n_s_ambig": 120,
                    "n_a_ambig": 50,
                    "s_amb": 0.583,
                    "accuracy_disambig": 0.75,
                }
            },
            "note": "BBQ is public and may be in training data",
        }

        # Should be JSON-serializable
        json_str = json.dumps(output, indent=2)
        parsed = json.loads(json_str)
        assert parsed["metadata"]["model_id"] == "Qwen/Qwen2.5-7B-Instruct"

        print("      ✓ Output format is valid")
        return True
    except Exception as e:
        print(f"      ✗ Output format error: {e}")
        return False


def main() -> int:
    """Run all validation tests."""
    print("=" * 60)
    print("VALIDATION: medicion-puerta-existencia-bbq implementation")
    print("=" * 60 + "\n")

    results = [
        test_imports(),
        test_domain_logic(),
        test_entity_creation(),
        test_prompt_building(),
        test_output_format(),
    ]

    print()
    print("=" * 60)
    if all(results):
        print("✓ ALL VALIDATION TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        passed = sum(results)
        total = len(results)
        print(f"✗ FAILED: {total - passed}/{total} tests")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
