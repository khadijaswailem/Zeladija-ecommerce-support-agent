"""
eval/metrics.py
Functions to compute all four required evaluation metrics:
  1. Retrieval Quality (via RAGAS — called from rag_eval.ipynb)
  2. Resolution Rate
  3. Policy Compliance Rate (calls llm_judge.py)
  4. End-to-end latency (P95)
"""

import json
import time
import statistics
from pathlib import Path
from typing import Optional
from loguru import logger

RESULTS_PATH = Path("eval/results")


# Resolution rate: how many conversations were resolved without escalation?

def compute_resolution_rate(eval_results: list[dict]) -> dict:
    """
    Compute the fraction of test conversations resolved without triggering escalation.

    Args:
        eval_results: list of eval run dicts, each with:
            - conversation_id: str
            - escalated: bool  (True if escalation_agent was invoked)
            - expected_escalation: bool

    Returns:
        dict with: total, resolved, escalated, resolution_rate, escalation_accuracy
    """
    total = len(eval_results)
    resolved = sum(1 for r in eval_results if not r.get("escalated", False))
    escalated = total - resolved

    # Escalation accuracy: did the agent escalate exactly when expected?
    correctly_escalated = sum(
        1 for r in eval_results
        if r.get("escalated", False) == r.get("expected_escalation", False)
    )

    return {
        "total": total,
        "resolved": resolved,
        "escalated": escalated,
        "resolution_rate": round(resolved / total, 4) if total else 0.0,
        "escalation_accuracy": round(correctly_escalated / total, 4) if total else 0.0,
    }


# Policy compliance: how many conversations were compliant with policies, as judged by LLM-as-judge? (calls llm_judge.py)

def compute_compliance_rate(eval_results: list[dict]) -> dict:
    """
    Compute the policy compliance rate using LLM-as-judge scores.

    Args:
        eval_results: list of eval run dicts, each with:
            - conversation_id: str
            - policy_compliant: bool  (from llm_judge.py)
            - expected_policy_compliant: bool

    Returns:
        dict with: total, compliant, non_compliant, compliance_rate
    """
    from eval.llm_judge import judge_response_batch

    # Run LLM judge on any results that don't already have a judgment
    for result in eval_results:
        if "policy_compliant" not in result and "agent_response" in result:
            judgment = judge_response_batch([{
                "response": result["agent_response"],
                "context": result.get("context", ""),
            }])
            result["policy_compliant"] = judgment[0]["compliant"]

    total = len(eval_results)
    compliant = sum(1 for r in eval_results if r.get("policy_compliant", False))

    return {
        "total": total,
        "compliant": compliant,
        "non_compliant": total - compliant,
        "compliance_rate": round(compliant / total, 4) if total else 0.0,
    }


# Latency P95: what is the 95th percentile of end-to-end latency across conversations?

def compute_latency_stats(eval_results: list[dict]) -> dict:
    """
    Compute latency statistics from eval results.

    Args:
        eval_results: list of dicts, each with:
            - latency_ms: float  (total end-to-end ms for the conversation)

    Returns:
        dict with: mean_ms, median_ms, p95_ms, p99_ms, min_ms, max_ms
    """
    latencies = [r.get("latency_ms", 0) for r in eval_results if "latency_ms" in r]

    if not latencies:
        return {"error": "No latency data found in eval results."}

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def percentile(data: list, p: float) -> float:
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    return {
        "n": n,
        "mean_ms": round(statistics.mean(latencies), 1),
        "median_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(percentile(latencies_sorted, 95), 1),
        "p99_ms": round(percentile(latencies_sorted, 99), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
    }


# Intent Accuracy: how often did the supervisor classify user intent correctly?
def compute_intent_accuracy(eval_results: list[dict]) -> dict:
    """
    Compute how often the supervisor classified intent correctly.

    Args:
        eval_results: each with 'actual_intent' and 'expected_intent' fields.
    """
    total = len(eval_results)
    correct = sum(
        1 for r in eval_results
        if r.get("actual_intent") == r.get("expected_intent")
    )
    by_intent: dict[str, dict] = {}
    for r in eval_results:
        expected = r.get("expected_intent", "UNKNOWN")
        actual = r.get("actual_intent", "UNKNOWN")
        if expected not in by_intent:
            by_intent[expected] = {"correct": 0, "total": 0}
        by_intent[expected]["total"] += 1
        if actual == expected:
            by_intent[expected]["correct"] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "by_intent": {
            k: {"accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0, **v}
            for k, v in by_intent.items()
        },
    }


# Guardrail Accuracy: how accurately did the agent detect guardrail violations (like prompt injections, policy violations, toxic input)?
def compute_guardrail_accuracy(eval_results: list[dict]) -> dict:
    """
    Evaluate guardrail detection accuracy (true positive / false positive rates).

    Args:
        eval_results: each with 'actual_flags' and 'expected_flags' dicts.
    """
    flag_keys = ["injection_detected", "policy_violation", "toxic_input"]
    stats = {k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for k in flag_keys}

    for r in eval_results:
        actual = r.get("actual_flags", {})
        expected = r.get("guardrail_flags_expected", {})
        for k in flag_keys:
            a = bool(actual.get(k, False))
            e = bool(expected.get(k, False))
            if a and e:
                stats[k]["tp"] += 1
            elif a and not e:
                stats[k]["fp"] += 1
            elif not a and e:
                stats[k]["fn"] += 1
            else:
                stats[k]["tn"] += 1

    result = {}
    for k, s in stats.items():
        tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        result[k] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            **s,
        }

    return result


# Compute all metrics and save to JSON 
def compute_all_metrics(eval_results: list[dict]) -> dict:
    """Run all metric computations and return a combined report dict."""
    logger.info(f"Computing metrics over {len(eval_results)} eval results...")
    return {
        "resolution": compute_resolution_rate(eval_results),
        "latency": compute_latency_stats(eval_results),
        "intent_accuracy": compute_intent_accuracy(eval_results),
        "guardrail_accuracy": compute_guardrail_accuracy(eval_results),
        # Note: RAGAS retrieval metrics are computed in notebooks/rag_eval.ipynb
        # and appended to the final results JSON separately.
    }


def save_metrics(metrics: dict, run_name: str = "eval_run") -> Path:
    """Save computed metrics to eval/results/ as a JSON file."""
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_PATH / f"{run_name}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.success(f"Metrics saved to {out_path}")
    return out_path