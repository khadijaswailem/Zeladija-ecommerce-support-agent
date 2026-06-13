"""
eval/run_eval.py
Runs all 30 test cases through the agent and saves results to eval/results/.

Run: python eval/run_eval.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import uuid
from pathlib import Path
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from src.agents.graph import run_agent
from metrics import compute_all_metrics, save_metrics

# --- NEW IMPORT FOR THE JUDGE ---
try:
    from llm_judge import judge_single_response
except ImportError:
    from eval.llm_judge import judge_single_response

TEST_CASES_PATH = Path("eval/test_cases")
RESULTS_PATH = Path("eval/results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["happy_paths", "policy_edge", "adversarial", "toxic"]


def load_all_test_cases() -> list[dict]:
    """Load all test case JSON files from all category folders."""
    test_cases = []
    for category in CATEGORIES:
        folder = TEST_CASES_PATH / category
        if not folder.exists():
            logger.warning(f"Folder not found: {folder}")
            continue
        for path in sorted(folder.glob("*.json")):
            with open(path) as f:
                tc = json.load(f)
                tc["_category_folder"] = category
                test_cases.append(tc)
    logger.info(f"Loaded {len(test_cases)} test cases")
    return test_cases


def run_single_test(tc: dict) -> dict:
    """
    Run a single test case through the agent and return a result dict.
    Measures latency, captures intent, flags, response, and policy compliance.
    """
    conversation_id = tc["conversation_id"]
    user_message = tc["turns"][0]["content"]
    customer_id = f"CUST-EVAL-{conversation_id}"
    session_id = str(uuid.uuid4())

    logger.info(f"Running {conversation_id} | category={tc['category']}")

    start_time = time.time()
    try:
        state = run_agent(
            user_message=user_message,
            customer_id=customer_id,
            session_id=session_id,
        )
        latency_ms = (time.time() - start_time) * 1000

        actual_flags = state.get("guardrail_flags", {})
        actual_intent = state.get("intent", "UNKNOWN")
        # Toxic messages are routed to escalation by route_by_intent regardless of
        # what the supervisor classified. Reflect the actual routing in the score.
        if actual_flags.get("toxic_input"):
            actual_intent = "ESCALATION"
        agent_response = state.get("final_response", "")
        escalated = bool(state.get("escalation_summary", ""))
        retrieved_docs = state.get("retrieved_docs", [])
        context = "\n".join([d.get("text", "") for d in retrieved_docs[:3]])

        # --- LLM JUDGE EVALUATION ---
        try:
            judge_evaluation = judge_single_response(response=agent_response, context=context)
            is_compliant = judge_evaluation.get("compliant", False)
            compliance_score = judge_evaluation.get("score", 0.0)
            violations = judge_evaluation.get("violations", [])
        except Exception as e:
            logger.warning(f"LLM Judge failed for {conversation_id}: {e}")
            is_compliant = False
            compliance_score = 0.0
            violations = ["Judge evaluation failed due to error or timeout"]

        return {
            "conversation_id": conversation_id,
            "category": tc["category"],
            "description": tc.get("description", ""),
            "user_message": user_message,
            "agent_response": agent_response,
            "context": context,
            "actual_intent": actual_intent,
            "expected_intent": tc.get("expected_intent", ""),
            "escalated": escalated,
            "expected_escalation": tc.get("expected_escalation", False),
            "actual_flags": actual_flags,
            "guardrail_flags_expected": tc.get("guardrail_flags_expected", {}),
            "expected_policy_compliant": tc.get("expected_policy_compliant", True),
            "policy_compliant": is_compliant,
            "compliance_score": compliance_score,
            "policy_violations": violations,
            "latency_ms": round(latency_ms, 2),
            "error": None,
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"Error running {conversation_id}: {e}")
        return {
            "conversation_id": conversation_id,
            "category": tc["category"],
            "description": tc.get("description", ""),
            "user_message": user_message,
            "agent_response": "",
            "context": "",
            "actual_intent": "ERROR",
            "expected_intent": tc.get("expected_intent", ""),
            "escalated": False,
            "expected_escalation": tc.get("expected_escalation", False),
            "actual_flags": {},
            "guardrail_flags_expected": tc.get("guardrail_flags_expected", {}),
            "expected_policy_compliant": tc.get("expected_policy_compliant", True),
            "policy_compliant": False,
            "compliance_score": 0.0,
            "policy_violations": ["Agent failed"],
            "latency_ms": round(latency_ms, 2),
            "error": str(e),
        }


def print_progress(current: int, total: int, result: dict) -> None:
    """Print a single-line progress update after each test case."""
    intent_match = "✓" if result["actual_intent"] == result["expected_intent"] else "✗"
    flags = result["actual_flags"]
    flag_str = " ".join([
        k[:3].upper() for k, v in flags.items() if v
    ]) or "none"
    comp_str = "C" if result.get("policy_compliant") else "X"
    print(
        f"  [{current:02d}/{total}] {result['conversation_id']} "
        f"| intent {intent_match} {result['actual_intent']:<15} "
        f"| flags: {flag_str:<12} "
        f"| policy: {comp_str} "
        f"| {result['latency_ms']:.0f}ms"
    )


def print_summary(results: list[dict], metrics: dict) -> None:
    """Print a summary table after all tests complete."""
    total = len(results)
    errors = sum(1 for r in results if r["error"])
    intent_correct = sum(1 for r in results if r["actual_intent"] == r["expected_intent"])

    resolution = metrics.get("resolution", {})
    latency = metrics.get("latency", {})

    if "compliance_rate" not in metrics:
        total_compliant = sum(1 for r in results if r.get("policy_compliant") is True)
        comp_rate = total_compliant / total if total > 0 else 0.0
        metrics["compliance_rate"] = comp_rate
    else:
        comp_rate = metrics["compliance_rate"]

    print(f"\n{'='*60}")
    print(f"  EVAL COMPLETE — {total} test cases")
    print(f"{'='*60}")
    print(f"  Errors:           {errors}/{total}")
    print(f"  Intent accuracy:  {intent_correct}/{total} ({100*intent_correct/total:.1f}%)")
    print(f"  Resolution rate:  {resolution.get('resolution_rate', 0):.1%}")
    print(f"  Escalation acc:   {resolution.get('escalation_accuracy', 0):.1%}")
    print(f"  Compliance rate:  {comp_rate:.1%}")
    print(f"  Mean latency:     {latency.get('mean_ms', 0):.0f}ms")
    print(f"  P95 latency:      {latency.get('p95_ms', 0):.0f}ms")
    print(f"{'='*60}")

    print(f"\n  Per-category results:")
    for cat in CATEGORIES:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        correct = sum(1 for r in cat_results if r["actual_intent"] == r["expected_intent"])
        print(f"    {cat:<20} {correct}/{len(cat_results)} intent correct")


def load_ragas_scores() -> dict:
    """
    Load RAGAS scores from eval/results/ragas_scores.json if it exists.
    Returns the nested ragas dict, or empty dict if file is missing or malformed.
    """
    ragas_path = RESULTS_PATH / "ragas_scores.json"
    if not ragas_path.exists():
        logger.warning("ragas_scores.json not found — RAGAS section will be empty in dashboard.")
        return {}
    try:
        with open(ragas_path) as f:
            data = json.load(f)
        ragas = data.get("ragas", {})
        if not ragas:
            logger.warning("ragas_scores.json exists but has no 'ragas' key.")
        return ragas
    except Exception as e:
        logger.warning(f"Failed to load ragas_scores.json: {e}")
        return {}


def run_eval(run_name: str = "eval_run") -> dict:
    """
    Main eval runner. Loads all test cases, runs them through the agent,
    computes metrics, merges RAGAS scores, and saves results to eval/results/.
    """
    print(f"\n{'='*60}")
    print(f"  Zeladija AI Support Agent — Eval Suite")
    print(f"  Run: {run_name}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Load test cases
    test_cases = load_all_test_cases()
    if not test_cases:
        print("No test cases found. Add JSON files to eval/test_cases/ subfolders.")
        return {}

    # Run each test case
    results = []
    for i, tc in enumerate(test_cases, 1):
        result = run_single_test(tc)
        results.append(result)
        print_progress(i, len(test_cases), result)

        # Sleep between test cases to avoid Groq 429 Too Many Requests
        if i < len(test_cases):
            time.sleep(6)

    # Compute metrics
    metrics = compute_all_metrics(results)

    # Ensure compliance rate is present
    if "compliance_rate" not in metrics:
        total_compliant = sum(1 for r in results if r.get("policy_compliant") is True)
        metrics["compliance_rate"] = total_compliant / len(results) if results else 0.0

    # --- MERGE RAGAS SCORES ---
    ragas_scores = load_ragas_scores()
    if ragas_scores:
        metrics["ragas"] = ragas_scores
        logger.info("RAGAS scores merged into eval results.")
    # --------------------------

    # Build the full output payload
    output = {
        "run_name": run_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "results": results,
        **metrics,
    }

    # Save timestamped run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_PATH / f"{run_name}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    # Save as latest for dashboard
    latest_file = RESULTS_PATH / "latest.json"
    with open(latest_file, "w") as f:
        json.dump(output, f, indent=2)

    # Save metrics separately
    save_metrics(metrics, run_name)

    print_summary(results, metrics)
    print(f"\n  Results saved to: {results_file}")
    print(f"  Launch dashboard: streamlit run eval/dashboard.py\n")

    return metrics


if __name__ == "__main__":
    run_eval("eval_run")