"""
eval/dashboard.py
Streamlit evaluation dashboard for the Zeladija AI Support Agent.
Loads eval/results/*.json and renders metric charts.

Run: streamlit run eval/dashboard.py
"""

import json
import os
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

RESULTS_PATH = Path("eval/results")
STORE_NAME = os.getenv("STORE_NAME", "Zeladija")

st.set_page_config(
    page_title=f"{STORE_NAME} AI Support — Eval Dashboard",
    page_icon="📊",
    layout="wide",
)


# Helpers

def load_results() -> dict:
    """Load the most recent eval results JSON files from eval/results/."""
    data = {}
    if not RESULTS_PATH.exists():
        return data
    for path in sorted(RESULTS_PATH.glob("*.json"), reverse=True):
        with open(path) as f:
            data[path.stem] = json.load(f)
    return data


def load_test_cases() -> list[dict]:
    """Load all test case JSONs from eval/test_cases/."""
    test_cases = []
    tc_path = Path("eval/test_cases")
    if tc_path.exists():
        for path in tc_path.rglob("*.json"):
            with open(path) as f:
                test_cases.append(json.load(f))
    return test_cases


# Page layout

st.title(f"📊 {STORE_NAME} AI Support Agent — Evaluation Dashboard")
st.caption("Tracks retrieval quality, resolution rate, policy compliance, and latency.")

all_results = load_results()
test_cases = load_test_cases()

if not all_results:
    st.warning(
        "No eval results found in `eval/results/`. "
        "Run the eval suite first: `python eval/run_eval.py`"
    )
    st.stop()

# Sidebar: select result file 
with st.sidebar:
    st.header("Settings")
    selected_run = st.selectbox("Select eval run:", list(all_results.keys()))
    st.caption(f"Found {len(all_results)} result file(s)")
    st.divider()
    st.metric("Test cases loaded", len(test_cases))

results = all_results[selected_run]

# KPI Row 
col1, col2, col3, col4 = st.columns(4)

resolution = results.get("resolution", {})
latency = results.get("latency", {})
intent = results.get("intent_accuracy", {})

with col1:
    rate = resolution.get("resolution_rate", 0)
    st.metric("Resolution Rate", f"{rate:.1%}", help="% of conversations resolved without escalation")

with col2:
    compliance = results.get("compliance_rate", results.get("policy_compliance_rate", None))
    if compliance is not None:
        st.metric("Policy Compliance", f"{compliance:.1%}", help="LLM-as-judge pass rate")
    else:
        st.metric("Policy Compliance", "N/A")

with col3:
    p95 = latency.get("p95_ms", None)
    st.metric("P95 Latency", f"{p95:.0f}ms" if p95 else "N/A", help="95th percentile end-to-end latency")

with col4:
    acc = intent.get("accuracy", None)
    st.metric("Intent Accuracy", f"{acc:.1%}" if acc is not None else "N/A", help="Supervisor classification accuracy")

st.divider()

# Resolution breakdown 
st.subheader("🎯 Resolution & Escalation")
col_a, col_b = st.columns(2)

with col_a:
    if resolution:
        fig = go.Figure(data=[go.Pie(
            labels=["Resolved", "Escalated"],
            values=[resolution.get("resolved", 0), resolution.get("escalated", 0)],
            hole=0.4,
            marker_colors=["#22c55e", "#f97316"],
        )])
        fig.update_layout(title="Resolution vs Escalation", height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

with col_b:
    if intent and "by_intent" in intent:
        by_intent = intent["by_intent"]
        df_intent = pd.DataFrame([
            {"Intent": k, "Accuracy": v["accuracy"], "Total": v["total"], "Correct": v["correct"]}
            for k, v in by_intent.items()
        ])
        fig = px.bar(df_intent, x="Intent", y="Accuracy", color="Accuracy",
                     color_continuous_scale="RdYlGn", range_color=[0, 1],
                     title="Intent Classification Accuracy by Category")
        fig.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# Guardrail accuracy 
st.subheader("🛡️ Guardrail Performance")
guardrail = results.get("guardrail_accuracy", {})

if guardrail:
    df_guard = pd.DataFrame([
        {
            "Guardrail": k.replace("_", " ").title(),
            "Precision": v.get("precision", 0),
            "Recall": v.get("recall", 0),
            "True Positives": v.get("tp", 0),
            "False Positives": v.get("fp", 0),
            "False Negatives": v.get("fn", 0),
        }
        for k, v in guardrail.items()
    ])
    st.dataframe(df_guard, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Precision", x=df_guard["Guardrail"], y=df_guard["Precision"], marker_color="#3b82f6"))
    fig.add_trace(go.Bar(name="Recall", x=df_guard["Guardrail"], y=df_guard["Recall"], marker_color="#8b5cf6"))
    fig.update_layout(barmode="group", title="Guardrail Precision & Recall", height=300, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Latency distribution 
st.subheader("⚡ Latency")
if latency and "mean_ms" in latency:
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("Mean", f"{latency['mean_ms']:.0f}ms")
    lc2.metric("Median", f"{latency['median_ms']:.0f}ms")
    lc3.metric("P95", f"{latency['p95_ms']:.0f}ms")
    lc4.metric("Max", f"{latency['max_ms']:.0f}ms")

st.divider()

# RAGAS scores (if present)
st.subheader("📚 RAG Retrieval Quality (RAGAS)")
ragas = results.get("ragas", {})
if ragas:
    col_r1, col_r2 = st.columns(2)
    baseline = ragas.get("baseline", {})
    advanced = ragas.get("advanced", {})
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    df_ragas = pd.DataFrame({
        "Metric": metrics,
        "Baseline (Naive)": [baseline.get(m, 0) for m in metrics],
        "Advanced (Hybrid)": [advanced.get(m, 0) for m in metrics],
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Naive RAG", x=df_ragas["Metric"], y=df_ragas["Baseline (Naive)"], marker_color="#94a3b8"))
    fig.add_trace(go.Bar(name="Advanced RAG", x=df_ragas["Metric"], y=df_ragas["Advanced (Hybrid)"], marker_color="#6366f1"))
    fig.update_layout(barmode="group", title="RAGAS: Naive vs Advanced RAG", height=350, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("RAGAS scores not yet computed. Run `notebooks/rag_eval.ipynb` and save results to `eval/results/`.")

# Raw results table 
with st.expander("Raw eval result JSON"):
    st.json(results)