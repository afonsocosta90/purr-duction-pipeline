"""
Streamlit demo application — "Am I a Cat?" MLOps portfolio showcase.

Provides a full closed-loop MLOps experience:
  1. Drag-and-drop image → live prediction via FastAPI backend
  2. Confidence gauge + prediction card
  3. Submit Feedback (correct label) → saved to monitoring/feedback_log.csv
  4. Inject Synthetic Drift → runs simulate_drift.py against the API
  5. Trigger Retraining → runs dvc repro --force, streams live output, shows
     before/after model metrics and promotion status

Environment variables:
  API_URL            FastAPI base URL (default: http://localhost:3000)
  PROJECT_ROOT       Absolute path to repo root for dvc/dvc commands
                     (default: parent of this file's directory)
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL: str = os.getenv("API_URL", "http://localhost:3000")
PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).parent.parent)))
FEEDBACK_LOG: Path = PROJECT_ROOT / "monitoring" / "feedback_log.csv"
INFERENCE_LOG: Path = PROJECT_ROOT / "monitoring" / "inference_log.csv"
SIMULATE_SCRIPT: Path = Path(__file__).parent / "simulate_drift.py"
METRICS_PATH: Path = PROJECT_ROOT / "monitoring" / "last_metrics.json"
POLL_INTERVAL_S: float = 0.3  # subprocess output polling cadence

LABEL_EMOJI: dict[str, str] = {"cat": "🐱", "not_cat": "🚫"}
LABEL_COLOR: dict[str, str] = {"cat": "#4CAF50", "not_cat": "#F44336"}


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Am I a Cat? — MLOps Demo",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global CSS for polished card styling
st.markdown(
    """
    <style>
    .prediction-card {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 16px;
    }
    .cat-card   { background: linear-gradient(135deg,#e8f5e9,#c8e6c9); }
    .notcat-card{ background: linear-gradient(135deg,#fce4ec,#f8bbd0); }
    .metric-box {
        border-radius: 8px;
        padding: 12px;
        background: #f8f9fa;
        border-left: 4px solid #1976D2;
        margin: 6px 0;
    }
    .log-box {
        background: #0d1117;
        color: #e6edf3;
        font-family: monospace;
        font-size: 12px;
        padding: 14px;
        border-radius: 8px;
        height: 300px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults: dict = {
        "last_prediction": None,  # dict: {label, confidence, is_cat}
        "last_image_bytes": None,  # bytes of the last uploaded image
        "feedback_submitted": False,
        "drift_injected": False,
        "drift_log": "",
        "retrain_log": "",
        "retrain_done": False,
        "before_metrics": None,  # dict captured before retraining
        "after_metrics": None,  # dict captured after retraining
        "prediction_history": [],  # list[dict] for the history table
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _api_health() -> tuple[bool, str]:
    """Return (is_healthy, status_message)."""
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3.0)
        if r.status_code == 200:
            return True, "API online ✅"
        return False, f"API returned {r.status_code}"
    except httpx.ConnectError:
        return False, "API unreachable ❌"
    except Exception as exc:
        return False, f"Error: {exc}"


def _predict(image_bytes: bytes, filename: str) -> dict:
    """POST image bytes to /predict, return response dict."""
    with httpx.Client(timeout=15.0) as client:
        response = client.post(
            f"{API_URL}/predict",
            files={"file": (filename, image_bytes, "image/jpeg")},
        )
    response.raise_for_status()
    return response.json()


def _save_feedback(
    image_filename: str, predicted_label: str, correct_label: str
) -> None:
    """Append one row to monitoring/feedback_log.csv."""
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FEEDBACK_LOG.exists()
    with FEEDBACK_LOG.open("a", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["timestamp", "filename", "predicted_label", "correct_label"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "filename": image_filename,
                "predicted_label": predicted_label,
                "correct_label": correct_label,
            }
        )


def _confidence_gauge(confidence: float, label: str) -> go.Figure:
    """Return a Plotly gauge chart for the model confidence score."""
    color = LABEL_COLOR.get(label, "#1976D2")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(confidence * 100, 1),
            title={"text": "Confidence", "font": {"size": 16}},
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 60], "color": "#ffebee"},
                    {"range": [60, 80], "color": "#fff8e1"},
                    {"range": [80, 100], "color": "#e8f5e9"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.75,
                    "value": 80,
                },
            },
        )
    )
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def _load_metrics_from_mlflow() -> dict | None:
    """
    Read the most-recent MLflow run metrics from mlruns/.
    Returns a flat dict of metric_name → value, or None if not found.
    """
    mlruns_dir = PROJECT_ROOT / "mlruns"
    if not mlruns_dir.exists():
        return None
    try:
        # Walk all experiment / run directories looking for val_accuracy metric
        best: dict | None = None
        best_ts: float = 0.0
        for metric_file in mlruns_dir.glob("*/*/metrics/val_accuracy"):
            lines = metric_file.read_text().strip().splitlines()
            if not lines:
                continue
            # MLflow metric file format: "timestamp value step"
            ts, val, _ = lines[-1].split()
            if float(ts) > best_ts:
                best_ts = float(ts)
                run_dir = metric_file.parent.parent
                metrics = {}
                for mf in (run_dir / "metrics").iterdir():
                    mlines = mf.read_text().strip().splitlines()
                    if mlines:
                        metrics[mf.name] = float(mlines[-1].split()[1])
                best = metrics
        return best
    except Exception:
        return None


def _run_subprocess_streaming(
    cmd: list[str],
    cwd: Path,
    log_placeholder,
) -> int:
    """
    Run a shell command, stream stdout+stderr into a Streamlit placeholder.
    Returns the exit code.
    """
    log_lines: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:  # type: ignore[union-attr]
        log_lines.append(line.rstrip())
        # Show last 60 lines to avoid the box growing indefinitely
        visible = "\n".join(log_lines[-60:])
        log_placeholder.markdown(
            f'<div class="log-box">{visible}</div>', unsafe_allow_html=True
        )
    proc.wait()
    return proc.returncode


def _feedback_stats() -> tuple[int, int]:
    """Return (total_feedback_rows, mislabelled_count)."""
    if not FEEDBACK_LOG.exists():
        return 0, 0
    df = pd.read_csv(FEEDBACK_LOG)
    mismatches = (df["predicted_label"] != df["correct_label"]).sum()
    return len(df), int(mismatches)


def _inference_log_stats() -> dict:
    """Return basic stats from inference_log.csv."""
    if not INFERENCE_LOG.exists():
        return {"rows": 0}
    df = pd.read_csv(INFERENCE_LOG)
    stats: dict = {"rows": len(df)}
    if "confidence" in df.columns and len(df):
        stats["avg_confidence"] = float(df["confidence"].mean())
        stats["min_confidence"] = float(df["confidence"].min())
    if "predicted_label" in df.columns:
        stats["label_counts"] = df["predicted_label"].value_counts().to_dict()
    return stats


# ---------------------------------------------------------------------------
# Sidebar — system status & settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f431.svg",
        width=60,
    )
    st.title("Am I a Cat?")
    st.caption("MLOps Portfolio Demo · Phase 9")
    st.divider()

    healthy, status_msg = _api_health()
    st.metric("Backend API", status_msg)
    st.caption(f"Endpoint: `{API_URL}`")

    st.divider()
    st.subheader("⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence threshold", min_value=0.5, max_value=1.0, value=0.80, step=0.01
    )
    st.caption("Predictions below this threshold are flagged as uncertain.")

    st.divider()
    st.subheader("📋 Session Stats")
    total_predictions = len(st.session_state.prediction_history)
    total_feedback, mismatches = _feedback_stats()
    st.metric("Predictions made", total_predictions)
    st.metric("Feedback submitted", total_feedback)
    st.metric("Model errors flagged", mismatches)

    st.divider()
    if st.button("🔄 Reset session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        _init_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Main content area — three tabs
# ---------------------------------------------------------------------------

tab_predict, tab_pipeline, tab_monitor = st.tabs(
    ["🔮 Live Prediction", "🔄 Pipeline Control", "📊 Monitoring"]
)


# ===========================================================================
# TAB 1 — Live Prediction
# ===========================================================================

with tab_predict:
    st.header("Live Image Prediction")
    st.caption(
        "Upload any image and the ResNet50 model will classify it in real time. "
        "Submit feedback if the prediction is wrong — your corrections feed the retraining loop."
    )

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        uploaded_file = st.file_uploader(
            "Drag & drop or click to upload",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            st.session_state.last_image_bytes = image_bytes
            st.session_state.feedback_submitted = False

            # Display uploaded image
            pil_img = Image.open(io.BytesIO(image_bytes))
            st.image(pil_img, caption=uploaded_file.name, use_container_width=True)

            if not healthy:
                st.error(
                    "API is offline. Start the API with `make serve` or `make demo`."
                )
            else:
                with st.spinner("Classifying image…"):
                    try:
                        result = _predict(image_bytes, uploaded_file.name)
                        st.session_state.last_prediction = result
                        # Append to history
                        st.session_state.prediction_history.append(
                            {
                                "time": datetime.utcnow().strftime("%H:%M:%S"),
                                "filename": uploaded_file.name,
                                "label": result["label"],
                                "confidence": result["confidence"],
                            }
                        )
                    except httpx.HTTPStatusError as exc:
                        st.error(
                            f"API error {exc.response.status_code}: {exc.response.text}"
                        )
                        st.session_state.last_prediction = None
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
                        st.session_state.last_prediction = None

    with col_result:
        pred = st.session_state.last_prediction
        if pred is None:
            st.info("Upload an image on the left to see the prediction here.")
        else:
            label: str = pred["label"]
            confidence: float = pred["confidence"]
            is_uncertain = confidence < confidence_threshold

            # Prediction card
            card_class = "cat-card" if label == "cat" else "notcat-card"
            emoji = LABEL_EMOJI.get(label, "❓")
            st.markdown(
                f"""
                <div class="prediction-card {card_class}">
                    <h1 style="font-size:3rem;margin:0">{emoji}</h1>
                    <h2 style="margin:8px 0">{label.replace("_", " ").title()}</h2>
                    {"<p style='color:#E65100'><b>⚠️ Low confidence — model is uncertain</b></p>" if is_uncertain else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence gauge
            st.plotly_chart(
                _confidence_gauge(confidence, label),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # Feedback section
            st.subheader("Submit Feedback")
            st.caption("Was this prediction correct? Help improve the model.")
            correct_label = st.radio(
                "Correct label",
                options=["cat", "not_cat"],
                index=0 if label == "cat" else 1,
                horizontal=True,
                key="correct_label_radio",
            )
            agree_with_prediction = correct_label == label

            if st.session_state.feedback_submitted:
                if agree_with_prediction:
                    st.success("✅ Confirmed correct! Thanks.")
                else:
                    st.warning(
                        f"📝 Feedback recorded: predicted **{label}**, correct is **{correct_label}**."
                    )
            else:
                btn_label = (
                    "✅ Confirm correct prediction"
                    if agree_with_prediction
                    else "📝 Submit correction"
                )
                if st.button(btn_label, use_container_width=True):
                    filename = (
                        st.session_state.get("last_uploaded_name", "unknown.jpg")
                        if uploaded_file is None
                        else uploaded_file.name
                    )
                    _save_feedback(filename, label, correct_label)
                    st.session_state.feedback_submitted = True
                    st.rerun()

    # Prediction history table
    if st.session_state.prediction_history:
        st.divider()
        st.subheader("Prediction History (this session)")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df["confidence"] = history_df["confidence"].map("{:.1%}".format)
        st.dataframe(history_df, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 2 — Pipeline Control
# ===========================================================================

with tab_pipeline:
    st.header("Closed-Loop Retraining Pipeline")
    st.caption(
        "Simulate real-world model degradation by injecting synthetic drift, "
        "then trigger a full DVC retraining pipeline and observe promotion status."
    )

    # -----------------------------------------------------------------------
    # Step 1 — Inject Synthetic Drift
    # -----------------------------------------------------------------------
    st.subheader("Step 1 — Inject Synthetic Drift")
    col_drift_info, col_drift_btn = st.columns([3, 1])
    with col_drift_info:
        st.markdown("""
            Sends **out-of-distribution images** (random noise, solid colours, blur) to the
            `/predict` API endpoint, simulating production traffic that diverges from the
            training distribution. The inference log (`monitoring/inference_log.csv`) is also
            populated with synthetic entries for Evidently AI drift analysis.
            """)
        drift_count = st.number_input(
            "Number of synthetic images", min_value=10, max_value=500, value=50, step=10
        )

    with col_drift_btn:
        st.write("")
        st.write("")
        inject_btn = st.button(
            "💉 Inject Drift",
            use_container_width=True,
            type="primary" if not st.session_state.drift_injected else "secondary",
        )

    drift_log_placeholder = st.empty()
    if st.session_state.drift_log:
        drift_log_placeholder.markdown(
            f'<div class="log-box">{st.session_state.drift_log}</div>',
            unsafe_allow_html=True,
        )

    if inject_btn:
        if not healthy:
            st.error(
                "API must be running to inject drift. Start with `make serve` or `make demo`."
            )
        else:
            st.session_state.drift_log = ""
            cmd = [
                sys.executable,
                str(SIMULATE_SCRIPT),
                "--count",
                str(int(drift_count)),
                "--api-url",
                API_URL,
                "--output-log",
                str(INFERENCE_LOG),
            ]
            with st.spinner(f"Injecting {drift_count} synthetic images…"):
                rc = _run_subprocess_streaming(cmd, PROJECT_ROOT, drift_log_placeholder)

            if rc == 0:
                st.session_state.drift_injected = True
                st.success(
                    f"✅ Drift injection complete — {drift_count} synthetic requests sent."
                )
            else:
                st.error(
                    f"Drift injection failed (exit code {rc}). Check the log above."
                )

    # -----------------------------------------------------------------------
    # Step 2 — Trigger Retraining
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Step 2 — Trigger Retraining")

    if not st.session_state.drift_injected:
        st.info(
            "💡 Inject synthetic drift first (Step 1) to simulate a degraded model scenario."
        )

    col_retrain_info, col_retrain_btn = st.columns([3, 1])
    with col_retrain_info:
        st.markdown("""
            Runs `dvc repro --force` against the full pipeline:
            **ingest → validate → features → train → evaluate**.
            Model promotion requires `val_accuracy ≥ 0.94` **and** `val_f1 ≥ 0.93`.
            Live output is streamed below.
            """)
    with col_retrain_btn:
        st.write("")
        st.write("")
        retrain_btn = st.button(
            "🚀 Retrain Model",
            use_container_width=True,
            type="primary",
            disabled=not PROJECT_ROOT.exists(),
        )

    retrain_log_placeholder = st.empty()
    if st.session_state.retrain_log:
        retrain_log_placeholder.markdown(
            f'<div class="log-box">{st.session_state.retrain_log}</div>',
            unsafe_allow_html=True,
        )

    if retrain_btn:
        # Capture before-metrics
        st.session_state.before_metrics = _load_metrics_from_mlflow()
        st.session_state.retrain_log = ""
        st.session_state.retrain_done = False

        cmd = ["poetry", "run", "dvc", "repro", "--force"]
        with st.spinner("Running DVC pipeline… this may take several minutes."):
            rc = _run_subprocess_streaming(cmd, PROJECT_ROOT, retrain_log_placeholder)

        if rc == 0:
            time.sleep(1)  # brief pause for MLflow to flush
            st.session_state.after_metrics = _load_metrics_from_mlflow()
            st.session_state.retrain_done = True
            st.success("✅ Retraining pipeline completed successfully.")
        else:
            st.error(f"Pipeline failed (exit code {rc}). Check the log above.")

    # -----------------------------------------------------------------------
    # Step 3 — Before / After Metrics Comparison
    # -----------------------------------------------------------------------
    if st.session_state.retrain_done:
        st.divider()
        st.subheader("Step 3 — Model Metrics: Before vs After")

        before = st.session_state.before_metrics or {}
        after = st.session_state.after_metrics or {}

        key_metrics = [
            "val_accuracy",
            "val_f1",
            "val_precision",
            "val_recall",
            "val_roc_auc",
        ]
        thresholds = {"val_accuracy": 0.94, "val_f1": 0.93}

        col_b, col_a = st.columns(2)
        with col_b:
            st.markdown("**Before retraining**")
            for k in key_metrics:
                v = before.get(k)
                disp = f"{v:.4f}" if v is not None else "—"
                st.markdown(
                    f'<div class="metric-box">{k}: <b>{disp}</b></div>',
                    unsafe_allow_html=True,
                )

        with col_a:
            st.markdown("**After retraining**")
            for k in key_metrics:
                v_after = after.get(k)
                v_before = before.get(k)
                disp = f"{v_after:.4f}" if v_after is not None else "—"
                delta = ""
                if v_after is not None and v_before is not None:
                    diff = v_after - v_before
                    sign = "+" if diff >= 0 else ""
                    delta = f" ({sign}{diff:.4f})"
                threshold = thresholds.get(k)
                promoted_str = ""
                if v_after is not None and threshold is not None:
                    promoted_str = " ✅" if v_after >= threshold else " ❌"
                st.markdown(
                    f'<div class="metric-box">{k}: <b>{disp}{delta}</b>{promoted_str}</div>',
                    unsafe_allow_html=True,
                )

        # Promotion verdict
        promoted = (
            (after.get("val_accuracy", 0) >= 0.94 and after.get("val_f1", 0) >= 0.93)
            if after
            else False
        )
        if after:
            if promoted:
                st.success(
                    "🏆 **Model PROMOTED to staging** — accuracy ≥ 0.94 AND F1 ≥ 0.93"
                )
            else:
                st.error(
                    "⛔ Model **did not pass** promotion gates — check metrics above"
                )


# ===========================================================================
# TAB 3 — Monitoring
# ===========================================================================

with tab_monitor:
    st.header("Live Monitoring Dashboard")
    st.caption("Real-time metrics from the inference log and Prometheus/Grafana stack.")

    col_refresh, col_grafana_link = st.columns([1, 3])
    with col_refresh:
        if st.button("🔄 Refresh metrics"):
            st.rerun()

    grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3001")
    with col_grafana_link:
        st.markdown(
            f"📊 [Open full Grafana dashboard ↗]({grafana_url}) &nbsp;&nbsp; "
            f"🔥 [Open Prometheus ↗](http://localhost:9090)"
        )

    st.divider()

    # --- Inference log stats ---
    stats = _inference_log_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total inferences logged", stats["rows"])
    col2.metric(
        "Avg confidence",
        f"{stats.get('avg_confidence', 0):.1%}" if stats["rows"] else "—",
        delta=(
            "⚠️ below 0.80 threshold"
            if stats.get("avg_confidence", 1.0) < 0.80 and stats["rows"] > 0
            else None
        ),
    )
    col3.metric(
        "Min confidence",
        f"{stats.get('min_confidence', 0):.1%}" if stats["rows"] else "—",
    )

    # --- Label distribution from inference log ---
    if stats["rows"] > 0 and "label_counts" in stats:
        st.subheader("Label Distribution (inference log)")
        label_df = pd.DataFrame(
            stats["label_counts"].items(), columns=["Label", "Count"]
        )
        fig_bar = go.Figure(
            go.Bar(
                x=label_df["Label"],
                y=label_df["Count"],
                marker_color=[
                    LABEL_COLOR.get(lbl, "#1976D2") for lbl in label_df["Label"]
                ],
                text=label_df["Count"],
                textposition="outside",
            )
        )
        fig_bar.update_layout(
            height=280,
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False,
        )
        st.plotly_chart(
            fig_bar, use_container_width=True, config={"displayModeBar": False}
        )

    # --- Recent inference log rows ---
    st.subheader("Recent Inference Log Entries")
    if INFERENCE_LOG.exists() and stats["rows"] > 0:
        df_log = pd.read_csv(INFERENCE_LOG).tail(20)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No inference log yet. Upload images in **Live Prediction** tab or "
            "run **Inject Drift** to populate the log."
        )

    # --- Feedback log ---
    st.subheader("Feedback Log")
    total_fb, mismatch_fb = _feedback_stats()
    if total_fb:
        df_fb = pd.read_csv(FEEDBACK_LOG).tail(10)
        st.dataframe(df_fb, use_container_width=True, hide_index=True)
        st.caption(
            f"Total feedback: {total_fb} entries · {mismatch_fb} model errors flagged "
            f"({mismatch_fb/total_fb:.0%} error rate)"
        )
    else:
        st.info(
            "No feedback submitted yet. Use the **Live Prediction** tab to submit corrections."
        )

    # --- Grafana embed (best-effort) ---
    st.divider()
    st.subheader("Grafana Dashboard")
    st.markdown(f"""
        The Grafana dashboard at **{grafana_url}** shows live Prometheus metrics:
        - `catops_predictions_total` — prediction counter by label
        - `catops_prediction_confidence` — confidence score histogram
        - HTTP request rate and latency

        Open it at [{grafana_url}]({grafana_url}) (login: **admin / catops**).

        > 💡 Start the full monitoring stack with `make demo` — this launches
        > FastAPI + Streamlit + Prometheus + Grafana in one command.
        """)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "**Am I a Cat?** — End-to-end MLOps portfolio project · "
    "Poetry · DVC · Hydra · PyTorch ResNet50 · MLflow · FastAPI · "
    "Evidently AI · Prometheus · Grafana · GitHub Actions"
)
