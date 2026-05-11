"""
Streamlit demo — "Am I a Cat?" MLOps Portfolio Showcase  ·  Phase 9
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A fully closed-loop MLOps demonstration:
  • Live image classification via ResNet50 / FastAPI
  • Confidence gauge + human feedback collection
  • Synthetic drift injection + live DVC retraining pipeline
  • Prometheus / Grafana monitoring integration

Environment variables:
  API_URL        FastAPI base URL             (default: http://localhost:3000)
  PROJECT_ROOT   Repo root for dvc commands   (default: parent of this file)
  GRAFANA_URL    Grafana dashboard URL        (default: http://localhost:3001)
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

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_URL: str = os.getenv("API_URL", "http://localhost:3000")
PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).parent.parent)))
GRAFANA_URL: str = os.getenv("GRAFANA_URL", "http://localhost:3001")
FEEDBACK_LOG: Path = PROJECT_ROOT / "monitoring" / "feedback_log.csv"
INFERENCE_LOG: Path = PROJECT_ROOT / "monitoring" / "inference_log.csv"
SIMULATE_SCRIPT: Path = Path(__file__).parent / "simulate_drift.py"

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────

TEAL = "#06d6a0"
TEAL_DIM = "#059669"
NAVY = "#0a0e1a"
SURFACE = "#111827"
SURFACE_2 = "#1a2235"
BORDER = "#1e3050"
TEXT_PRIMARY = "#f1f5f9"
TEXT_SECONDARY = "#94a3b8"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER = "#ef4444"
CAT_GRADIENT = "linear-gradient(135deg, #064e3b 0%, #065f46 50%, #0d9488 100%)"
NOT_CAT_GRADIENT = "linear-gradient(135deg, #7f1d1d 0%, #991b1b 50%, #dc2626 100%)"

# ─────────────────────────────────────────────────────────────────────────────
# Page setup (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Am I a Cat? · MLOps Demo",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — design system
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <style>
    /* ── Base resets ────────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {NAVY};
        color: {TEXT_PRIMARY};
    }}
    [data-testid="stSidebar"] {{
        background: {SURFACE} !important;
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.5rem;
    }}

    /* ── Typography ─────────────────────────────────────────────────── */
    h1, h2, h3, h4 {{ color: {TEXT_PRIMARY} !important; font-weight: 700; }}
    p, li, span {{ color: {TEXT_SECONDARY}; }}
    .stMarkdown p {{ color: {TEXT_SECONDARY}; }}

    /* ── Tab styling ────────────────────────────────────────────────── */
    [data-testid="stTabs"] [role="tablist"] {{
        border-bottom: 2px solid {BORDER};
        gap: 0.25rem;
    }}
    [data-testid="stTabs"] [role="tab"] {{
        color: {TEXT_SECONDARY};
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.4rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        color: {TEAL};
        border-bottom: 2px solid {TEAL};
        background: rgba(6,214,160,0.07);
    }}
    [data-testid="stTabs"] [role="tab"]:hover {{
        color: {TEAL};
        background: rgba(6,214,160,0.05);
    }}

    /* ── Cards ──────────────────────────────────────────────────────── */
    .card {{
        background: {SURFACE_2};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
        transition: box-shadow 0.2s ease;
    }}
    .card:hover {{ box-shadow: 0 6px 32px rgba(0,0,0,0.5); }}

    .card-cat {{
        background: {CAT_GRADIENT};
        border: 1px solid {TEAL_DIM};
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(6,214,160,0.2);
        margin-bottom: 1rem;
    }}
    .card-notcat {{
        background: {NOT_CAT_GRADIENT};
        border: 1px solid #991b1b;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(239,68,68,0.2);
        margin-bottom: 1rem;
    }}

    /* ── Metric chips ───────────────────────────────────────────────── */
    .chip {{
        display: inline-block;
        background: {SURFACE_2};
        border: 1px solid {BORDER};
        border-radius: 999px;
        padding: 0.3rem 0.85rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: {TEXT_PRIMARY};
        margin: 0.15rem;
    }}
    .chip-teal  {{ border-color: {TEAL}; color: {TEAL}; background: rgba(6,214,160,0.08); }}
    .chip-green {{ border-color: {SUCCESS}; color: {SUCCESS}; background: rgba(34,197,94,0.08); }}
    .chip-red   {{ border-color: {DANGER}; color: {DANGER}; background: rgba(239,68,68,0.08); }}
    .chip-amber {{ border-color: {WARNING}; color: {WARNING}; background: rgba(245,158,11,0.08); }}

    /* ── Health pill ────────────────────────────────────────────────── */
    .health-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border-radius: 999px;
        padding: 0.3rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }}
    .health-up   {{ background: rgba(34,197,94,0.12); border: 1px solid {SUCCESS}; color: {SUCCESS}; }}
    .health-down {{ background: rgba(239,68,68,0.12); border: 1px solid {DANGER};  color: {DANGER}; }}
    .health-dot  {{ width: 7px; height: 7px; border-radius: 50%; }}
    .health-up   .health-dot {{ background: {SUCCESS}; box-shadow: 0 0 6px {SUCCESS}; animation: pulse 2s infinite; }}
    .health-down .health-dot {{ background: {DANGER}; }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50%       {{ opacity: 0.4; }}
    }}

    /* ── Terminal log box ───────────────────────────────────────────── */
    .log-box {{
        background: #060a0f;
        border: 1px solid {BORDER};
        border-radius: 10px;
        color: #a8ff78;
        font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", monospace;
        font-size: 0.75rem;
        line-height: 1.6;
        padding: 1rem 1.2rem;
        height: 320px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-all;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.5);
    }}
    .log-box::-webkit-scrollbar       {{ width: 4px; }}
    .log-box::-webkit-scrollbar-track {{ background: transparent; }}
    .log-box::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}

    /* ── Step wizard ────────────────────────────────────────────────── */
    .step-header {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }}
    .step-badge {{
        width: 32px; height: 32px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.85rem;
        flex-shrink: 0;
    }}
    .step-active   {{ background: {TEAL}; color: #000; box-shadow: 0 0 12px rgba(6,214,160,0.5); }}
    .step-done     {{ background: {SUCCESS}; color: #000; }}
    .step-locked   {{ background: {SURFACE_2}; color: {TEXT_SECONDARY}; border: 1px solid {BORDER}; }}
    .step-title    {{ font-size: 1.05rem; font-weight: 600; color: {TEXT_PRIMARY}; }}
    .step-subtitle {{ font-size: 0.82rem; color: {TEXT_SECONDARY}; }}

    /* ── Metric comparison row ──────────────────────────────────────── */
    .metric-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0.9rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid {BORDER};
    }}
    .metric-label {{ font-size: 0.82rem; color: {TEXT_SECONDARY}; }}
    .metric-value {{ font-size: 0.95rem; font-weight: 600; color: {TEXT_PRIMARY}; font-family: monospace; }}
    .delta-up   {{ color: {SUCCESS}; font-size: 0.8rem; }}
    .delta-down {{ color: {DANGER};  font-size: 0.8rem; }}
    .delta-flat {{ color: {TEXT_SECONDARY}; font-size: 0.8rem; }}

    /* ── Promotion banners ──────────────────────────────────────────── */
    .banner-promoted {{
        background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(6,214,160,0.08));
        border: 1px solid {SUCCESS};
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        text-align: center;
        margin-top: 1rem;
    }}
    .banner-failed {{
        background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(245,158,11,0.06));
        border: 1px solid {DANGER};
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        text-align: center;
        margin-top: 1rem;
    }}

    /* ── Upload area ────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {{
        border-radius: 12px;
    }}
    [data-testid="stFileUploader"] > div {{
        border: 2px dashed {BORDER};
        border-radius: 12px;
        background: {SURFACE_2};
        transition: border-color 0.2s;
    }}
    [data-testid="stFileUploader"] > div:hover {{
        border-color: {TEAL};
    }}

    /* ── Low confidence warning ─────────────────────────────────────── */
    .warning-banner {{
        background: rgba(245,158,11,0.1);
        border: 1px solid {WARNING};
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: {WARNING};
        font-size: 0.88rem;
        font-weight: 500;
        margin: 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* ── Sidebar stat cards ─────────────────────────────────────────── */
    .sidebar-stat {{
        background: rgba(255,255,255,0.04);
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
    }}
    .sidebar-stat-label {{ font-size: 0.72rem; color: {TEXT_SECONDARY}; text-transform: uppercase; letter-spacing: 0.05em; }}
    .sidebar-stat-value {{ font-size: 1.3rem; font-weight: 700; color: {TEXT_PRIMARY}; margin-top: 0.1rem; }}

    /* ── Divider ────────────────────────────────────────────────────── */
    hr {{ border-color: {BORDER} !important; margin: 1rem 0; }}

    /* ── Streamlit overrides ────────────────────────────────────────── */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 14px rgba(0,0,0,0.4); }}
    [data-testid="stMetricValue"] {{ font-size: 1.6rem !important; }}
    .stDataFrame {{ border-radius: 10px; overflow: hidden; }}
    .stSlider > div {{ padding: 0.5rem 0; }}
    footer {{ display: none !important; }}
    #MainMenu {{ display: none !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────


def _init_state() -> None:
    defaults: dict = {
        "last_prediction": None,
        "last_image_bytes": None,
        "last_filename": None,
        "feedback_submitted": False,
        "drift_injected": False,
        "drift_log": "",
        "retrain_log": "",
        "retrain_done": False,
        "before_metrics": None,
        "after_metrics": None,
        "prediction_history": [],
        "uncertain_count": 0,
        "confirm_reset": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────


def _api_health() -> tuple[bool, str]:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3.0)
        return r.status_code == 200, (
            "Healthy" if r.status_code == 200 else f"HTTP {r.status_code}"
        )
    except httpx.ConnectError:
        return False, "Unreachable"
    except Exception as exc:
        return False, str(exc)[:40]


def _predict(image_bytes: bytes, filename: str) -> dict:
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            f"{API_URL}/predict",
            files={"file": (filename, image_bytes, "image/jpeg")},
        )
    resp.raise_for_status()
    return resp.json()


def _save_feedback(filename: str, predicted: str, correct: str) -> None:
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FEEDBACK_LOG.exists()
    with FEEDBACK_LOG.open("a", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["timestamp", "filename", "predicted_label", "correct_label"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "filename": filename,
                "predicted_label": predicted,
                "correct_label": correct,
            }
        )


def _feedback_stats() -> tuple[int, int]:
    if not FEEDBACK_LOG.exists():
        return 0, 0
    df = pd.read_csv(FEEDBACK_LOG)
    mismatches = int((df["predicted_label"] != df["correct_label"]).sum())
    return len(df), mismatches


def _inference_stats() -> dict:
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


def _load_mlflow_metrics() -> dict | None:
    mlruns = PROJECT_ROOT / "tracking"
    if not mlruns.exists():
        return None
    try:
        best: dict | None = None
        best_ts: float = 0.0
        for mf in mlruns.glob("*/*/metrics/val_accuracy"):
            lines = mf.read_text().strip().splitlines()
            if not lines:
                continue
            ts, _val, _ = lines[-1].split()
            if float(ts) > best_ts:
                best_ts = float(ts)
                run_dir = mf.parent.parent
                metrics: dict = {}
                for metric_file in (run_dir / "metrics").iterdir():
                    mlines = metric_file.read_text().strip().splitlines()
                    if mlines:
                        metrics[metric_file.name] = float(mlines[-1].split()[1])
                best = metrics
        return best
    except Exception:
        return None


def _stream_subprocess(cmd: list[str], cwd: Path, placeholder) -> int:
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
        visible = "\n".join(log_lines[-80:])
        placeholder.markdown(
            f'<div class="log-box">{visible}</div>', unsafe_allow_html=True
        )
    proc.wait()
    return proc.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Plotly helpers — dark-themed charts
# ─────────────────────────────────────────────────────────────────────────────

_PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_PRIMARY, family="Inter, system-ui, sans-serif"),
)


def _confidence_gauge(confidence: float, label: str) -> go.Figure:
    bar_color = TEAL if label == "cat" else DANGER
    pct = round(confidence * 100, 1)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 36, "color": TEXT_PRIMARY}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": TEXT_SECONDARY,
                    "tickfont": {"color": TEXT_SECONDARY, "size": 10},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": SURFACE_2,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(239,68,68,0.12)"},
                    {"range": [50, 80], "color": "rgba(245,158,11,0.10)"},
                    {"range": [80, 100], "color": "rgba(6,214,160,0.10)"},
                ],
                "threshold": {
                    "line": {"color": WARNING, "width": 2},
                    "thickness": 0.8,
                    "value": 80,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(t=30, b=10, l=30, r=30),
        **_PLOTLY_DARK,
    )
    return fig


def _label_bar_chart(label_counts: dict) -> go.Figure:
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    colors = [TEAL if lbl == "cat" else DANGER for lbl in labels]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            marker_color=colors,
            marker_line_width=0,
            text=counts,
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, size=13),
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(t=10, b=10, l=20, r=20),
        xaxis=dict(showgrid=False, tickfont=dict(color=TEXT_SECONDARY)),
        yaxis=dict(
            showgrid=True, gridcolor=BORDER, tickfont=dict(color=TEXT_SECONDARY)
        ),
        **_PLOTLY_DARK,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem;">
            <div style="font-size:3.5rem; line-height:1;">🐱</div>
            <div style="font-size:1.35rem; font-weight:800; color:#f1f5f9; margin-top:0.4rem;">
                Am I a Cat?
            </div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.2rem; letter-spacing:0.05em;">
                MLOps PORTFOLIO DEMO · PHASE 9
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Live API health
    healthy, health_msg = _api_health()
    ts = datetime.now().strftime("%H:%M:%S")
    pill_cls = "health-up" if healthy else "health-down"
    status_icon = "●" if healthy else "●"
    st.markdown(
        f"""
        <div style="margin-bottom:0.4rem; font-size:0.72rem; color:{TEXT_SECONDARY};
                    text-transform:uppercase; letter-spacing:0.05em;">Backend API</div>
        <div class="health-pill {pill_cls}">
            <span class="health-dot"></span>
            <span>{health_msg}</span>
        </div>
        <div style="font-size:0.7rem; color:{TEXT_SECONDARY}; margin-top:0.2rem;">
            {API_URL} · checked {ts}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Confidence threshold slider
    st.markdown(
        f'<div style="font-size:0.72rem; color:{TEXT_SECONDARY}; text-transform:uppercase; '
        f'letter-spacing:0.05em; margin-bottom:0.4rem;">Confidence Threshold</div>',
        unsafe_allow_html=True,
    )
    confidence_threshold = st.slider(
        "threshold",
        min_value=0.50,
        max_value=1.00,
        value=0.80,
        step=0.01,
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div style="font-size:0.72rem; color:{TEXT_SECONDARY}; margin-top:-0.4rem;">'
        f'Predictions below <b style="color:{WARNING}">{confidence_threshold:.0%}</b> '
        f"flagged as uncertain</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Session statistics
    st.markdown(
        f'<div style="font-size:0.72rem; color:{TEXT_SECONDARY}; text-transform:uppercase; '
        f'letter-spacing:0.05em; margin-bottom:0.75rem;">Session Statistics</div>',
        unsafe_allow_html=True,
    )
    total_preds = len(st.session_state.prediction_history)
    total_fb, _ = _feedback_stats()
    uncertain = st.session_state.uncertain_count

    for label, value, chip_cls in [
        ("Predictions", total_preds, "chip-teal"),
        ("Feedback submitted", total_fb, "chip-green"),
        ("Uncertain flags", uncertain, "chip-amber"),
    ]:
        st.markdown(
            f"""
            <div class="sidebar-stat">
                <div class="sidebar-stat-label">{label}</div>
                <div class="sidebar-stat-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Reset session button
    if not st.session_state.confirm_reset:
        if st.button("↺  Reset Session", use_container_width=True):
            st.session_state.confirm_reset = True
            st.rerun()
    else:
        st.warning("Are you sure? All session data will be cleared.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, reset", type="primary", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                _init_state()
                st.rerun()
        with c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_reset = False
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_predict, tab_pipeline, tab_monitor = st.tabs(
    ["🔮  Live Prediction", "⚡  Pipeline Control", "📊  Monitoring"]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Prediction
# ═══════════════════════════════════════════════════════════════════════════

with tab_predict:
    st.markdown(
        f"""
        <h2 style="margin-bottom:0.25rem;">Live Image Classification</h2>
        <p style="color:{TEXT_SECONDARY}; margin-bottom:1.5rem;">
            Upload any image — the ResNet50 model classifies it in real time via the FastAPI backend.
            Submit feedback to feed the closed-loop retraining pipeline.
        </p>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_result = st.columns([1, 1], gap="large")

    # ── Left: Upload + preview ───────────────────────────────────────────

    with col_upload:
        st.markdown(
            f'<div style="font-weight:600; color:{TEXT_PRIMARY}; margin-bottom:0.5rem;">'
            "📁 Upload Image</div>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Drop an image here (JPG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="visible",
        )

        if uploaded_file:
            image_bytes = uploaded_file.read()
            # Only run prediction when the file changes
            if image_bytes != st.session_state.last_image_bytes:
                st.session_state.last_image_bytes = image_bytes
                st.session_state.last_filename = uploaded_file.name
                st.session_state.last_prediction = None
                st.session_state.feedback_submitted = False

            pil_img = Image.open(io.BytesIO(image_bytes))
            st.image(
                pil_img,
                caption=uploaded_file.name,
                use_container_width=True,
            )

            if not healthy:
                st.markdown(
                    '<div class="warning-banner">⚠️ API is offline — start it with '
                    "<code>make serve</code> or <code>make demo</code></div>",
                    unsafe_allow_html=True,
                )
            elif st.session_state.last_prediction is None:
                with st.spinner("Classifying…"):
                    try:
                        result = _predict(image_bytes, uploaded_file.name)
                        st.session_state.last_prediction = result
                        is_uncertain = result["confidence"] < confidence_threshold
                        if is_uncertain:
                            st.session_state.uncertain_count += 1
                        st.session_state.prediction_history.append(
                            {
                                "Time": datetime.now().strftime("%H:%M:%S"),
                                "File": uploaded_file.name,
                                "Label": result["label"],
                                "Confidence": f'{result["confidence"]:.1%}',
                            }
                        )
                    except httpx.HTTPStatusError as exc:
                        st.error(
                            f"API error {exc.response.status_code}: {exc.response.text[:200]}"
                        )
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
        else:
            # Placeholder state
            st.markdown(
                f"""
                <div style="border: 2px dashed {BORDER}; border-radius:14px;
                            background:{SURFACE_2}; padding:3rem 1.5rem;
                            text-align:center; color:{TEXT_SECONDARY};">
                    <div style="font-size:3rem; margin-bottom:0.75rem;">🖼️</div>
                    <div style="font-weight:600; color:{TEXT_PRIMARY}; margin-bottom:0.4rem;">
                        Drop an image to begin
                    </div>
                    <div style="font-size:0.82rem;">Supports JPG, PNG, WebP</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Right: Prediction result ─────────────────────────────────────────

    with col_result:
        pred = st.session_state.last_prediction

        if pred is None:
            st.markdown(
                f"""
                <div class="card" style="text-align:center; padding:3rem 1.5rem; min-height:250px;
                                        display:flex; flex-direction:column; align-items:center; justify-content:center;">
                    <div style="font-size:2.5rem; margin-bottom:1rem; opacity:0.4;">🔮</div>
                    <div style="font-weight:600; color:{TEXT_SECONDARY};">
                        Prediction will appear here
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            label: str = pred["label"]
            confidence: float = pred["confidence"]
            is_uncertain = confidence < confidence_threshold

            # Result card
            if label == "cat":
                card_class = "card-cat"
                emoji = "🐱"
                label_display = "It's a Cat!"
            else:
                card_class = "card-notcat"
                emoji = "❌"
                label_display = "Not a Cat"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="font-size:4rem; line-height:1.1;">{emoji}</div>
                    <div style="font-size:1.8rem; font-weight:800; color:#fff;
                                margin-top:0.5rem; text-shadow:0 2px 8px rgba(0,0,0,0.4);">
                        {label_display}
                    </div>
                    <div style="margin-top:0.4rem;">
                        <span class="chip {'chip-teal' if label == 'cat' else 'chip-red'}">
                            {label.replace("_", " ")}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Low-confidence warning
            if is_uncertain:
                st.markdown(
                    f"""
                    <div class="warning-banner">
                        ⚠️ Low confidence ({confidence:.1%}) — below your {confidence_threshold:.0%} threshold.
                        The model is uncertain about this image.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Gauge
            st.plotly_chart(
                _confidence_gauge(confidence, label),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # Feedback section
            st.markdown(
                f'<div style="font-weight:600; color:{TEXT_PRIMARY}; margin-top:0.5rem;">Submit Feedback</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-size:0.8rem; color:{TEXT_SECONDARY}; margin-bottom:0.75rem;">'
                "Was this correct? Your correction feeds the retraining loop.</div>",
                unsafe_allow_html=True,
            )

            correct_label = st.radio(
                "Correct label",
                options=["cat", "not_cat"],
                index=0 if label == "cat" else 1,
                horizontal=True,
                key="correct_label_radio",
                label_visibility="collapsed",
            )

            if st.session_state.feedback_submitted:
                if correct_label == label:
                    st.success("✅ Confirmed — thanks for the validation!")
                else:
                    st.info(
                        f"📝 Correction recorded: **{label}** → **{correct_label}**"
                    )
            else:
                btn_label = (
                    "✅ Confirm correct"
                    if correct_label == label
                    else "📝 Submit correction"
                )
                btn_type = "secondary" if correct_label == label else "primary"
                if st.button(btn_label, type=btn_type, use_container_width=True):
                    fname = st.session_state.last_filename or "unknown.jpg"
                    _save_feedback(fname, label, correct_label)
                    st.session_state.feedback_submitted = True
                    st.rerun()

    # ── Prediction history ────────────────────────────────────────────────

    if st.session_state.prediction_history:
        st.markdown("<hr>", unsafe_allow_html=True)
        with st.expander("📋 Prediction History (this session)", expanded=False):
            hist_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Pipeline Control
# ═══════════════════════════════════════════════════════════════════════════

with tab_pipeline:
    st.markdown(
        f"""
        <h2 style="margin-bottom:0.25rem;">Closed-Loop Retraining Pipeline</h2>
        <p style="color:{TEXT_SECONDARY}; margin-bottom:1.5rem;">
            Simulate real-world model degradation by injecting synthetic drift, then trigger a full
            DVC retraining run and observe before/after metrics and promotion status.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Step 1: Inject Drift ─────────────────────────────────────────────

    step1_done = st.session_state.drift_injected
    step1_badge = "step-done" if step1_done else "step-active"
    step1_icon = "✓" if step1_done else "1"

    st.markdown(
        f"""
        <div class="step-header">
            <div class="step-badge {step1_badge}">{step1_icon}</div>
            <div>
                <div class="step-title">Inject Synthetic Drift</div>
                <div class="step-subtitle">
                    Sends OOD images to /predict — simulates production distribution shift
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            f"""
            <div class="card">
                <p style="color:{TEXT_SECONDARY}; margin:0 0 1rem;">
                    Generates <strong style="color:{TEXT_PRIMARY}">noise, gradient, solid-colour, and blurred</strong>
                    images — maximally out-of-distribution from the cat/not-cat training set.
                    Each image is POSTed to the API and logged to
                    <code>monitoring/inference_log.csv</code> for Evidently AI drift analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_drift_ctrl, col_drift_btn = st.columns([3, 1])
        with col_drift_ctrl:
            drift_count = st.number_input(
                "Number of synthetic images",
                min_value=10,
                max_value=500,
                value=50,
                step=10,
                help="More images = stronger statistical signal for drift detection",
            )
        with col_drift_btn:
            st.write("")
            st.write("")
            inject_btn = st.button(
                "💉 Inject Drift",
                use_container_width=True,
                type="primary" if not step1_done else "secondary",
                disabled=not healthy,
            )

        drift_log_ph = st.empty()
        if st.session_state.drift_log:
            drift_log_ph.markdown(
                f'<div class="log-box">{st.session_state.drift_log}</div>',
                unsafe_allow_html=True,
            )

    if inject_btn:
        st.session_state.drift_log = "Starting drift injection…\n"
        cmd = [
            sys.executable,
            "-u",
            str(SIMULATE_SCRIPT),
            "--count",
            str(int(drift_count)),
            "--api-url",
            API_URL,
            "--output-log",
            str(INFERENCE_LOG),
        ]
        with st.spinner(f"Injecting {int(drift_count)} synthetic images…"):
            rc = _stream_subprocess(cmd, PROJECT_ROOT, drift_log_ph)

        if rc == 0:
            st.session_state.drift_injected = True
            st.success(
                f"✅ Drift injection complete — {int(drift_count)} images sent to API."
            )
        else:
            st.error(f"⛔ Injection failed (exit code {rc}) — check the log above.")
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Step 2: Trigger Retraining ────────────────────────────────────────

    step2_locked = not step1_done
    step2_done = st.session_state.retrain_done
    step2_badge = (
        "step-done"
        if step2_done
        else ("step-active" if not step2_locked else "step-locked")
    )
    step2_icon = "✓" if step2_done else "2"

    st.markdown(
        f"""
        <div class="step-header">
            <div class="step-badge {step2_badge}">{step2_icon}</div>
            <div>
                <div class="step-title" style="color: {'#94a3b8' if step2_locked else '#f1f5f9'}">
                    Trigger Retraining
                </div>
                <div class="step-subtitle">
                    {"🔒 Complete Step 1 first" if step2_locked else "Runs the full DVC pipeline end-to-end"}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            f"""
            <div class="card" style="{'opacity:0.5;' if step2_locked else ''}">
                <p style="color:{TEXT_SECONDARY}; margin:0 0 1rem;">
                    Executes <code>dvc repro --force</code>:
                    <strong style="color:{TEXT_PRIMARY}">ingest → validate → features → train → evaluate</strong>.
                    Promotion gates: <span class="chip chip-teal">val_accuracy ≥ 0.94</span>
                    <span class="chip chip-teal">val_f1 ≥ 0.93</span>.
                    Live output streams below.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_rt_info, col_rt_btn = st.columns([3, 1])
        with col_rt_btn:
            retrain_btn = st.button(
                "🚀 Retrain Model",
                use_container_width=True,
                type="primary",
                disabled=step2_locked,
            )

        retrain_log_ph = st.empty()
        if st.session_state.retrain_log:
            retrain_log_ph.markdown(
                f'<div class="log-box">{st.session_state.retrain_log}</div>',
                unsafe_allow_html=True,
            )

    if retrain_btn and not step2_locked:
        st.session_state.before_metrics = _load_mlflow_metrics()
        st.session_state.retrain_log = "Starting DVC pipeline…\n"
        st.session_state.retrain_done = False

        cmd = [sys.executable, "-m", "dvc", "repro", "--force"]
        with st.spinner("Running DVC pipeline — this may take several minutes…"):
            rc = _stream_subprocess(cmd, PROJECT_ROOT, retrain_log_ph)

        if rc == 0:
            time.sleep(1)  # let MLflow flush metrics to disk
            st.session_state.after_metrics = _load_mlflow_metrics()
            st.session_state.retrain_done = True
            st.success("✅ Retraining pipeline finished successfully.")
        else:
            st.error(f"⛔ Pipeline failed (exit code {rc}) — check the log above.")
        st.rerun()

    # ── Step 3: Metrics Comparison ────────────────────────────────────────

    if st.session_state.retrain_done:
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="step-header">
                <div class="step-badge step-done">✓</div>
                <div>
                    <div class="step-title">Before / After Metrics</div>
                    <div class="step-subtitle">Model quality delta + promotion verdict</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        before = st.session_state.before_metrics or {}
        after = st.session_state.after_metrics or {}

        KEY_METRICS = [
            ("val_accuracy", "Accuracy", 0.94),
            ("val_f1", "F1 Score", 0.93),
            ("val_precision", "Precision", None),
            ("val_recall", "Recall", None),
            ("val_roc_auc", "ROC-AUC", None),
        ]

        col_b, col_a = st.columns(2, gap="medium")

        with col_b:
            st.markdown(
                f'<div style="font-weight:600; color:{TEXT_SECONDARY}; '
                f"font-size:0.85rem; margin-bottom:0.6rem; text-transform:uppercase; "
                f'letter-spacing:0.05em;">Before Retraining</div>',
                unsafe_allow_html=True,
            )
            for key, display, _ in KEY_METRICS:
                v = before.get(key)
                disp = f"{v:.4f}" if v is not None else "—"
                st.markdown(
                    f'<div class="metric-row">'
                    f'<span class="metric-label">{display}</span>'
                    f'<span class="metric-value">{disp}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with col_a:
            st.markdown(
                f'<div style="font-weight:600; color:{TEAL}; '
                f"font-size:0.85rem; margin-bottom:0.6rem; text-transform:uppercase; "
                f'letter-spacing:0.05em;">After Retraining</div>',
                unsafe_allow_html=True,
            )
            for key, display, gate in KEY_METRICS:
                v_a = after.get(key)
                v_b = before.get(key)
                disp = f"{v_a:.4f}" if v_a is not None else "—"

                delta_html = ""
                if v_a is not None and v_b is not None:
                    diff = v_a - v_b
                    sign = "+" if diff >= 0 else ""
                    delta_cls = (
                        "delta-up"
                        if diff > 0
                        else ("delta-down" if diff < 0 else "delta-flat")
                    )
                    delta_html = f'<span class="{delta_cls}">  {sign}{diff:.4f}</span>'

                gate_html = ""
                if gate is not None and v_a is not None:
                    gate_html = f'<span style="margin-left:0.4rem;">{"✅" if v_a >= gate else "❌"}</span>'

                st.markdown(
                    f'<div class="metric-row">'
                    f'<span class="metric-label">{display}</span>'
                    f'<span class="metric-value">{disp}{delta_html}{gate_html}</span>'
                    f"</div>",
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
                st.markdown(
                    f"""
                    <div class="banner-promoted">
                        <div style="font-size:1.6rem; margin-bottom:0.3rem;">🏆</div>
                        <div style="font-size:1.1rem; font-weight:800; color:{SUCCESS};">
                            Model PROMOTED to Staging
                        </div>
                        <div style="font-size:0.82rem; color:{TEXT_SECONDARY}; margin-top:0.3rem;">
                            val_accuracy ≥ 0.94 ✓  ·  val_f1 ≥ 0.93 ✓
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="banner-failed">
                        <div style="font-size:1.6rem; margin-bottom:0.3rem;">⛔</div>
                        <div style="font-size:1.1rem; font-weight:800; color:{DANGER};">
                            Model Did Not Pass Promotion Gates
                        </div>
                        <div style="font-size:0.82rem; color:{TEXT_SECONDARY}; margin-top:0.3rem;">
                            Requires val_accuracy ≥ 0.94 AND val_f1 ≥ 0.93
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Monitoring
# ═══════════════════════════════════════════════════════════════════════════

with tab_monitor:
    col_title, col_refresh = st.columns([5, 1])
    with col_title:
        st.markdown(
            f"""
            <h2 style="margin-bottom:0.25rem;">Live Monitoring Dashboard</h2>
            <p style="color:{TEXT_SECONDARY}; margin-bottom:1rem;">
                Inference metrics from <code>monitoring/inference_log.csv</code>
                + Prometheus/Grafana stack.
            </p>
            """,
            unsafe_allow_html=True,
        )
    with col_refresh:
        st.write("")
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # External links row
    col_g, col_p, col_spacer = st.columns([1, 1, 4])
    with col_g:
        st.link_button("📊 Open Grafana ↗", GRAFANA_URL, use_container_width=True)
    with col_p:
        st.link_button(
            "🔥 Prometheus ↗", "http://localhost:9090", use_container_width=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Metric chips row
    stats = _inference_stats()
    total_fb, mismatch_fb = _feedback_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Inferences", stats["rows"])
    with col2:
        avg_conf = stats.get("avg_confidence")
        delta_msg = None
        if avg_conf is not None and avg_conf < 0.80:
            delta_msg = "⚠️ below threshold"
        st.metric(
            "Avg Confidence",
            f"{avg_conf:.1%}" if avg_conf is not None else "—",
            delta=delta_msg,
        )
    with col3:
        min_conf = stats.get("min_confidence")
        st.metric("Min Confidence", f"{min_conf:.1%}" if min_conf is not None else "—")
    with col4:
        error_rate = f"{mismatch_fb/total_fb:.0%}" if total_fb else "—"
        st.metric(
            "Feedback Error Rate",
            error_rate,
            delta=f"{mismatch_fb} corrections" if mismatch_fb else None,
        )

    # Label distribution chart
    if stats["rows"] > 0 and "label_counts" in stats:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-weight:600; color:{TEXT_PRIMARY}; margin-bottom:0.5rem;">'
            "Label Distribution</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _label_bar_chart(stats["label_counts"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # Recent inferences table
    st.markdown("<hr>", unsafe_allow_html=True)
    col_inf_h, col_inf_count = st.columns([3, 1])
    with col_inf_h:
        st.markdown(
            f'<div style="font-weight:600; color:{TEXT_PRIMARY}; margin-bottom:0.5rem;">'
            "Recent Inferences (last 20)</div>",
            unsafe_allow_html=True,
        )

    if INFERENCE_LOG.exists() and stats["rows"] > 0:
        df_log = pd.read_csv(INFERENCE_LOG).tail(20)
        # Format numeric columns
        for col in [
            "pixel_mean_r",
            "pixel_mean_g",
            "pixel_mean_b",
            "pixel_std_r",
            "pixel_std_g",
            "pixel_std_b",
        ]:
            if col in df_log.columns:
                df_log[col] = df_log[col].map("{:.1f}".format)
        if "confidence" in df_log.columns:
            df_log["confidence"] = df_log["confidence"].map("{:.4f}".format)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            f"""
            <div class="card" style="text-align:center; padding:1.5rem;">
                <div style="font-size:1.5rem; margin-bottom:0.5rem;">📭</div>
                <div style="color:{TEXT_SECONDARY}; font-size:0.88rem;">
                    No inference data yet. Upload images in <strong>Live Prediction</strong>
                    or run <strong>Inject Drift</strong> to populate the log.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Feedback log table
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-weight:600; color:{TEXT_PRIMARY}; margin-bottom:0.5rem;">'
        "Feedback Log (last 10)</div>",
        unsafe_allow_html=True,
    )

    if total_fb > 0:
        df_fb = pd.read_csv(FEEDBACK_LOG).tail(10)
        st.dataframe(df_fb, use_container_width=True, hide_index=True)
        error_pct = f"{mismatch_fb/total_fb:.0%}" if total_fb else "0%"
        chip_cls = "chip-red" if mismatch_fb > 0 else "chip-green"
        st.markdown(
            f'<span class="chip {chip_cls}">{total_fb} entries</span> '
            f'<span class="chip chip-amber">{mismatch_fb} corrections ({error_pct} error rate)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="card" style="text-align:center; padding:1.5rem;">
                <div style="font-size:1.5rem; margin-bottom:0.5rem;">💬</div>
                <div style="color:{TEXT_SECONDARY}; font-size:0.88rem;">
                    No feedback submitted yet. Use <strong>Live Prediction</strong>
                    to submit corrections.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Grafana info block
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card">
            <div style="font-weight:700; color:{TEXT_PRIMARY}; margin-bottom:0.75rem;">
                📊 Grafana & Prometheus Setup
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;">
                <div>
                    <div style="font-size:0.78rem; color:{TEXT_SECONDARY}; text-transform:uppercase;
                                letter-spacing:0.05em; margin-bottom:0.4rem;">Grafana Login</div>
                    <span class="chip">URL: {GRAFANA_URL}</span><br>
                    <span class="chip chip-teal" style="margin-top:0.3rem;">admin / catops</span>
                </div>
                <div>
                    <div style="font-size:0.78rem; color:{TEXT_SECONDARY}; text-transform:uppercase;
                                letter-spacing:0.05em; margin-bottom:0.4rem;">Prometheus Metrics</div>
                    <span class="chip chip-teal">catops_predictions_total</span><br>
                    <span class="chip chip-teal" style="margin-top:0.3rem;">catops_prediction_confidence</span><br>
                    <span class="chip chip-teal" style="margin-top:0.3rem;">http_requests_total</span>
                </div>
            </div>
            <div style="margin-top:1rem; font-size:0.8rem; color:{TEXT_SECONDARY};">
                💡 Start the full monitoring stack with <code>make demo</code> —
                launches FastAPI + Streamlit + Prometheus + Grafana in one command.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="text-align:center; padding:0.5rem 0 1rem; color:{TEXT_SECONDARY}; font-size:0.75rem;">
        <strong style="color:{TEAL};">Am I a Cat?</strong>
        · End-to-end MLOps portfolio project
        · Poetry · DVC · Hydra · PyTorch ResNet50 · MLflow · FastAPI
        · Evidently AI · Prometheus · Grafana · GitHub Actions
    </div>
    """,
    unsafe_allow_html=True,
)
