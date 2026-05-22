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
DEMO_DATA_DIR: Path = Path(__file__).parent / "demo_data"

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────

# Refined Dark Teal — deeper near-black canvas, crisper teal accent.
TEAL = "#06d6a0"
TEAL_DIM = "#0e9f78"
NAVY = "#070b14"
SURFACE = "#0e1521"
SURFACE_2 = "#1a2438"
BORDER = "#26334d"
TEXT_PRIMARY = "#f1f5f9"
TEXT_SECONDARY = "#9aa7bd"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER = "#ef4444"
CAT_GRADIENT = "linear-gradient(140deg, #06443a 0%, #0a6e54 60%, #0f9b78 100%)"
NOT_CAT_GRADIENT = "linear-gradient(140deg, #6e1f1f 0%, #9a2725 60%, #c0392f 100%)"
UNCERTAIN_GRADIENT = "linear-gradient(140deg, #6b3410 0%, #8a4a12 60%, #ab6510 100%)"

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
    /* ════════════════════════════════════════════════════════════════
       "Am I a Cat?" — design system  ·  Refined Dark Teal
       ════════════════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Base ───────────────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        background-color: {NAVY};
        color: {TEXT_PRIMARY};
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}
    [data-testid="stAppViewContainer"] {{
        background:
            radial-gradient(1000px 520px at 84% -10%, rgba(6,214,160,0.07), transparent 70%),
            {NAVY};
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stAppDeployButton"] {{ display: none !important; }}
    ::selection {{ background: rgba(6,214,160,0.30); color: #fff; }}

    /* tighten default main padding so the hero sits near the top */
    [data-testid="stMainBlockContainer"], .block-container {{
        padding-top: 2.4rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    /* ── Typography ─────────────────────────────────────────────────── */
    h1, h2, h3, h4 {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 700;
        letter-spacing: -0.021em;
        line-height: 1.22;
    }}
    h2 {{ font-size: 1.72rem !important; }}
    h3 {{ font-size: 1.2rem !important; }}
    p, li, span {{ color: {TEXT_SECONDARY}; }}
    p, li {{ line-height: 1.62; }}
    .stMarkdown p {{ color: {TEXT_SECONDARY}; }}
    a {{ color: {TEAL}; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{
        background: rgba(6,214,160,0.10);
        color: {TEAL};
        border-radius: 5px;
        padding: 0.07rem 0.36rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85em;
    }}

    /* ── Sidebar ────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {SURFACE} !important;
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] > div:first-child {{ padding-top: 1.4rem; }}

    /* ── Tabs ───────────────────────────────────────────────────────── */
    [data-testid="stTabs"] [role="tablist"] {{
        border-bottom: 1px solid {BORDER};
        gap: 0.3rem;
    }}
    [data-testid="stTabs"] [role="tab"] {{
        color: {TEXT_SECONDARY};
        border-radius: 9px 9px 0 0;
        padding: 0.55rem 1.25rem;
        font-weight: 600;
        font-size: 0.92rem;
        transition: color .18s ease, background .18s ease;
    }}
    [data-testid="stTabs"] [role="tab"]:hover {{
        color: {TEXT_PRIMARY};
        background: rgba(255,255,255,0.035);
    }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        color: {TEAL};
        background: rgba(6,214,160,0.09);
        box-shadow: inset 0 -2px 0 {TEAL};
    }}

    /* ── Cards ──────────────────────────────────────────────────────── */
    .card {{
        background: {SURFACE_2};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 28px rgba(0,0,0,0.38);
        transition: border-color .2s ease, box-shadow .2s ease;
    }}
    .card:hover {{
        border-color: rgba(6,214,160,0.30);
        box-shadow: 0 12px 38px rgba(0,0,0,0.5);
    }}

    /* result cards — animated entrance smooths the inference handoff */
    .card-cat, .card-notcat, .card-uncertain {{
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
        animation: cardIn .42s cubic-bezier(.2,.7,.2,1);
    }}
    .card-cat {{
        background: {CAT_GRADIENT};
        border: 1px solid {TEAL_DIM};
        box-shadow: 0 14px 42px rgba(6,214,160,0.22);
    }}
    .card-notcat {{
        background: {NOT_CAT_GRADIENT};
        border: 1px solid #b1342f;
        box-shadow: 0 14px 42px rgba(239,68,68,0.20);
    }}
    .card-uncertain {{
        background: {UNCERTAIN_GRADIENT};
        border: 1px solid {WARNING};
        box-shadow: 0 14px 42px rgba(245,158,11,0.20);
    }}

    /* ── Chips ──────────────────────────────────────────────────────── */
    .chip {{
        display: inline-block;
        background: {SURFACE_2};
        border: 1px solid {BORDER};
        border-radius: 999px;
        padding: 0.3rem 0.8rem;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: {TEXT_PRIMARY};
        margin: 0.15rem;
    }}
    .chip-teal  {{ border-color: rgba(6,214,160,0.55);  color: {TEAL};    background: rgba(6,214,160,0.11); }}
    .chip-green {{ border-color: rgba(34,197,94,0.55);  color: {SUCCESS}; background: rgba(34,197,94,0.11); }}
    .chip-red   {{ border-color: rgba(239,68,68,0.55);  color: {DANGER};  background: rgba(239,68,68,0.11); }}
    .chip-amber {{ border-color: rgba(245,158,11,0.55); color: {WARNING}; background: rgba(245,158,11,0.11); }}

    /* ── Health pill ────────────────────────────────────────────────── */
    .health-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        border-radius: 999px;
        padding: 0.32rem 0.85rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.45rem 0;
    }}
    .health-up   {{ background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.55); color: {SUCCESS}; }}
    .health-down {{ background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.55); color: {DANGER}; }}
    .health-dot  {{ width: 7px; height: 7px; border-radius: 50%; }}
    .health-up   .health-dot {{ background: {SUCCESS}; box-shadow: 0 0 8px {SUCCESS}; animation: pulse 2s infinite; }}
    .health-down .health-dot {{ background: {DANGER}; }}

    /* ── Terminal log box ───────────────────────────────────────────── */
    .log-box {{
        background: #05080d;
        border: 1px solid {BORDER};
        border-radius: 12px;
        color: #7cffb2;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.75rem;
        line-height: 1.65;
        padding: 1rem 1.15rem;
        height: 320px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-all;
        box-shadow: inset 0 2px 14px rgba(0,0,0,0.6);
    }}

    /* ── Step wizard ────────────────────────────────────────────────── */
    .step-header {{
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.8rem;
    }}
    .step-badge {{
        width: 34px; height: 34px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.9rem;
        flex-shrink: 0;
    }}
    .step-active   {{ background: {TEAL}; color: #04110d; box-shadow: 0 0 0 4px rgba(6,214,160,0.16); }}
    .step-done     {{ background: {SUCCESS}; color: #04130a; }}
    .step-locked   {{ background: {SURFACE_2}; color: {TEXT_SECONDARY}; border: 1px solid {BORDER}; }}
    .step-title    {{ font-size: 1.06rem; font-weight: 700; color: {TEXT_PRIMARY}; letter-spacing: -0.01em; }}
    .step-subtitle {{ font-size: 0.82rem; color: {TEXT_SECONDARY}; }}

    /* ── Metric comparison rows ─────────────────────────────────────── */
    .metric-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.62rem 0.9rem;
        border-radius: 9px;
        margin-bottom: 0.4rem;
        background: rgba(255,255,255,0.025);
        border: 1px solid {BORDER};
    }}
    .metric-label {{ font-size: 0.82rem; color: {TEXT_SECONDARY}; }}
    .metric-value {{ font-size: 0.95rem; font-weight: 600; color: {TEXT_PRIMARY}; font-family: 'JetBrains Mono', monospace; }}
    .delta-up   {{ color: {SUCCESS}; font-size: 0.8rem; }}
    .delta-down {{ color: {DANGER};  font-size: 0.8rem; }}
    .delta-flat {{ color: {TEXT_SECONDARY}; font-size: 0.8rem; }}

    /* ── Promotion banners ──────────────────────────────────────────── */
    .banner-promoted, .banner-failed {{
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        text-align: center;
        margin-top: 1rem;
        animation: cardIn .42s cubic-bezier(.2,.7,.2,1);
    }}
    .banner-promoted {{
        background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(6,214,160,0.08));
        border: 1px solid rgba(34,197,94,0.6);
    }}
    .banner-failed {{
        background: linear-gradient(135deg, rgba(239,68,68,0.14), rgba(245,158,11,0.07));
        border: 1px solid rgba(239,68,68,0.6);
    }}

    /* ── Low-confidence / inline warning banner ─────────────────────── */
    .warning-banner {{
        background: rgba(245,158,11,0.10);
        border: 1px solid rgba(245,158,11,0.45);
        border-left: 3px solid {WARNING};
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #fbd9a5;
        font-size: 0.86rem;
        font-weight: 500;
        margin: 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* ── Sidebar stat cards ─────────────────────────────────────────── */
    .sidebar-stat {{
        background: rgba(255,255,255,0.035);
        border: 1px solid {BORDER};
        border-radius: 11px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
        transition: border-color .18s ease;
    }}
    .sidebar-stat:hover {{ border-color: rgba(6,214,160,0.3); }}
    .sidebar-stat-label {{ font-size: 0.68rem; color: {TEXT_SECONDARY}; text-transform: uppercase; letter-spacing: 0.07em; }}
    .sidebar-stat-value {{ font-size: 1.35rem; font-weight: 800; color: {TEXT_PRIMARY}; margin-top: 0.1rem; }}

    /* ── Buttons ────────────────────────────────────────────────────── */
    .stButton button {{
        border-radius: 9px;
        font-weight: 600;
        border: 1px solid {BORDER};
        background: {SURFACE_2};
        color: {TEXT_PRIMARY};
        transition: transform .12s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease;
    }}
    .stButton button:hover {{
        transform: translateY(-1px);
        border-color: rgba(6,214,160,0.5);
        box-shadow: 0 6px 18px rgba(0,0,0,0.45);
    }}
    .stButton button:active {{ transform: translateY(0); }}
    .stButton button[data-testid="stBaseButton-primary"],
    .stButton button[kind="primary"] {{
        background: {TEAL};
        color: #04110d;
        border-color: {TEAL};
        box-shadow: 0 6px 18px rgba(6,214,160,0.28);
    }}
    .stButton button[data-testid="stBaseButton-primary"]:hover,
    .stButton button[kind="primary"]:hover {{
        background: #0ee9b0;
        border-color: #0ee9b0;
        box-shadow: 0 8px 24px rgba(6,214,160,0.42);
    }}
    .stLinkButton a {{
        border-radius: 9px !important;
        border: 1px solid {BORDER} !important;
        font-weight: 600 !important;
    }}

    /* ── Inputs ─────────────────────────────────────────────────────── */
    [data-testid="stFileUploaderDropzone"] {{
        border: 1.5px dashed {BORDER};
        border-radius: 12px;
        background: rgba(255,255,255,0.02);
        transition: border-color .18s ease, background .18s ease;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {TEAL};
        background: rgba(6,214,160,0.045);
    }}
    .stSlider > div {{ padding: 0.4rem 0; }}
    [data-baseweb="slider"] [role="slider"] {{ background: {TEAL} !important; }}
    [data-testid="stImage"] img, .stImage img {{
        border-radius: 12px;
        border: 1px solid {BORDER};
    }}

    /* ── Alerts — success / error / warning / info ──────────────────── */
    [data-testid="stAlertContainer"] {{
        border-radius: 11px;
        border: 1px solid {BORDER};
        font-size: 0.88rem;
    }}

    /* ── Spinner — inference loading state ──────────────────────────── */
    [data-testid="stSpinner"] {{
        background: {SURFACE_2};
        border: 1px solid rgba(6,214,160,0.28);
        border-radius: 12px;
        padding: 0.8rem 1.05rem;
        box-shadow: 0 8px 26px rgba(0,0,0,0.4);
    }}
    [data-testid="stSpinner"] > div {{ color: {TEAL}; font-weight: 600; }}

    /* ── Metrics ────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {{
        background: {SURFACE_2};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.9rem 1rem;
    }}
    [data-testid="stMetricValue"] {{ font-size: 1.55rem !important; font-weight: 800; }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_SECONDARY}; }}

    /* ── DataFrame / expander ───────────────────────────────────────── */
    .stDataFrame {{ border-radius: 11px; overflow: hidden; border: 1px solid {BORDER}; }}
    [data-testid="stExpander"] {{
        border: 1px solid {BORDER};
        border-radius: 11px;
        background: {SURFACE_2};
        overflow: hidden;
    }}

    /* ── Divider ────────────────────────────────────────────────────── */
    hr {{ border-color: {BORDER} !important; margin: 1.1rem 0; }}

    /* ── Scrollbars ─────────────────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 9px; height: 9px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 6px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #36476a; }}
    .log-box::-webkit-scrollbar {{ width: 5px; }}
    .log-box::-webkit-scrollbar-thumb {{ background: #1d3a2c; }}

    /* ── Animations ─────────────────────────────────────────────────── */
    @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.35; }} }}
    @keyframes cardIn {{
        from {{ opacity: 0; transform: translateY(10px) scale(0.99); }}
        to   {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}

    /* ── Chrome ─────────────────────────────────────────────────────── */
    footer, #MainMenu {{ display: none !important; }}

    /* ── Responsive ─────────────────────────────────────────────────── */
    @media (max-width: 900px) {{
        [data-testid="stMainBlockContainer"], .block-container {{
            padding-left: 1.1rem; padding-right: 1.1rem; padding-top: 1.6rem;
        }}
        h2 {{ font-size: 1.42rem !important; }}
        .card-cat, .card-notcat, .card-uncertain {{ padding: 1.5rem 1rem; }}
    }}
    @media (max-width: 640px) {{
        .card {{ padding: 1.15rem; }}
        [data-testid="stTabs"] [role="tab"] {{ padding: 0.5rem 0.8rem; font-size: 0.84rem; }}
        [data-testid="stMetricValue"] {{ font-size: 1.3rem !important; }}
    }}
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


_SAMPLE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _list_sample_images() -> list[Path]:
    """Return the bundled demo images under demo/demo_data, sorted by name."""
    if not DEMO_DATA_DIR.is_dir():
        return []
    return sorted(
        p for p in DEMO_DATA_DIR.iterdir() if p.suffix.lower() in _SAMPLE_EXTS
    )


def _sample_caption(path: Path) -> str:
    """Human-friendly button label for a bundled sample image."""
    stem = path.stem.lower()
    if stem.startswith(("not_cat", "notcat", "dog")):
        return "Dog"
    if stem.startswith("cat"):
        return "Cat"
    return path.stem


def _set_active_image(image_bytes: bytes, filename: str) -> None:
    """Make `image_bytes` the image to classify; reset stale prediction state.

    No-op when the bytes are unchanged, so a rerun does not re-trigger
    inference for an image that is already classified.
    """
    if image_bytes != st.session_state.last_image_bytes:
        st.session_state.last_image_bytes = image_bytes
        st.session_state.last_filename = filename
        st.session_state.last_prediction = None
        st.session_state.feedback_submitted = False


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


def _confidence_gauge(
    confidence: float, label: str, uncertain: bool = False
) -> go.Figure:
    bar_color = WARNING if uncertain else (TEAL if label == "cat" else DANGER)
    pct = round(confidence * 100, 1)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={
                "suffix": "%",
                "font": {
                    "size": 42,
                    "color": TEXT_PRIMARY,
                    "family": "Inter, sans-serif",
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 0,
                    "ticklen": 0,
                    "tickfont": {"color": TEXT_SECONDARY, "size": 9},
                },
                "bar": {"color": bar_color, "thickness": 0.34},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(239,68,68,0.10)"},
                    {"range": [50, 80], "color": "rgba(245,158,11,0.10)"},
                    {"range": [80, 100], "color": "rgba(6,214,160,0.10)"},
                ],
                "threshold": {
                    "line": {"color": TEXT_SECONDARY, "width": 2},
                    "thickness": 0.75,
                    "value": 80,
                },
            },
        )
    )
    fig.update_layout(
        height=212,
        margin=dict(t=26, b=8, l=30, r=30),
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
            width=0.5,
            text=counts,
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, size=13, family="Inter, sans-serif"),
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(t=16, b=10, l=20, r=20),
        xaxis=dict(showgrid=False, tickfont=dict(color=TEXT_SECONDARY)),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            tickfont=dict(color=TEXT_SECONDARY),
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
        <p style="color:{TEXT_SECONDARY}; margin-bottom:0.5rem;">
            Pick a bundled sample or upload your own — the ResNet50 model classifies it
            in real time via the FastAPI backend.
        </p>
        <p style="color:{TEXT_SECONDARY}; font-size:0.82rem; margin-bottom:1.5rem;">
            ℹ️ Trained on the Oxford-IIIT Pet dataset, the model only tells
            <b>cats</b> from <b>dogs</b>. Images that are neither (a road, a car…)
            are out-of-distribution and may be misclassified — even confidently.
        </p>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_result = st.columns([1, 1], gap="large")

    # ── Left: Upload + preview ───────────────────────────────────────────

    with col_upload:
        input_mode = st.radio(
            "Image source",
            options=["🖼️ Sample images", "📤 Upload your own"],
            horizontal=True,
            label_visibility="collapsed",
            key="input_mode",
        )

        if input_mode == "📤 Upload your own":
            st.markdown(
                f'<div style="font-weight:600; color:{TEXT_PRIMARY}; '
                'margin-bottom:0.5rem;">📁 Upload Image</div>',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Drop an image here (JPG, PNG, WebP)",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="visible",
            )
            if uploaded_file is not None:
                _set_active_image(uploaded_file.getvalue(), uploaded_file.name)
        else:
            samples = _list_sample_images()
            if not samples:
                st.markdown(
                    '<div class="warning-banner">⚠️ No sample images found in '
                    "<code>demo/demo_data</code> — switch to upload mode.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="font-size:0.82rem; color:{TEXT_SECONDARY}; '
                    'margin-bottom:0.6rem;">Click a sample to classify it — '
                    "these are cats and dogs, the two classes the model knows."
                    "</div>",
                    unsafe_allow_html=True,
                )
                per_row = 3
                for start in range(0, len(samples), per_row):
                    cols = st.columns(per_row)
                    for col, path in zip(cols, samples[start : start + per_row]):
                        with col:
                            st.image(str(path), use_container_width=True)
                            if st.button(
                                _sample_caption(path),
                                key=f"sample_{path.name}",
                                use_container_width=True,
                            ):
                                _set_active_image(path.read_bytes(), path.name)

        # ── Selected image preview + prediction (shared by both modes) ──────
        image_bytes = st.session_state.last_image_bytes
        if image_bytes:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.image(
                Image.open(io.BytesIO(image_bytes)),
                caption=st.session_state.last_filename,
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
                        result = _predict(
                            image_bytes,
                            st.session_state.last_filename or "image.jpg",
                        )
                        st.session_state.last_prediction = result
                        if result["confidence"] < confidence_threshold:
                            st.session_state.uncertain_count += 1
                        st.session_state.prediction_history.append(
                            {
                                "Time": datetime.now().strftime("%H:%M:%S"),
                                "File": st.session_state.last_filename,
                                "Label": result["label"],
                                "Confidence": f'{result["confidence"]:.1%}',
                            }
                        )
                    except httpx.HTTPStatusError as exc:
                        st.error(
                            f"API error {exc.response.status_code}: "
                            f"{exc.response.text[:200]}"
                        )
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
        else:
            st.markdown(
                f"""
                <div style="border: 2px dashed {BORDER}; border-radius:14px;
                            background:{SURFACE_2}; padding:2.5rem 1.5rem;
                            text-align:center; color:{TEXT_SECONDARY}; margin-top:1rem;">
                    <div style="font-size:3rem; margin-bottom:0.75rem;">🖼️</div>
                    <div style="font-weight:600; color:{TEXT_PRIMARY}; margin-bottom:0.4rem;">
                        Pick a sample or upload an image to begin
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

            # Result card — below the confidence threshold the verdict is
            # withheld: the model's raw guess is shown only as a secondary
            # "leaning", never as a definitive cat / not-cat answer.
            if is_uncertain:
                card_class = "card-uncertain"
                emoji = "🤔"
                label_display = "Uncertain"
                chip_cls = "chip-amber"
                chip_text = f"leans {label.replace('_', ' ')}"
            elif label == "cat":
                card_class = "card-cat"
                emoji = "🐱"
                label_display = "It's a Cat!"
                chip_cls = "chip-teal"
                chip_text = "cat"
            else:
                card_class = "card-notcat"
                emoji = "❌"
                label_display = "Not a Cat"
                chip_cls = "chip-red"
                chip_text = "not cat"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="font-size:4rem; line-height:1.1;">{emoji}</div>
                    <div style="font-size:1.8rem; font-weight:800; color:#fff;
                                margin-top:0.5rem; text-shadow:0 2px 8px rgba(0,0,0,0.4);">
                        {label_display}
                    </div>
                    <div style="margin-top:0.4rem;">
                        <span class="chip {chip_cls}">{chip_text}</span>
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
                _confidence_gauge(confidence, label, uncertain=is_uncertain),
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
        # No --output-log: the API's inference_logger is the sole writer of
        # inference_log.csv, so each injected image produces exactly one row.
        cmd = [
            sys.executable,
            "-u",
            str(SIMULATE_SCRIPT),
            "--count",
            str(int(drift_count)),
            "--api-url",
            API_URL,
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
            # Tell the running API to pick up the freshly registered @staging
            # model — keeps the closed-loop demo coherent without a restart.
            try:
                with httpx.Client(timeout=30.0) as client:
                    reload_resp = client.post(f"{API_URL}/internal/reload-model")
                if reload_resp.status_code == 200:
                    st.success(
                        f"🔄 API reloaded model "
                        f"({reload_resp.json().get('source', 'unknown')})."
                    )
                else:
                    st.warning(
                        f"Retraining done, but model reload returned "
                        f"HTTP {reload_resp.status_code}. The API will use the "
                        "new model after its next restart."
                    )
            except Exception as exc:
                st.warning(
                    f"Retraining done, but the model-reload call failed ({exc}). "
                    "Restart the API to serve the new model."
                )
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
