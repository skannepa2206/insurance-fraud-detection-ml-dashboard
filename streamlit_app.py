from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_KEYS = {
    "XGBoost": "xgboost",
    "Random Forest": "random_forest",
}
SECTIONS = ["Overview", "Threshold Logic", "Feature Signals", "Artifacts", "Reproduce"]
THEMES = {
    "Light": {
        "paper": "#f4efe6",
        "paper_2": "#faf7f1",
        "sidebar": "#f1ece3",
        "panel": "rgba(255,255,255,0.72)",
        "panel_strong": "rgba(255,255,255,0.92)",
        "text": "#171717",
        "muted": "#706a62",
        "stroke": "rgba(23,23,23,0.12)",
        "stroke_strong": "rgba(23,23,23,0.28)",
        "accent": "#ff5a52",
        "accent_2": "#141414",
        "grid": "rgba(18,18,18,0.08)",
        "chip": "#f9f4ec",
        "shadow": "0 18px 40px rgba(0,0,0,0.08)",
        "plot_bg": "#fbf8f4",
        "button_text": "#111111",
        "button_bg": "#ffffff",
        "button_primary_bg": "#171717",
        "button_primary_text": "#f7f3eb",
    },
    "Dark": {
        "paper": "#080808",
        "paper_2": "#0f0f10",
        "sidebar": "#101011",
        "panel": "rgba(20,20,22,0.82)",
        "panel_strong": "rgba(22,22,24,0.96)",
        "text": "#f2eee6",
        "muted": "#aca69e",
        "stroke": "rgba(255,255,255,0.12)",
        "stroke_strong": "rgba(255,255,255,0.22)",
        "accent": "#ff5a52",
        "accent_2": "#f2eee6",
        "grid": "rgba(255,255,255,0.08)",
        "chip": "#131315",
        "shadow": "0 18px 40px rgba(0,0,0,0.34)",
        "plot_bg": "#121214",
        "button_text": "#151515",
        "button_bg": "#ffffff",
        "button_primary_bg": "#f4efe7",
        "button_primary_text": "#111111",
    },
}


st.set_page_config(
    page_title="Auto Fraud Dashboard",
    page_icon="AF",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def human_size(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024**2:.2f} MB"


def apply_css(theme_name: str) -> None:
    t = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap');

        :root {{
          --paper: {t["paper"]};
          --paper-2: {t["paper_2"]};
          --sidebar: {t["sidebar"]};
          --panel: {t["panel"]};
          --panel-strong: {t["panel_strong"]};
          --text: {t["text"]};
          --muted: {t["muted"]};
          --stroke: {t["stroke"]};
          --stroke-strong: {t["stroke_strong"]};
          --accent: {t["accent"]};
          --accent-2: {t["accent_2"]};
          --grid: {t["grid"]};
          --chip: {t["chip"]};
          --shadow: {t["shadow"]};
          --plot-bg: {t["plot_bg"]};
          --button-text: {t["button_text"]};
          --button-bg: {t["button_bg"]};
          --button-primary-bg: {t["button_primary_bg"]};
          --button-primary-text: {t["button_primary_text"]};
        }}

        html, body, [class*="st-"] {{
          font-family: 'Space Grotesk', sans-serif;
        }}

        .stApp {{
          color: var(--text);
          background:
            radial-gradient(circle at 1px 1px, var(--grid) 1px, transparent 0),
            linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.01)),
            var(--paper);
          background-size: 28px 28px, auto, auto;
        }}

        [data-testid="stSidebar"] {{
          background: linear-gradient(180deg, var(--sidebar) 0%, var(--paper-2) 100%);
          border-right: 1px solid var(--stroke);
        }}

        [data-testid="stHeader"] {{
          background: transparent;
        }}

        [data-testid="stToolbar"] {{
          display: none;
        }}

        .block-container {{
          padding-top: 2.2rem;
          padding-bottom: 3rem;
        }}

        h1, h2, h3 {{
          color: var(--text);
          letter-spacing: -0.03em;
        }}

        .sidebar-brand {{
          background: var(--panel);
          border: 1px solid var(--stroke);
          border-radius: 20px;
          padding: 18px 18px 16px 18px;
          box-shadow: var(--shadow);
          margin-bottom: 18px;
        }}

        .sidebar-label {{
          font-family: 'IBM Plex Mono', monospace;
          color: var(--muted);
          font-size: 0.78rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          margin-bottom: 8px;
        }}

        .hero {{
          background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
          border: 1px solid var(--stroke);
          border-radius: 28px;
          padding: 34px 32px 28px 32px;
          box-shadow: var(--shadow);
          text-align: center;
          margin-bottom: 22px;
        }}

        .hero-kicker {{
          font-family: 'IBM Plex Mono', monospace;
          color: var(--muted);
          letter-spacing: 0.08em;
          text-transform: uppercase;
          font-size: 0.82rem;
          margin-bottom: 12px;
        }}

        .hero-title {{
          font-size: clamp(2.7rem, 5vw, 4.4rem);
          line-height: 0.98;
          margin: 0;
          color: var(--text);
        }}

        .hero-subtitle {{
          margin: 18px auto 16px auto;
          max-width: 860px;
          color: var(--muted);
          font-size: 1.2rem;
          line-height: 1.55;
        }}

        .hero-divider {{
          width: 140px;
          height: 2px;
          margin: 0 auto;
          background: linear-gradient(90deg, transparent, var(--accent), transparent);
        }}

        .chip-note {{
          background: var(--chip);
          border: 1px solid var(--stroke);
          border-radius: 18px;
          padding: 18px 20px;
          min-height: 92px;
          box-shadow: var(--shadow);
          display: flex;
          align-items: center;
          justify-content: center;
          text-align: center;
          color: var(--text);
          font-size: 1rem;
        }}

        .metric-card {{
          background: var(--panel);
          border: 1px solid var(--stroke);
          border-radius: 22px;
          padding: 18px 18px 16px 18px;
          box-shadow: var(--shadow);
          min-height: 128px;
        }}

        .metric-label {{
          font-family: 'IBM Plex Mono', monospace;
          color: var(--muted);
          letter-spacing: 0.08em;
          text-transform: uppercase;
          font-size: 0.76rem;
          margin-bottom: 14px;
        }}

        .metric-value {{
          font-size: 2rem;
          line-height: 1;
          font-weight: 700;
          color: var(--text);
          margin-bottom: 8px;
        }}

        .metric-meta {{
          color: var(--muted);
          font-size: 0.94rem;
        }}

        .section-card {{
          background: var(--panel);
          border: 1px solid var(--stroke);
          border-radius: 24px;
          padding: 18px 18px 16px 18px;
          box-shadow: var(--shadow);
          margin-bottom: 18px;
        }}

        .section-title {{
          color: var(--text);
          font-size: 1.45rem;
          margin: 0 0 0.2rem 0;
        }}

        .section-copy {{
          color: var(--muted);
          font-size: 1rem;
          margin-bottom: 0.4rem;
        }}

        .artifact-card {{
          background: var(--panel);
          border: 1px solid var(--stroke);
          border-radius: 18px;
          padding: 16px;
          box-shadow: var(--shadow);
          min-height: 144px;
        }}

        .artifact-name {{
          font-size: 1rem;
          font-weight: 700;
          color: var(--text);
          margin-bottom: 6px;
          word-break: break-word;
        }}

        .artifact-meta {{
          color: var(--muted);
          font-size: 0.92rem;
          margin-bottom: 14px;
        }}

        div[data-testid="stMetric"] {{
          background: var(--panel);
          border: 1px solid var(--stroke);
          border-radius: 20px;
          padding: 14px 16px;
          box-shadow: var(--shadow);
        }}

        .stButton > button {{
          width: 100%;
          border-radius: 16px;
          border: 1px solid var(--stroke-strong);
          background: var(--button-bg);
          color: var(--button-text);
          font-weight: 600;
          padding: 0.72rem 1rem;
        }}

        .stDownloadButton > button {{
          width: 100%;
          border-radius: 14px;
          border: 1px solid var(--stroke-strong);
          background: var(--button-primary-bg);
          color: var(--button-primary-text);
          font-weight: 700;
        }}

        .stSelectbox label, .stTextInput label, .stRadio label {{
          color: var(--muted) !important;
          font-family: 'IBM Plex Mono', monospace !important;
          font-size: 0.76rem !important;
          letter-spacing: 0.08em !important;
          text-transform: uppercase !important;
        }}

        div[data-baseweb="select"] > div,
        .stTextInput input {{
          background: var(--panel-strong) !important;
          color: var(--text) !important;
          border-radius: 16px !important;
          border: 1px solid var(--stroke-strong) !important;
        }}

        .stRadio [role="radiogroup"] {{
          gap: 12px;
        }}

        .stDataFrame, [data-testid="stImage"] {{
          border-radius: 20px;
          overflow: hidden;
          border: 1px solid var(--stroke);
        }}

        code, pre {{
          font-family: 'IBM Plex Mono', monospace !important;
        }}

        .note-box {{
          background: var(--chip);
          border: 1px solid var(--stroke);
          border-radius: 18px;
          padding: 16px 18px;
          color: var(--muted);
          margin-top: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def make_metric_card(label: str, value: str, meta: str) -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-meta">{meta}</div>
    </div>
    """


def make_section_header(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
          <h3 class="section-title">{title}</h3>
          <div class="section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_model_comparison(df: pd.DataFrame, theme_name: str) -> go.Figure:
    palette = THEMES[theme_name]
    labels = {
        "random_forest": "Random Forest",
        "catboost": "CatBoost",
        "xgboost": "XGBoost",
        "logistic": "Logistic Regression",
    }
    chart = df.copy()
    chart["label"] = chart["model"].map(labels)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart["label"],
            y=chart["test_average_precision"],
            marker_color=[palette["accent"], "#ff8d84", "#d6a13d", "#767676"],
            text=[f"{v:.3f}" for v in chart["test_average_precision"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Benchmark Comparison",
        yaxis_title="Test Average Precision",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=palette["plot_bg"],
        font=dict(color=palette["text"], family="Space Grotesk"),
        margin=dict(l=20, r=20, t=60, b=30),
        height=360,
    )
    fig.update_yaxes(gridcolor=palette["grid"])
    return fig


def plot_threshold_tradeoff(df: pd.DataFrame, theme_name: str, selected_threshold: float) -> go.Figure:
    palette = THEMES[theme_name]
    fig = go.Figure()
    series = [
        ("precision", palette["accent"]),
        ("recall", palette["accent_2"]),
        ("f1", "#d9a441" if theme_name == "Light" else "#f0c25f"),
    ]
    for metric, color in series:
        fig.add_trace(
            go.Scatter(
                x=df["threshold"],
                y=df[metric],
                mode="lines+markers",
                name=metric.title(),
                line=dict(width=3, color=color),
                marker=dict(size=9),
            )
        )
    fig.add_vline(x=selected_threshold, line_dash="dot", line_color=palette["muted"])
    fig.update_layout(
        title="Threshold Tradeoff",
        xaxis_title="Probability Threshold",
        yaxis_title="Score",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=palette["plot_bg"],
        font=dict(color=palette["text"], family="Space Grotesk"),
        margin=dict(l=20, r=20, t=60, b=30),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(range=[0, 1], gridcolor=palette["grid"])
    fig.update_xaxes(gridcolor=palette["grid"])
    return fig


def plot_feature_importance(df: pd.DataFrame, theme_name: str) -> go.Figure:
    palette = THEMES[theme_name]
    chart = df.head(10).sort_values("importance", ascending=True)
    fig = px.bar(
        chart,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#3c3c3c", palette["accent"]],
    )
    fig.update_layout(
        title="Top Feature Signals",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=palette["plot_bg"],
        font=dict(color=palette["text"], family="Space Grotesk"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=420,
        coloraxis_showscale=False,
    )
    fig.update_xaxes(gridcolor=palette["grid"])
    fig.update_yaxes(gridcolor=palette["grid"])
    return fig


def show_overview(
    theme_name: str,
    summary: dict,
    comparison_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    model_label: str,
    model_key: str,
) -> None:
    test_metrics = summary["test_metrics_at_selected_threshold"]
    top_model = comparison_df.sort_values("test_average_precision", ascending=False).iloc[0]

    cols = st.columns(5)
    cards = [
        ("Test Average Precision", format_metric(summary["test_average_precision"]), f"Selected model: {model_label}"),
        ("ROC-AUC", format_metric(summary["test_roc_auc"]), "Held-out test evaluation"),
        ("Chosen Threshold", format_metric(test_metrics["threshold"]), "Selected on validation F1"),
        ("Claims Flagged", f"{int(test_metrics['flagged_claims'])}", f"Out of {summary['test_shape'][0]} test claims"),
        ("Fraud Rate", format_pct(summary["test_fraud_rate"]), "Positive class prevalence"),
    ]
    for column, (label, value, meta) in zip(cols, cards):
        with column:
            st.markdown(make_metric_card(label, value, meta), unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])
    with left:
        make_section_header(
            "Benchmark snapshot",
            "The assignment-aligned XGBoost result is shown throughout the dashboard, while the benchmark chart keeps the stronger Random Forest run visible for comparison.",
        )
        st.plotly_chart(plot_model_comparison(comparison_df, theme_name), use_container_width=True)
    with right:
        make_section_header(
            "Selected model summary",
            f"{model_label} was evaluated with a validation-selected threshold. The strongest overall test AP in the benchmark table was {model_label if top_model['model'] == model_key else top_model['model'].replace('_', ' ').title()}.",
        )
        c1, c2 = st.columns(2)
        c1.metric("Precision", format_metric(test_metrics["precision"]))
        c2.metric("Recall", format_metric(test_metrics["recall"]))
        c3, c4 = st.columns(2)
        c3.metric("F1", format_metric(test_metrics["f1"]))
        c4.metric("Balanced Accuracy", format_metric(test_metrics["balanced_accuracy"]))
        st.markdown(
            '<div class="note-box">This dashboard keeps the XGBoost narrative front and center because it matches the assignment prompt, but it also preserves the benchmark evidence that model choice should be validated empirically.</div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.15, 1])
    with left:
        make_section_header(
            "Precision-recall evidence",
            "This static figure is the same artifact used in the paper and shows validation versus test precision-recall behavior.",
        )
        st.image(str(ARTIFACT_DIR / f"{model_key}_precision_recall_curve.png"), use_container_width=True)
    with right:
        make_section_header(
            "Threshold behavior",
            "Lower thresholds catch more fraud but increase manual review volume. The selected line is the validation-based operating point used in the report.",
        )
        st.plotly_chart(
            plot_threshold_tradeoff(threshold_df, theme_name, float(test_metrics["threshold"])),
            use_container_width=True,
        )

    conf_cols = st.columns(4)
    confusion = [
        ("True Positives", str(int(test_metrics["tp"])), "Fraud cases caught"),
        ("False Positives", str(int(test_metrics["fp"])), "Legitimate claims flagged"),
        ("False Negatives", str(int(test_metrics["fn"])), "Fraud cases missed"),
        ("True Negatives", str(int(test_metrics["tn"])), "Legitimate claims cleared"),
    ]
    for col, (label, value, meta) in zip(conf_cols, confusion):
        with col:
            st.markdown(make_metric_card(label, value, meta), unsafe_allow_html=True)


def show_threshold_logic(theme_name: str, summary: dict, threshold_df: pd.DataFrame) -> None:
    test_metrics = summary["test_metrics_at_selected_threshold"]
    make_section_header(
        "Threshold selection logic",
        "The threshold was selected on the validation split by maximizing F1 rather than using the default 0.50 cutoff. This makes the operating point explainable in both model and business terms.",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected threshold", format_metric(test_metrics["threshold"]))
    c2.metric("Validation F1", format_metric(summary["validation_metrics_at_selected_threshold"]["f1"]))
    c3.metric("Test F2", format_metric(test_metrics["f2"]))

    left, right = st.columns([1.1, 0.9])
    with left:
        st.plotly_chart(
            plot_threshold_tradeoff(threshold_df, theme_name, float(test_metrics["threshold"])),
            use_container_width=True,
        )
    with right:
        st.markdown(
            """
            <div class="section-card">
              <h3 class="section-title">Interpretation</h3>
              <div class="section-copy">
                At a lower threshold, recall improves but the investigation queue grows quickly. At a higher threshold, precision improves but the model misses more fraud. The chosen threshold keeps that tradeoff visible and defensible.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            threshold_df.assign(
                threshold=threshold_df["threshold"].map(lambda x: f"{x:.3f}"),
                precision=threshold_df["precision"].map(lambda x: f"{x:.3f}"),
                recall=threshold_df["recall"].map(lambda x: f"{x:.3f}"),
                f1=threshold_df["f1"].map(lambda x: f"{x:.3f}"),
            ),
            use_container_width=True,
            hide_index=True,
        )


def show_feature_signals(theme_name: str, feature_df: pd.DataFrame, model_label: str, model_key: str) -> None:
    make_section_header(
        "Feature signals",
        f"The chart combines the saved feature importance artifact with an interactive view so the strongest model signals are easy to inspect for {model_label}.",
    )
    left, right = st.columns([1.05, 0.95])
    with left:
        st.plotly_chart(plot_feature_importance(feature_df, theme_name), use_container_width=True)
    with right:
        st.image(str(ARTIFACT_DIR / f"{model_key}_top10_feature_importance.png") if model_key == "xgboost" else str(ARTIFACT_DIR / f"{model_key}_precision_recall_curve.png"), use_container_width=True)
        if model_key != "xgboost":
            st.markdown(
                '<div class="note-box">Only the XGBoost run has a saved top-10 static feature image in the artifact bundle, so the Random Forest view keeps the interactive chart as the primary feature-importance source.</div>',
                unsafe_allow_html=True,
            )

    st.dataframe(
        feature_df.head(12).assign(importance=feature_df["importance"].map(lambda x: f"{x:.4f}")),
        use_container_width=True,
        hide_index=True,
    )


def show_artifacts(search_term: str) -> None:
    make_section_header(
        "Artifacts and outputs",
        "These are the exact files that should go into the clean submission repo. They are enough to demonstrate the work without shipping the full course dataset or unrelated project files.",
    )
    files = sorted(ARTIFACT_DIR.iterdir())
    if search_term:
        files = [path for path in files if search_term.lower() in path.name.lower()]

    if not files:
        st.info("No artifacts matched the current search term.")
        return

    for start in range(0, len(files), 3):
        row = files[start : start + 3]
        cols = st.columns(3)
        for col, path in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="artifact-card">
                      <div class="artifact-name">{path.name}</div>
                      <div class="artifact-meta">{human_size(path)} · {path.suffix.upper().lstrip('.')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.download_button(
                    label=f"Download {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/octet-stream",
                    key=f"download-{path.name}",
                )


def show_reproduce() -> None:
    make_section_header(
        "Reproduction guide",
        "The repo is intentionally small: scripts, artifacts, the Streamlit app, and repo metadata. The original course dataset can stay local or be added later if the repo is private and redistribution is allowed.",
    )
    st.code(
        "pip install -r requirements.txt\n"
        "streamlit run streamlit_app.py\n\n"
        "# If the datasets are available locally\n"
        "python scripts/fraud_detection_assignment.py --model xgboost\n"
        "python scripts/generate_submission_visuals.py",
        language="bash",
    )
    st.markdown(
        """
        <div class="note-box">
          Recommended repo contents: <strong>streamlit_app.py</strong>, <strong>scripts/</strong>, <strong>artifacts/</strong>, 
          <strong>README.md</strong>, <strong>requirements.txt</strong>, <strong>.gitignore</strong>, and optional <strong>.streamlit/config.toml</strong>.
          Do not push the whole virtual environment or unrelated healthcare project files.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    if "theme_name" not in st.session_state:
        st.session_state["theme_name"] = "Light"
    if "section" not in st.session_state:
        st.session_state["section"] = "Overview"

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
              <div class="sidebar-label">Assignment Repo</div>
              <h2 style="margin:0; font-size:2rem;">Auto Fraud</h2>
              <p style="margin:8px 0 0 0; color:var(--muted);">Clean submission bundle with code, outputs, and a reviewer-friendly dashboard.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Reset View", use_container_width=True):
            st.session_state["section"] = "Overview"

        artifact_search = st.text_input("Search artifacts", placeholder="Search files")
        theme_name = st.radio("Theme", ["Light", "Dark"], horizontal=True, key="theme_name")
        model_label = st.selectbox("Model", list(MODEL_KEYS.keys()), index=0)
        selected_section = st.radio("View", SECTIONS, index=SECTIONS.index(st.session_state["section"]))
        st.session_state["section"] = selected_section

        st.markdown('<div class="sidebar-label" style="margin-top:18px;">Submission</div>', unsafe_allow_html=True)
        st.markdown(
            "Use the repo link in the written submission and keep the text paper focused on the narrative, visuals, and results."
        )

    apply_css(theme_name)

    model_key = MODEL_KEYS[model_label]
    summary = load_json(ARTIFACT_DIR / f"{model_key}_summary.json")
    threshold_df = load_csv(ARTIFACT_DIR / f"{model_key}_threshold_table.csv")
    feature_df = load_csv(ARTIFACT_DIR / f"{model_key}_feature_importance.csv")
    comparison_df = load_csv(ARTIFACT_DIR / "model_comparison.csv")

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">Machine Learning Assignment Dashboard</div>
          <h1 class="hero-title">What did I build?</h1>
          <div class="hero-subtitle">
            A local fraud-detection workflow using preprocessed insurance claims data, validation-based threshold selection,
            and a reviewer-friendly dashboard that makes the model evidence easy to inspect.
          </div>
          <div class="hero-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chip_cols = st.columns(3)
    chip_copy = [
        "I trained and benchmarked multiple models, then kept the XGBoost narrative for prompt alignment.",
        "I evaluated performance with precision-recall because fraud is rare and accuracy would be misleading.",
        "I selected the threshold on validation F1 so the operating point could be justified in business terms.",
    ]
    for col, text in zip(chip_cols, chip_copy):
        with col:
            st.markdown(f'<div class="chip-note">{text}</div>', unsafe_allow_html=True)

    nav_cols = st.columns(len(SECTIONS))
    for col, section_name in zip(nav_cols, SECTIONS):
        with col:
            if st.button(section_name, key=f"nav-{section_name}"):
                st.session_state["section"] = section_name

    st.write("")

    if st.session_state["section"] == "Overview":
        show_overview(theme_name, summary, comparison_df, threshold_df, feature_df, model_label, model_key)
    elif st.session_state["section"] == "Threshold Logic":
        show_threshold_logic(theme_name, summary, threshold_df)
    elif st.session_state["section"] == "Feature Signals":
        show_feature_signals(theme_name, feature_df, model_label, model_key)
    elif st.session_state["section"] == "Artifacts":
        show_artifacts(artifact_search)
    else:
        show_reproduce()


if __name__ == "__main__":
    main()
