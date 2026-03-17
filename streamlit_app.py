from __future__ import annotations

import html
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
    "CatBoost": "catboost",
    "Logistic Regression": "logistic",
}
THEMES = {
    "Light": {
        "paper": "#f4efe6",
        "paper_2": "#fbf8f3",
        "sidebar": "#efe8dc",
        "panel": "rgba(255,255,255,0.86)",
        "panel_strong": "#fffdfa",
        "panel_soft": "#fff9f1",
        "text": "#151515",
        "label": "#5a534a",
        "muted": "#645c53",
        "subtle": "#7a736a",
        "stroke": "rgba(24,24,24,0.14)",
        "stroke_strong": "rgba(24,24,24,0.28)",
        "accent": "#ff5a52",
        "accent_soft": "#fff1ef",
        "accent_text": "#151515",
        "grid": "rgba(18,18,18,0.08)",
        "plot_bg": "#fbf8f4",
        "shadow": "0 18px 40px rgba(0,0,0,0.08)",
        "chip": "#f8f2e9",
        "tab_bg": "rgba(255,255,255,0.62)",
        "tab_active_bg": "#171717",
        "tab_active_text": "#f7f3eb",
    },
    "Dark": {
        "paper": "#0b0b0c",
        "paper_2": "#111113",
        "sidebar": "#121214",
        "panel": "rgba(19,19,21,0.94)",
        "panel_strong": "#17181b",
        "panel_soft": "#1b1c1f",
        "text": "#f4efe7",
        "label": "#ddd6cc",
        "muted": "#cbc4ba",
        "subtle": "#a79f95",
        "stroke": "rgba(255,255,255,0.14)",
        "stroke_strong": "rgba(255,255,255,0.26)",
        "accent": "#ff5a52",
        "accent_soft": "#2a1414",
        "accent_text": "#f7f3eb",
        "grid": "rgba(255,255,255,0.08)",
        "plot_bg": "#141417",
        "shadow": "0 18px 40px rgba(0,0,0,0.34)",
        "chip": "#151518",
        "tab_bg": "rgba(255,255,255,0.04)",
        "tab_active_bg": "#f4efe7",
        "tab_active_text": "#111111",
    },
}


st.set_page_config(
    page_title="Insurance Fraud Risk Dashboard",
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


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def info_badge(copy: str) -> str:
    return f'<span class="info-dot" title="{html.escape(copy)}">i</span>'


def metric_card(label: str, value: str, tooltip: str, compact: bool = False) -> str:
    compact_class = " metric-card-compact" if compact else ""
    return f"""
    <div class="metric-card{compact_class}">
      <div class="metric-head">
        <div class="metric-name">{label}</div>
        {info_badge(tooltip)}
      </div>
      <div class="metric-value">{value}</div>
    </div>
    """


def feature_table_html(df: pd.DataFrame) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(str(row.feature))}</td><td>{row.importance:.4f}</td></tr>"
        for row in df.itertuples(index=False)
    )
    return f"""
    <div class="table-shell">
      <table class="data-table">
        <thead>
          <tr><th>Feature</th><th>Importance</th></tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
    """


def apply_css(theme_name: str) -> None:
    t = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

        :root {{
          --paper: {t["paper"]};
          --paper-2: {t["paper_2"]};
          --sidebar: {t["sidebar"]};
          --panel: {t["panel"]};
          --panel-strong: {t["panel_strong"]};
          --panel-soft: {t["panel_soft"]};
          --text: {t["text"]};
          --label: {t["label"]};
          --muted: {t["muted"]};
          --subtle: {t["subtle"]};
          --stroke: {t["stroke"]};
          --stroke-strong: {t["stroke_strong"]};
          --accent: {t["accent"]};
          --accent-soft: {t["accent_soft"]};
          --accent-text: {t["accent_text"]};
          --grid: {t["grid"]};
          --plot-bg: {t["plot_bg"]};
          --shadow: {t["shadow"]};
          --chip: {t["chip"]};
          --tab-bg: {t["tab_bg"]};
          --tab-active-bg: {t["tab_active_bg"]};
          --tab-active-text: {t["tab_active_text"]};
        }}

        html, body {{
          font-family: 'Space Grotesk', sans-serif;
        }}

        .stApp,
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp input,
        .stApp textarea,
        .stApp [data-testid="stMarkdownContainer"] *,
        .stApp [data-testid="stCaptionContainer"] *,
        .stApp [data-testid="stMetricLabel"] *,
        .stApp [data-testid="stMetricValue"] * {{
          font-family: 'Space Grotesk', sans-serif;
        }}

        .material-icons,
        .material-icons-round,
        .material-icons-outlined,
        .material-symbols-rounded,
        .material-symbols-outlined,
        [class*="material-symbol"] {{
          font-family: "Material Symbols Rounded" !important;
          font-feature-settings: 'liga' !important;
          -webkit-font-feature-settings: 'liga' !important;
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

        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        button[aria-label="Open sidebar"],
        button[aria-label="Close sidebar"] {{
          display: none !important;
        }}

        .block-container {{
          padding-top: 2rem;
          padding-bottom: 3rem;
        }}

        .brand-card {{
          background: var(--panel);
          border: 1px solid var(--stroke);
          border-radius: 22px;
          padding: 18px 18px 16px 18px;
          box-shadow: var(--shadow);
          margin-bottom: 18px;
        }}

        .sidebar-kicker {{
          font-family: 'IBM Plex Mono', monospace;
          color: var(--label);
          font-size: 0.78rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          margin-bottom: 8px;
        }}

        .brand-title {{
          margin: 0;
          color: var(--text) !important;
          font-size: 2rem;
          line-height: 1.05;
        }}

        .brand-copy {{
          color: var(--muted) !important;
          margin: 10px 0 0 0;
          font-size: 0.96rem;
          line-height: 1.5;
        }}

        .hero {{
          background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
          border: 1px solid var(--stroke);
          border-radius: 30px;
          padding: 32px 34px 26px 34px;
          box-shadow: var(--shadow);
          margin-bottom: 18px;
        }}

        .hero-kicker {{
          font-family: 'IBM Plex Mono', monospace;
          color: var(--label);
          letter-spacing: 0.08em;
          text-transform: uppercase;
          font-size: 0.82rem;
          margin-bottom: 12px;
          text-align: center;
        }}

        .hero-title {{
          font-size: clamp(2.4rem, 4.2vw, 4rem);
          line-height: 1;
          margin: 0;
          color: var(--text) !important;
          text-align: center;
        }}

        .hero-subtitle {{
          margin: 16px auto 16px auto;
          max-width: 920px;
          text-align: center;
          color: var(--muted) !important;
          font-size: 1.04rem;
          line-height: 1.5;
        }}

        .hero-divider {{
          width: 170px;
          height: 2px;
          margin: 0 auto;
          background: linear-gradient(90deg, transparent, var(--accent), transparent);
        }}

        .snapshot-chip {{
          background: var(--chip);
          border: 1px solid var(--stroke);
          border-radius: 18px;
          padding: 16px 18px;
          box-shadow: var(--shadow);
          min-height: 92px;
          display: flex;
          flex-direction: column;
          justify-content: center;
        }}

        .snapshot-head {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          margin-bottom: 8px;
        }}

        .snapshot-label {{
          font-family: 'IBM Plex Mono', monospace;
          color: var(--label);
          font-size: 0.76rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          margin-bottom: 0;
        }}

        .snapshot-value {{
          color: var(--text) !important;
          font-size: 0.98rem;
          line-height: 1.4;
        }}

        .metric-card {{
          background: var(--panel);
          border: 1px solid var(--stroke);
          border-radius: 22px;
          padding: 16px 18px 18px 18px;
          box-shadow: var(--shadow);
          min-height: 132px;
          margin-bottom: 10px;
        }}

        .metric-card-compact {{
          min-height: 120px;
        }}

        .metric-head {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          margin-bottom: 12px;
        }}

        .metric-name {{
          color: var(--label) !important;
          font-size: 0.96rem;
          line-height: 1.2;
          font-weight: 500;
        }}

        .metric-value {{
          color: var(--text) !important;
          font-size: clamp(1.85rem, 2.3vw, 2.55rem);
          line-height: 1;
          font-weight: 700;
          letter-spacing: -0.03em;
        }}

        .section-card {{
          background: var(--panel);
          border: 1px solid var(--stroke);
          border-radius: 24px;
          padding: 18px 18px 16px 18px;
          box-shadow: var(--shadow);
          margin-bottom: 22px;
        }}

        .section-row {{
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 8px;
        }}

        .section-title {{
          color: var(--text) !important;
          font-size: 1.2rem;
          font-weight: 700;
          margin: 0;
        }}

        .section-copy {{
          color: var(--muted) !important;
          font-size: 0.94rem;
          line-height: 1.5;
          margin: 0;
        }}

        .info-dot {{
          width: 22px;
          height: 22px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          border-radius: 999px;
          border: 1px solid var(--stroke-strong);
          color: var(--label);
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.78rem;
          cursor: help;
          background: var(--panel-soft);
        }}

        .info-dot:hover {{
          border-color: var(--accent);
          background: var(--accent-soft);
          color: var(--text) !important;
        }}

        .stTabs [data-baseweb="tab-border"] {{
          background: var(--stroke) !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
          gap: 14px;
        }}

        .stTabs [data-baseweb="tab"] {{
          min-height: 54px;
          background: var(--panel) !important;
          border: 1px solid var(--stroke) !important;
          border-radius: 18px 18px 0 0 !important;
          color: var(--text) !important;
          padding: 10px 20px 11px 20px !important;
          font-weight: 700 !important;
          opacity: 1 !important;
          box-shadow: none !important;
        }}

        .stTabs [data-baseweb="tab"] *,
        .stTabs [data-baseweb="tab"] p,
        .stTabs [data-baseweb="tab"] span,
        .stTabs [data-baseweb="tab"] div {{
          color: var(--text) !important;
          -webkit-text-fill-color: var(--text) !important;
          opacity: 1 !important;
        }}

        .stTabs [data-baseweb="tab"]:hover {{
          background: var(--accent-soft);
          color: var(--text) !important;
          border-color: var(--accent) !important;
        }}

        .stTabs [data-baseweb="tab"]:hover *,
        .stTabs [data-baseweb="tab"]:hover p,
        .stTabs [data-baseweb="tab"]:hover span,
        .stTabs [data-baseweb="tab"]:hover div {{
          color: var(--text) !important;
          -webkit-text-fill-color: var(--text) !important;
        }}

        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
          background: var(--tab-active-bg) !important;
          color: var(--tab-active-text) !important;
          border-color: var(--tab-active-bg) !important;
          opacity: 1 !important;
        }}

        .stTabs [data-baseweb="tab"][aria-selected="true"] *,
        .stTabs [data-baseweb="tab"][aria-selected="true"] p,
        .stTabs [data-baseweb="tab"][aria-selected="true"] span,
        .stTabs [data-baseweb="tab"][aria-selected="true"] div {{
          color: var(--tab-active-text) !important;
          -webkit-text-fill-color: var(--tab-active-text) !important;
          opacity: 1 !important;
        }}

        .stTabs [data-baseweb="tab-highlight"] {{
          background: transparent !important;
        }}

        .stSelectbox label,
        .stRadio label,
        .stToggle label,
        .stExpander label {{
          color: var(--label) !important;
          font-family: 'IBM Plex Mono', monospace !important;
          font-size: 0.76rem !important;
          letter-spacing: 0.08em !important;
          text-transform: uppercase !important;
        }}

        div[data-testid="stRadio"] [role="radiogroup"] {{
          gap: 10px;
        }}

        div[data-testid="stRadio"] label {{
          background: var(--panel) !important;
          border: 1px solid var(--stroke) !important;
          border-radius: 14px !important;
          padding: 10px 12px !important;
          margin-bottom: 8px !important;
          transition: border-color 0.15s ease, background 0.15s ease;
        }}

        div[data-testid="stRadio"] label:hover {{
          border-color: var(--accent) !important;
          background: var(--panel-soft) !important;
        }}

        div[data-testid="stRadio"] label:has(input:checked) {{
          background: var(--accent-soft) !important;
          border-color: var(--accent) !important;
        }}

        div[data-testid="stRadio"] label p {{
          color: var(--text) !important;
          font-size: 0.98rem !important;
          line-height: 1.3 !important;
        }}

        div[data-testid="stRadio"] input[type="radio"] {{
          accent-color: var(--accent) !important;
        }}

        div[data-baseweb="select"] > div {{
          background: var(--panel-strong) !important;
          color: var(--text) !important;
          border-radius: 16px !important;
          border: 1px solid var(--stroke-strong) !important;
        }}

        div[data-baseweb="select"] * {{
          color: var(--text) !important;
        }}

        div[data-baseweb="popover"] {{
          background: var(--panel-strong) !important;
          color: var(--text) !important;
          border: 1px solid var(--stroke-strong) !important;
          border-radius: 16px !important;
          box-shadow: var(--shadow) !important;
        }}

        div[data-baseweb="popover"] *,
        div[data-baseweb="menu"] *,
        div[role="listbox"] {{
          color: var(--text) !important;
        }}

        div[data-baseweb="menu"],
        ul[role="listbox"],
        div[role="listbox"] {{
          background: var(--panel-strong) !important;
        }}

        li[role="option"],
        div[role="option"] {{
          color: var(--text) !important;
          background: var(--panel-strong) !important;
        }}

        li[role="option"]:hover,
        div[role="option"]:hover,
        li[role="option"][aria-selected="true"],
        div[role="option"][aria-selected="true"] {{
          background: var(--accent-soft) !important;
          color: var(--text) !important;
        }}

        .stTextInput label {{
          color: var(--label) !important;
        }}

        .stTextInput input {{
          background: var(--panel-strong) !important;
          color: var(--text) !important;
          border-radius: 16px !important;
          border: 1px solid var(--stroke-strong) !important;
        }}

        .stToggle > label span {{
          color: var(--text) !important;
        }}

        .stToggle [data-baseweb="switch"] > div {{
          background: var(--panel-soft) !important;
          border: 1px solid var(--stroke-strong) !important;
        }}

        .stToggle [data-baseweb="switch"] [data-testid="stMarkdownContainer"] {{
          color: var(--text) !important;
        }}

        [data-testid="stWidgetLabel"] {{
          color: var(--label) !important;
          font-family: 'IBM Plex Mono', monospace !important;
          letter-spacing: 0.08em !important;
          text-transform: uppercase !important;
        }}

        [data-baseweb="slider"] * {{
          color: var(--text) !important;
        }}

        [data-baseweb="slider"] [role="slider"] {{
          box-shadow: none !important;
        }}

        [data-testid="stSelectSlider"] * {{
          color: var(--text) !important;
        }}

        [data-testid="stCaptionContainer"] p,
        .utility-note {{
          color: var(--subtle) !important;
          font-size: 0.98rem;
          line-height: 1.45;
        }}

        [data-testid="stExpander"] {{
          border: 1px solid var(--stroke);
          border-radius: 18px;
          background: var(--panel);
        }}

        [data-testid="stExpander"] summary {{
          color: var(--text) !important;
        }}

        [data-testid="stImage"], .stPlotlyChart, .stDataFrame {{
          border: 1px solid var(--stroke);
          border-radius: 22px;
          overflow: hidden;
        }}

        .table-shell {{
          background: var(--panel);
          border: 1px solid var(--stroke);
          border-radius: 22px;
          overflow: hidden;
          box-shadow: var(--shadow);
        }}

        .data-table {{
          width: 100%;
          border-collapse: collapse;
        }}

        .data-table th {{
          background: var(--panel-soft);
          color: var(--label);
          text-align: left;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.76rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          padding: 14px 16px;
        }}

        .data-table td {{
          color: var(--text);
          padding: 14px 16px;
          border-top: 1px solid var(--stroke);
          font-size: 0.98rem;
          line-height: 1.45;
        }}

        .data-table tbody tr:nth-child(even) td {{
          background: var(--panel-soft);
        }}

        .stDataFrame *,
        [data-testid="stDataFrame"] *,
        [data-testid="stImage"] * {{
          color: var(--text) !important;
        }}

        .mini-note {{
          color: var(--subtle);
          font-size: 0.94rem;
          line-height: 1.55;
          margin-top: 6px;
        }}

        .stack-gap {{
          height: 24px;
        }}

        .stack-gap-sm {{
          height: 10px;
        }}

        .use-card {{
          background: var(--panel);
          border: 1px solid var(--stroke);
          border-radius: 20px;
          padding: 14px 16px;
          box-shadow: var(--shadow);
        }}

        .use-card ul {{
          margin: 0;
          padding-left: 1.1rem;
          color: var(--muted);
          line-height: 1.65;
        }}

        .use-card li {{
          color: var(--muted) !important;
          margin-bottom: 0.4rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, tooltip: str, copy: str | None = None) -> None:
    body = f'<p class="section-copy">{copy}</p>' if copy else ""
    st.markdown(
        f"""
        <div class="section-card">
          <div class="section-row">
            <h3 class="section-title">{title}</h3>
            {info_badge(tooltip)}
          </div>
          {body}
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
            marker_color=[palette["accent"], "#ff8a7d", "#d9b15a", "#8b8b8b"],
            text=[f"{v:.3f}" for v in chart["test_average_precision"]],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(color=palette["text"], size=15),
            hovertemplate="<b>%{x}</b><br>Average precision: %{y:.3f}<extra></extra>",
        )
    )
    max_value = float(chart["test_average_precision"].max())
    fig.update_layout(
        title="Model ranking quality",
        title_font=dict(color=palette["text"], size=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=palette["plot_bg"],
        template="none",
        font=dict(color=palette["text"], family="Space Grotesk"),
        margin=dict(l=20, r=20, t=58, b=20),
        height=360,
        yaxis_title="Test average precision",
    )
    fig.update_yaxes(
        gridcolor=palette["grid"],
        range=[0, max_value * 1.18],
        title_font=dict(color=palette["label"], size=14),
        tickfont=dict(color=palette["label"], size=12),
        zerolinecolor=palette["grid"],
    )
    fig.update_xaxes(
        title_font=dict(color=palette["label"], size=14),
        tickfont=dict(color=palette["label"], size=12),
        linecolor=palette["grid"],
    )
    return fig


def plot_threshold_tradeoff(df: pd.DataFrame, theme_name: str, selected_threshold: float) -> go.Figure:
    palette = THEMES[theme_name]
    fig = go.Figure()
    series = [
        ("precision", palette["accent"]),
        ("recall", palette["text"]),
        ("f1", "#d9ab43" if theme_name == "Light" else "#f0c562"),
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
                hovertemplate=f"{metric.title()}: %{{y:.3f}}<br>Threshold: %{{x:.3f}}<extra></extra>",
            )
        )
    fig.add_vline(x=selected_threshold, line_dash="dot", line_color=palette["muted"])
    fig.update_layout(
        title="Threshold trade-off",
        title_font=dict(color=palette["text"], size=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=palette["plot_bg"],
        template="none",
        font=dict(color=palette["text"], family="Space Grotesk"),
        margin=dict(l=20, r=20, t=58, b=24),
        height=360,
        xaxis_title="Probability threshold",
        yaxis_title="Score",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(color=palette["text"], size=13),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_yaxes(
        range=[0, 1],
        gridcolor=palette["grid"],
        title_font=dict(color=palette["label"], size=14),
        tickfont=dict(color=palette["label"], size=12),
        zerolinecolor=palette["grid"],
    )
    fig.update_xaxes(
        gridcolor=palette["grid"],
        title_font=dict(color=palette["label"], size=14),
        tickfont=dict(color=palette["label"], size=12),
        linecolor=palette["grid"],
    )
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
        color_continuous_scale=["#494949", palette["accent"]],
    )
    fig.update_layout(
        title="Top driver signals",
        title_font=dict(color=palette["text"], size=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=palette["plot_bg"],
        template="none",
        font=dict(color=palette["text"], family="Space Grotesk"),
        margin=dict(l=20, r=20, t=58, b=20),
        height=430,
        coloraxis_showscale=False,
    )
    fig.update_xaxes(
        gridcolor=palette["grid"],
        title_font=dict(color=palette["label"], size=14),
        tickfont=dict(color=palette["label"], size=12),
        zerolinecolor=palette["grid"],
    )
    fig.update_yaxes(
        gridcolor=palette["grid"],
        title_font=dict(color=palette["label"], size=14),
        tickfont=dict(color=palette["label"], size=12),
    )
    return fig


def nearest_threshold_row(threshold_df: pd.DataFrame, threshold_value: float) -> pd.Series:
    match_index = (threshold_df["threshold"] - threshold_value).abs().idxmin()
    return threshold_df.loc[match_index]


def render_overview(
    theme_name: str,
    summary: dict,
    comparison_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    model_key: str,
) -> None:
    test_metrics = summary["test_metrics_at_selected_threshold"]
    review_rate = test_metrics["flagged_claims"] / summary["test_shape"][0]

    metric_cols = st.columns(4)
    metrics = [
        (
            "Average precision",
            format_metric(summary["test_average_precision"]),
            "Area under the precision-recall curve. Higher is better for rare-event screening.",
        ),
        (
            "Precision",
            format_metric(test_metrics["precision"]),
            "Of the claims flagged for review, the share that were actually fraud.",
        ),
        (
            "Recall",
            format_metric(test_metrics["recall"]),
            "Of all fraud cases in the test set, the share that were correctly identified.",
        ),
        (
            "Review rate",
            format_pct(review_rate),
            "Share of the test portfolio sent to manual review at the selected threshold.",
        ),
    ]
    for col, (label, value, help_text) in zip(metric_cols, metrics):
        with col:
            st.markdown(metric_card(label, value, help_text), unsafe_allow_html=True)

    st.markdown('<div class="stack-gap"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.08, 0.92])
    with left:
        section_header(
            "Screening quality",
            "This view compares the ranking strength of the saved candidate models on the same held-out test set.",
            "See how the selected scoring model ranks suspicious claims relative to the other saved candidates.",
        )
        st.plotly_chart(
            plot_model_comparison(comparison_df, theme_name),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with right:
        section_header(
            "Precision-recall curve",
            "Precision-recall is the main evaluation lens for rare fraud events because it highlights the trade-off between catching fraud and over-flagging clean claims.",
            "This curve shows the balance between fraud capture and review efficiency.",
        )
        st.image(str(ARTIFACT_DIR / f"{model_key}_precision_recall_curve.png"), use_container_width=True)

    st.markdown('<div class="stack-gap"></div>', unsafe_allow_html=True)

    section_header(
        "Review outcome",
        "This summary reflects the selected threshold currently shown in the saved model output.",
        "These counts translate the selected threshold into review workload and missed-risk impact.",
    )
    confusion_cols = st.columns(4)
    confusion = [
        ("Fraud caught", int(test_metrics["tp"]), "Suspicious claims correctly sent to review."),
        ("Extra reviews", int(test_metrics["fp"]), "Legitimate claims that still enter the review queue."),
        ("Fraud missed", int(test_metrics["fn"]), "Suspicious claims that remain unflagged at this threshold."),
        ("Clean claims cleared", int(test_metrics["tn"]), "Legitimate claims left out of review."),
    ]
    for col, (label, value, help_text) in zip(confusion_cols, confusion):
        with col:
            st.markdown(metric_card(label, f"{value}", help_text, compact=True), unsafe_allow_html=True)


def render_threshold_explorer(theme_name: str, summary: dict, threshold_df: pd.DataFrame) -> None:
    selected_threshold = float(summary["test_metrics_at_selected_threshold"]["threshold"])
    choices = sorted(threshold_df["threshold"].tolist())
    threshold_value = st.select_slider(
        "Review threshold",
        options=choices,
        value=selected_threshold,
        format_func=lambda x: f"{x:.3f}",
        help="Move between saved operating points to see how review volume, precision, and recall change.",
    )
    active = nearest_threshold_row(threshold_df, float(threshold_value))
    flagged_rate = active["flagged_claims"] / summary["test_shape"][0]

    metric_cols = st.columns(5)
    metrics = [
        ("Threshold", format_metric(active["threshold"]), "Probability cutoff used for this scenario."),
        ("Claims flagged", f"{int(active['flagged_claims'])}", "Number of test claims sent to manual review."),
        ("Flagged rate", format_pct(flagged_rate), "Share of the test set routed to review."),
        ("Precision", format_metric(active["precision"]), "Expected fraud hit rate inside the review queue."),
        ("Recall", format_metric(active["recall"]), "Expected share of fraud cases captured at this setting."),
    ]
    for col, (label, value, help_text) in zip(metric_cols, metrics):
        with col:
            st.markdown(metric_card(label, value, help_text), unsafe_allow_html=True)

    st.markdown('<div class="stack-gap"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.08, 0.92])
    with left:
        section_header(
            "Trade-off curve",
            "The dotted line marks the saved operating point selected during model evaluation.",
            "Use this view to balance review volume against fraud capture.",
        )
        st.plotly_chart(
            plot_threshold_tradeoff(threshold_df, theme_name, selected_threshold),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with right:
        section_header(
            "Scenario summary",
            "Use this panel to understand what the current threshold means in review terms.",
            "This panel converts a score cutoff into queue size and likely review outcome.",
        )
        c1, c2 = st.columns(2)
        c1.markdown(
            metric_card("F1", format_metric(active["f1"]), "Balance between precision and recall.", compact=True),
            unsafe_allow_html=True,
        )
        c2.markdown(
            metric_card("F2", format_metric(active["f2"]), "Recall-weighted variant of F1.", compact=True),
            unsafe_allow_html=True,
        )
        c3, c4 = st.columns(2)
        c3.markdown(
            metric_card("Missed fraud", f"{int(active['fn'])}", "Fraud cases not caught at this threshold.", compact=True),
            unsafe_allow_html=True,
        )
        c4.markdown(
            metric_card(
                "Clean claims cleared",
                f"{int(active['tn'])}",
                "Legitimate claims left out of review.",
                compact=True,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="mini-note">Tip: a lower threshold increases the review queue quickly. A higher threshold protects analyst capacity but lets more fraud slip through.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="stack-gap"></div>', unsafe_allow_html=True)

    with st.expander("Show saved threshold scenarios"):
        table = threshold_df.copy()
        table["threshold"] = table["threshold"].map(lambda x: f"{x:.3f}")
        for col in ["precision", "recall", "f1", "f2"]:
            table[col] = table[col].map(lambda x: f"{x:.3f}")
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_risk_drivers(theme_name: str, feature_df: pd.DataFrame) -> None:
    left, right = st.columns([1.12, 0.88])
    with left:
        section_header(
            "Top risk drivers",
            "Feature importance shows which signals had the strongest influence on the model's fraud ranking.",
            "These are the strongest signals pushing a claim higher in the risk queue.",
        )
        st.plotly_chart(
            plot_feature_importance(feature_df, theme_name),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with right:
        section_header(
            "Top 10 features",
            "This table keeps the strongest signals readable without scrolling through the full feature list.",
            "Use this table for a quick read on the most influential model inputs.",
        )
        st.markdown(feature_table_html(feature_df.head(10)), unsafe_allow_html=True)


def main() -> None:
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    if "_last_dark_mode" not in st.session_state:
        st.session_state["_last_dark_mode"] = st.session_state["dark_mode"]

    theme_name = "Dark" if st.session_state["dark_mode"] else "Light"
    apply_css(theme_name)

    with st.sidebar:
        st.markdown(
            """
            <div class="brand-card">
              <div class="sidebar-kicker">Fraud Risk</div>
              <h2 class="brand-title">Claims Monitor</h2>
              <p class="brand-copy">A focused view of claim risk, review thresholds, and the strongest fraud signals.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        dark_mode = st.toggle(
            "Dark theme",
            value=st.session_state["dark_mode"],
            help="Switch between light and dark viewing modes.",
        )
        st.session_state["dark_mode"] = dark_mode
        model_options = list(MODEL_KEYS.keys())
        default_index = model_options.index("XGBoost")
        model_label = st.selectbox(
            "Scoring model",
            model_options,
            index=default_index,
            help="Switch between the saved model outputs used in the analysis.",
        )
        st.markdown('<div class="stack-gap-sm"></div>', unsafe_allow_html=True)
        show_guide = st.toggle(
            "How to use",
            value=False,
            help="Open a short walkthrough for first-time viewers.",
        )

    theme_name = "Dark" if st.session_state["dark_mode"] else "Light"
    apply_css(theme_name)

    if st.session_state["dark_mode"] != st.session_state["_last_dark_mode"]:
        st.session_state["_last_dark_mode"] = st.session_state["dark_mode"]
        st.rerun()

    model_key = MODEL_KEYS[model_label]
    summary = load_json(ARTIFACT_DIR / f"{model_key}_summary.json")
    threshold_df = load_csv(ARTIFACT_DIR / f"{model_key}_threshold_table.csv")
    feature_df = load_csv(ARTIFACT_DIR / f"{model_key}_feature_importance.csv")
    comparison_df = load_csv(ARTIFACT_DIR / "model_comparison.csv")

    if show_guide:
        with st.sidebar:
            st.markdown('<div class="stack-gap-sm"></div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="use-card">
                  <ul>
                    <li>Choose a scoring model to refresh the charts and metrics.</li>
                    <li>Open Threshold Explorer to see how the review queue changes as the cutoff moves.</li>
                    <li>Hover the small <strong>i</strong> badges or metric help icons for plain-language definitions.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    test_metrics = summary["test_metrics_at_selected_threshold"]
    review_rate = test_metrics["flagged_claims"] / summary["test_shape"][0]

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">Insurance Fraud Risk Dashboard</div>
          <h1 class="hero-title">Auto Claim Fraud Screening</h1>
          <div class="hero-subtitle">
            Track screening quality, tune review thresholds, and understand which signals drive higher-risk claim prioritization.
          </div>
          <div class="hero-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    snapshot_cols = st.columns(3)
    snapshots = [
        (
            "Selected model",
            model_label,
            "Model currently used to populate the dashboard metrics and visuals.",
        ),
        (
            "Manual review rate",
            f"{format_pct(review_rate)} of claims flagged",
            "Share of claims that would be routed to human review at the current threshold.",
        ),
        (
            "Current threshold",
            f"Score cutoff {format_metric(test_metrics['threshold'])}",
            "Probability score used as the current decision cutoff for flagging a claim.",
        ),
    ]
    for col, (label, value, help_text) in zip(snapshot_cols, snapshots):
        with col:
            st.markdown(
                f"""
                <div class="snapshot-chip">
                  <div class="snapshot-head">
                    <div class="snapshot-label">{label}</div>
                    {info_badge(help_text)}
                  </div>
                  <div class="snapshot-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="utility-note">Hover the info markers for quick definitions.</div>',
        unsafe_allow_html=True,
    )

    tab_overview, tab_threshold, tab_drivers = st.tabs(
        ["Overview", "Threshold Explorer", "Risk Drivers"]
    )

    with tab_overview:
        render_overview(theme_name, summary, comparison_df, threshold_df, model_key)

    with tab_threshold:
        render_threshold_explorer(theme_name, summary, threshold_df)

    with tab_drivers:
        render_risk_drivers(theme_name, feature_df)


if __name__ == "__main__":
    main()
