from pathlib import Path
import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ML / DL Comparison",
    page_icon="◻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSET_DIR = BASE_DIR / "assets"

RESULT_FILE = DATA_DIR / "hasil_komparasi_ML_vs_DL_lengkap.xlsx"

COLORS = {
    "ink": "#18181B",
    "text": "#27272A",
    "muted": "#71717A",
    "soft": "#A1A1AA",
    "border": "#E4E4E7",
    "surface": "#FAFAFA",
    "surface2": "#F4F4F5",
    "white": "#FFFFFF",
    "accent": "#334155",
    "ml": "#52525B",
    "dl": "#111827",
    "green": "#166534",
    "red": "#B42318",
    "amber": "#B45309",
}

LABEL_DISPLAY = {
    "sentiment_positif": "Sentiment · Positive",
    "sentiment_netral": "Sentiment · Neutral",
    "sentiment_negatif": "Sentiment · Negative",
    "stress_kehamilan": "Pregnancy Stress",
    "external_support": "External Support",
    "local_wisdom": "Local Wisdom",
    "finansial": "Financial",
    "governance": "Governance",
    "app_teknis": "App Technical",
}

MODEL_ORDER = [
    "CNN",
    "BiLSTM",
    "Linear SVM",
    "Logistic Regression",
    "LSTM",
    "GRU",
    "Random Forest",
    "Multinomial Naive Bayes",
]


# ============================================================
# STYLING — CLEAN WHITE / MODERN MINIMAL
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --ink: #18181B;
        --text: #27272A;
        --muted: #71717A;
        --soft: #A1A1AA;
        --border: #E4E4E7;
        --surface: #FAFAFA;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .stApp {
        background: #FFFFFF;
        color: #18181B;
    }

    .block-container {
        max-width: 1420px;
        padding-top: 1.25rem;
        padding-bottom: 4rem;
        padding-left: 2.4rem;
        padding-right: 2.4rem;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    header[data-testid="stHeader"] {
        background: rgba(255,255,255,0);
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #E4E4E7;
        padding-bottom: 17px;
        margin-bottom: 19px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
    }

    .brand-mark {
        width: 34px;
        height: 34px;
        border: 1px solid #E4E4E7;
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        letter-spacing: .04em;
        font-weight: 750;
        color: #18181B;
        background: #FFFFFF;
    }

    .brand-title {
        color: #18181B;
        font-size: 13px;
        font-weight: 690;
        line-height: 1.2;
    }

    .brand-sub {
        color: #A1A1AA;
        font-size: 10.5px;
        margin-top: 2px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 9px;
        border: 1px solid #E4E4E7;
        border-radius: 999px;
        color: #52525B;
        font-size: 10.5px;
        font-weight: 560;
        background: #FAFAFA;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #22C55E;
        display: inline-block;
    }

    .eyebrow {
        color: #71717A;
        font-size: 10.5px;
        font-weight: 650;
        letter-spacing: .09em;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .page-title {
        color: #18181B;
        font-size: 36px;
        line-height: 1.05;
        font-weight: 680;
        letter-spacing: -0.04em;
        margin: 0;
    }

    .page-subtitle {
        color: #71717A;
        max-width: 820px;
        font-size: 13px;
        line-height: 1.7;
        margin-top: 11px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E4E4E7;
        border-radius: 11px;
        padding: 18px 19px;
        min-height: 118px;
    }

    .metric-label {
        color: #71717A;
        font-size: 10.5px;
        font-weight: 600;
        margin-bottom: 18px;
    }

    .metric-value {
        color: #18181B;
        font-size: 25px;
        line-height: 1.0;
        font-weight: 680;
        letter-spacing: -0.025em;
        word-break: break-word;
    }

    .metric-caption {
        color: #A1A1AA;
        font-size: 10.5px;
        line-height: 1.45;
        margin-top: 9px;
    }

    .section-head {
        margin-top: 31px;
        margin-bottom: 12px;
    }

    .section-title {
        color: #18181B;
        font-size: 16px;
        font-weight: 660;
        letter-spacing: -0.015em;
    }

    .section-sub {
        color: #A1A1AA;
        font-size: 11px;
        margin-top: 3px;
        line-height: 1.5;
    }

    .note {
        padding: 14px 15px;
        background: #FAFAFA;
        border: 1px solid #E4E4E7;
        border-radius: 9px;
        color: #52525B;
        font-size: 11px;
        line-height: 1.65;
    }

    .winner {
        padding: 17px 18px;
        border: 1px solid #D4D4D8;
        border-radius: 11px;
        background: #FAFAFA;
        min-height: 128px;
    }

    .winner-family {
        font-size: 10px;
        color: #A1A1AA;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 650;
    }

    .winner-name {
        font-size: 20px;
        font-weight: 680;
        color: #18181B;
        letter-spacing: -.025em;
        margin-top: 10px;
    }

    .winner-meta {
        font-size: 11px;
        color: #71717A;
        margin-top: 8px;
        line-height: 1.6;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E4E4E7;
        border-radius: 9px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #E4E4E7;
        border-radius: 9px;
        background: #FFFFFF;
    }

    div[role="radiogroup"] {
        gap: 4px;
        border-bottom: 1px solid #F1F1F3;
        padding-bottom: 10px;
        margin-bottom: 25px;
    }

    div[role="radiogroup"] label {
        background: transparent;
        border-radius: 7px;
        padding: 5px 9px;
        transition: background .12s ease;
    }

    div[role="radiogroup"] label:hover {
        background: #FAFAFA;
    }

    div[role="radiogroup"] label p {
        font-size: 11px !important;
        color: #52525B;
    }

    div[data-testid="stSelectbox"] label p,
    div[data-testid="stMultiSelect"] label p,
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stTextInput"] label p,
    div[data-testid="stSlider"] label p {
        font-size: 11px !important;
        color: #52525B;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] {
        border-color: #E4E4E7 !important;
        background: #FFFFFF !important;
        border-radius: 8px !important;
    }

    .stDownloadButton button {
        background: #18181B !important;
        color: #FFFFFF !important;
        border: 1px solid #18181B !important;
        border-radius: 8px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
    }

    .stButton button {
        border-radius: 8px !important;
        border-color: #D4D4D8 !important;
        font-size: 11px !important;
    }

    hr {
        border-color: #F1F1F3 !important;
    }

    @media (max-width: 760px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .page-title { font-size: 29px; }
        .status-pill { display: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================
@st.cache_data(show_spinner=False)
def load_results():
    leaderboard = pd.read_excel(RESULT_FILE, sheet_name="Leaderboard Test")
    split_results = pd.read_excel(RESULT_FILE, sheet_name="Train Val Test")
    per_label = pd.read_excel(RESULT_FILE, sheet_name="Per Label")
    prevalence = pd.read_excel(RESULT_FILE, sheet_name="Split Prevalence")

    # normalize
    leaderboard["Model"] = leaderboard["Model"].astype(str)
    split_results["Model"] = split_results["Model"].astype(str)
    per_label["Model"] = per_label["Model"].astype(str)
    per_label["Label Display"] = per_label["Label"].map(LABEL_DISPLAY).fillna(per_label["Label"])
    prevalence["Label Display"] = prevalence["Label"].map(LABEL_DISPLAY).fillna(prevalence["Label"])
    return leaderboard, split_results, per_label, prevalence


@st.cache_data(show_spinner=False)
def load_predictions():
    out = {}
    files = {
        "CNN": DATA_DIR / "predictions_test_CNN.csv",
        "Linear SVM": DATA_DIR / "predictions_test_Linear_SVM.csv",
    }
    for model, path in files.items():
        if path.exists():
            out[model] = pd.read_csv(path)
    return out


leaderboard, split_results, per_label, prevalence = load_results()
prediction_files = load_predictions()


# ============================================================
# HELPERS
# ============================================================
def fmt4(v):
    if pd.isna(v):
        return "—"
    return f"{float(v):.4f}"


def fmt_pct(v):
    if pd.isna(v):
        return "—"
    return f"{float(v) * 100:.2f}%"


def metric_card(label, value, caption=""):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{html.escape(str(label))}</div>
          <div class="metric-value">{html.escape(str(value))}</div>
          <div class="metric-caption">{html.escape(str(caption))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title, subtitle=""):
    st.markdown(
        f"""
        <div class="section-head">
          <div class="section-title">{html.escape(title)}</div>
          <div class="section-sub">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <div class="eyebrow">{html.escape(eyebrow)}</div>
        <div class="page-title">{html.escape(title)}</div>
        <div class="page-subtitle">{html.escape(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )


def winner_card(family, name, f1, acc, text=""):
    st.markdown(
        f"""
        <div class="winner">
          <div class="winner-family">{html.escape(family)}</div>
          <div class="winner-name">{html.escape(name)}</div>
          <div class="winner-meta">
            F1-Macro <b>{float(f1):.4f}</b> &nbsp;·&nbsp;
            Accuracy <b>{float(acc):.4f}</b>
            {("<br>" + html.escape(text)) if text else ""}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_fig(fig, height=430, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=46, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Inter, Arial, sans-serif",
            size=11,
            color=COLORS["text"],
        ),
        title_font=dict(size=14, color=COLORS["ink"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=None,
            font=dict(size=10),
        ) if legend else dict(visible=False),
        hoverlabel=dict(
            bgcolor="#18181B",
            font_color="#FFFFFF",
            bordercolor="#18181B",
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=COLORS["border"],
        tickfont=dict(size=10, color=COLORS["muted"]),
        title_font=dict(size=10, color=COLORS["muted"]),
    )
    fig.update_yaxes(
        gridcolor="#F1F1F3",
        zeroline=False,
        tickfont=dict(size=10, color=COLORS["muted"]),
        title_font=dict(size=10, color=COLORS["muted"]),
    )
    return fig


def make_download(df, filename, label):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


# ============================================================
# TOP BAR + NAV
# ============================================================
st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="brand-mark">ML/DL</div>
        <div>
          <div class="brand-title">Model Comparison Dashboard</div>
          <div class="brand-sub">Perinatal Stress Sentiment Analysis · Final Project</div>
        </div>
      </div>
      <div class="status-pill">
        <span class="status-dot"></span>
        Evaluation run completed
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    "Navigation",
    ["Overview", "Performance", "Per Label", "Diagnostics", "Prediction Explorer", "Method"],
    horizontal=True,
    label_visibility="collapsed",
)


# ============================================================
# CORE DERIVATIONS
# ============================================================
leaderboard_sorted = leaderboard.sort_values(["F1 Macro", "Accuracy"], ascending=[False, False]).reset_index(drop=True)
best_overall = leaderboard_sorted.iloc[0]
best_ml = (
    leaderboard_sorted[leaderboard_sorted["Family"] == "Machine Learning"]
    .sort_values(["F1 Macro", "Accuracy"], ascending=[False, False])
    .iloc[0]
)
best_dl = (
    leaderboard_sorted[leaderboard_sorted["Family"] == "Deep Learning"]
    .sort_values(["F1 Macro", "Accuracy"], ascending=[False, False])
    .iloc[0]
)
best_accuracy = leaderboard_sorted.sort_values("Accuracy", ascending=False).iloc[0]

delta_f1 = float(best_dl["F1 Macro"] - best_ml["F1 Macro"])
delta_acc = float(best_dl["Accuracy"] - best_ml["Accuracy"])


# ============================================================
# PAGE — OVERVIEW
# ============================================================
if page == "Overview":
    page_header(
        "Supplementary Benchmark",
        "Machine Learning vs Deep Learning",
        "Eight models evaluated on the same final TA pipeline: 19,443 modeling samples, "
        "nine binary targets, sentiment-stratified 80/10/10 split, random state 42, and "
        "F1-Macro as the primary metric.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Modeling Dataset", "19,443", "15,554 train · 1,944 validation · 1,945 test")
    with c2:
        metric_card("Models Evaluated", str(len(leaderboard_sorted)), "4 Machine Learning · 4 Deep Learning")
    with c3:
        metric_card("Best Overall · F1", best_overall["Model"], f'F1-Macro {best_overall["F1 Macro"]:.4f}')
    with c4:
        metric_card("Highest Accuracy", best_accuracy["Model"], f'Accuracy {best_accuracy["Accuracy"]:.4f}')

    section("Family winners", "Best model in each family based on test F1-Macro.")
    c1, c2 = st.columns(2)
    with c1:
        winner_card(
            "Machine Learning",
            best_ml["Model"],
            best_ml["F1 Macro"],
            best_ml["Accuracy"],
            "Best classic ML baseline.",
        )
    with c2:
        winner_card(
            "Deep Learning",
            best_dl["Model"],
            best_dl["F1 Macro"],
            best_dl["Accuracy"],
            "Best overall model under the primary metric.",
        )

    section("What changed from ML to DL?", "Delta between the best model in each family.")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        metric_card("Δ F1-Macro", f"{delta_f1:+.4f}", "CNN minus Linear SVM")
    with c2:
        metric_card("Δ Accuracy", f"{delta_acc:+.4f}", "CNN minus Linear SVM")
    with c3:
        st.markdown(
            """
            <div class="note">
            <b>Reading the result.</b> CNN is selected as the overall winner because F1-Macro
            is the primary metric. BiLSTM has the numerically highest label-wise accuracy,
            but its F1-Macro is slightly lower than CNN. This distinction is important in
            a multi-label setting with uneven label frequencies.
            </div>
            """,
            unsafe_allow_html=True,
        )

    section("Test-set ranking", "Primary metric: F1-Macro. Accuracy is shown as a supporting metric.")
    rank_plot = leaderboard_sorted.copy()
    rank_plot["Family Short"] = rank_plot["Family"].replace(
        {"Machine Learning": "ML", "Deep Learning": "DL"}
    )
    fig = px.bar(
        rank_plot.sort_values("F1 Macro"),
        x="F1 Macro",
        y="Model",
        orientation="h",
        color="Family Short",
        color_discrete_map={"ML": COLORS["ml"], "DL": COLORS["dl"]},
        text="F1 Macro",
        hover_data={"Accuracy": ":.4f", "Precision Macro": ":.4f", "Recall Macro": ":.4f"},
        title="F1-Macro ranking",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[max(0, rank_plot["F1 Macro"].min() - 0.08), min(1.0, rank_plot["F1 Macro"].max() + 0.08)])
    clean_fig(fig, height=460)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section("Quick leaderboard", "Core test metrics for all eight models.")
    show_cols = [
        "Rank", "Model", "Family", "Accuracy", "Precision Macro", "Recall Macro",
        "F1 Macro", "F1 Micro", "Hamming Loss", "Jaccard Samples"
    ]
    table = leaderboard_sorted.copy()
    table["Rank"] = np.arange(1, len(table) + 1)
    st.dataframe(
        table[show_cols].style.format({
            "Accuracy": "{:.4f}",
            "Precision Macro": "{:.4f}",
            "Recall Macro": "{:.4f}",
            "F1 Macro": "{:.4f}",
            "F1 Micro": "{:.4f}",
            "Hamming Loss": "{:.4f}",
            "Jaccard Samples": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE — PERFORMANCE
# ============================================================
elif page == "Performance":
    page_header(
        "Model Evaluation",
        "Performance Benchmark",
        "Compare test performance across model families and metrics. "
        "Use the controls below to focus on specific models or evaluation measures.",
    )

    selected_family = st.selectbox(
        "Model family",
        ["All", "Machine Learning", "Deep Learning"],
    )

    perf_df = leaderboard_sorted.copy()
    if selected_family != "All":
        perf_df = perf_df[perf_df["Family"] == selected_family].copy()

    metric_options = [
        "F1 Macro", "Accuracy", "Precision Macro", "Recall Macro",
        "F1 Micro", "F1 Weighted", "Jaccard Samples"
    ]
    selected_metrics = st.multiselect(
        "Metrics",
        metric_options,
        default=["F1 Macro", "Accuracy", "Precision Macro", "Recall Macro"],
    )
    if not selected_metrics:
        selected_metrics = ["F1 Macro"]

    section("Metric comparison", "All scores are from the held-out test set.")
    melted = perf_df.melt(
        id_vars=["Model", "Family"],
        value_vars=selected_metrics,
        var_name="Metric",
        value_name="Score",
    )

    fig = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_sequence=["#111827", "#475569", "#94A3B8", "#CBD5E1", "#52525B", "#A1A1AA", "#D4D4D8"],
        title="Selected test metrics",
    )
    fig.update_yaxes(range=[0, 1.0])
    clean_fig(fig, height=500)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section("Accuracy vs F1-Macro", "A model above and to the right is stronger on both dimensions.")
    fig = px.scatter(
        perf_df,
        x="Accuracy",
        y="F1 Macro",
        text="Model",
        color="Family",
        size="Jaccard Samples",
        size_max=24,
        color_discrete_map={"Machine Learning": COLORS["ml"], "Deep Learning": COLORS["dl"]},
        hover_data={
            "Precision Macro": ":.4f",
            "Recall Macro": ":.4f",
            "Hamming Loss": ":.4f",
            "Jaccard Samples": ":.4f",
        },
        title="Accuracy–F1 performance space",
    )
    fig.update_traces(textposition="top center")
    clean_fig(fig, height=500)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section("Full test leaderboard", "Downloadable benchmark table.")
    show = perf_df[
        [
            "Model", "Family", "Accuracy", "Subset Accuracy", "Precision Macro",
            "Recall Macro", "F1 Macro", "F1 Micro", "F1 Weighted",
            "Hamming Loss", "Jaccard Samples", "Elapsed Seconds"
        ]
    ].copy()
    st.dataframe(
        show.style.format({
            "Accuracy": "{:.4f}",
            "Subset Accuracy": "{:.4f}",
            "Precision Macro": "{:.4f}",
            "Recall Macro": "{:.4f}",
            "F1 Macro": "{:.4f}",
            "F1 Micro": "{:.4f}",
            "F1 Weighted": "{:.4f}",
            "Hamming Loss": "{:.4f}",
            "Jaccard Samples": "{:.4f}",
            "Elapsed Seconds": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    make_download(show, "leaderboard_ml_dl.csv", "Download leaderboard CSV")


# ============================================================
# PAGE — PER LABEL
# ============================================================
elif page == "Per Label":
    page_header(
        "Label-Level Evaluation",
        "Where does each model succeed?",
        "Inspect precision, recall, and F1 for each of the nine binary targets. "
        "This view is useful for identifying labels that are easy or difficult across architectures.",
    )

    models = list(leaderboard_sorted["Model"])
    default_models = [best_ml["Model"], best_dl["Model"]]
    chosen_models = st.multiselect(
        "Models",
        models,
        default=default_models,
    )
    if not chosen_models:
        chosen_models = default_models

    label_metric = st.selectbox("Metric", ["F1", "Precision", "Recall"])

    plot_df = per_label[per_label["Model"].isin(chosen_models)].copy()

    section(f"{label_metric} by label", "Direct comparison across the selected models.")
    fig = px.bar(
        plot_df,
        x="Label Display",
        y=label_metric,
        color="Model",
        barmode="group",
        color_discrete_sequence=["#111827", "#475569", "#94A3B8", "#D4D4D8", "#52525B", "#A1A1AA"],
        hover_data={"Support": True},
        title=f"{label_metric} across nine targets",
    )
    fig.update_yaxes(range=[0, 1.0])
    fig.update_xaxes(tickangle=-30)
    clean_fig(fig, height=520)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section("Heatmap", "F1-score for every model and target.")
    heat = (
        per_label.pivot(index="Model", columns="Label Display", values="F1")
        .reindex([m for m in MODEL_ORDER if m in set(per_label["Model"])])
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=heat.columns.tolist(),
            y=heat.index.tolist(),
            colorscale=[
                [0.0, "#FAFAFA"],
                [0.5, "#CBD5E1"],
                [1.0, "#111827"],
            ],
            zmin=0,
            zmax=1,
            text=np.round(heat.values, 3),
            texttemplate="%{text:.3f}",
            hovertemplate="Model=%{y}<br>Label=%{x}<br>F1=%{z:.4f}<extra></extra>",
            colorbar=dict(title="F1", thickness=11),
        )
    )
    fig.update_layout(title="Per-label F1 heatmap")
    clean_fig(fig, height=590, legend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section("Label prevalence", "Positive-label prevalence remains closely aligned across train, validation, and test.")
    prev_plot = prevalence.copy()
    fig = px.line(
        prev_plot,
        x="Label Display",
        y="Prevalence",
        color="Split",
        markers=True,
        color_discrete_map={
            "Train": "#111827",
            "Validation": "#64748B",
            "Test": "#CBD5E1",
        },
        title="Label prevalence by split",
    )
    fig.update_xaxes(tickangle=-30)
    clean_fig(fig, height=450)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section("Detailed per-label table")
    table = plot_df[["Model", "Family", "Label Display", "Support", "Precision", "Recall", "F1"]].copy()
    st.dataframe(
        table.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE — DIAGNOSTICS
# ============================================================
elif page == "Diagnostics":
    page_header(
        "Generalization & Training",
        "Diagnostics",
        "Compare train, validation, and test behavior, generalization gaps, runtime, "
        "learning curves, and binary confusion matrices for the family winners.",
    )

    selected_diag_metric = st.selectbox("Split metric", ["F1 Macro", "Accuracy"])

    section("Train / validation / test", "Large train-to-validation gaps should be interpreted as weaker generalization.")
    fig = px.bar(
        split_results,
        x="Model",
        y=selected_diag_metric,
        color="Split",
        barmode="group",
        color_discrete_map={
            "Train": "#18181B",
            "Validation": "#71717A",
            "Test": "#D4D4D8",
        },
        title=f"{selected_diag_metric} across data splits",
    )
    fig.update_yaxes(range=[0, 1.0])
    clean_fig(fig, height=500)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section("Generalization gap", "Train F1-Macro minus validation F1-Macro.")
    gap_df = leaderboard_sorted[["Model", "Family", "Train-Val F1 Gap"]].copy()
    gap_df["Gap Abs"] = gap_df["Train-Val F1 Gap"].abs()
    gap_df = gap_df.sort_values("Train-Val F1 Gap")

    fig = px.bar(
        gap_df,
        x="Train-Val F1 Gap",
        y="Model",
        orientation="h",
        color="Family",
        color_discrete_map={"Machine Learning": COLORS["ml"], "Deep Learning": COLORS["dl"]},
        text="Train-Val F1 Gap",
        title="Train–validation F1 gap",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside", cliponaxis=False)
    clean_fig(fig, height=470)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section("Training efficiency", "Elapsed training time is shown on a logarithmic scale.")
    eff = leaderboard_sorted.dropna(subset=["Elapsed Seconds"]).copy()
    fig = px.scatter(
        eff,
        x="Elapsed Seconds",
        y="F1 Macro",
        text="Model",
        color="Family",
        size="Accuracy",
        size_max=22,
        log_x=True,
        color_discrete_map={"Machine Learning": COLORS["ml"], "Deep Learning": COLORS["dl"]},
        hover_data={"Accuracy": ":.4f", "Elapsed Seconds": ":.2f"},
        title="Training time vs test F1-Macro",
    )
    fig.update_traces(textposition="top center")
    clean_fig(fig, height=480)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    learning_path = ASSET_DIR / "12_learning_curves_DL.png"
    if learning_path.exists():
        section("Deep-learning learning curves", "Original visualization exported from the comparison notebook.")
        st.image(str(learning_path), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        cm_ml = ASSET_DIR / "14_confusion_best_ML.png"
        if cm_ml.exists():
            section("Best ML · Binary confusion matrices", f'{best_ml["Model"]} on the test set.')
            st.image(str(cm_ml), use_container_width=True)
    with c2:
        cm_dl = ASSET_DIR / "15_confusion_best_DL.png"
        if cm_dl.exists():
            section("Best DL · Binary confusion matrices", f'{best_dl["Model"]} on the test set.')
            st.image(str(cm_dl), use_container_width=True)


# ============================================================
# PAGE — PREDICTION EXPLORER
# ============================================================
elif page == "Prediction Explorer":
    page_header(
        "Held-out Test Set",
        "Prediction Explorer",
        "Browse exported predictions for the best ML and best DL models. "
        "This is a read-only diagnostic view; the dashboard does not retrain or run inference.",
    )

    if not prediction_files:
        st.info("Prediction CSV files are not available.")
    else:
        selected_model = st.selectbox("Prediction file", list(prediction_files.keys()))
        pred_df = prediction_files[selected_model].copy()

        exact_rate = float(pred_df["exact_match"].mean())
        error_count = int((pred_df["exact_match"] == 0).sum())

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Samples", f"{len(pred_df):,}", "Held-out test examples")
        with c2:
            metric_card("Exact-match Rate", f"{exact_rate:.2%}", "All nine binary decisions correct")
        with c3:
            metric_card("Exact-match Errors", f"{error_count:,}", "Samples with ≥1 mismatched label")

        section("Filter examples")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            only_errors = st.checkbox("Errors only", value=True)
        with c2:
            only_exact = st.checkbox("Exact matches only", value=False)
        with c3:
            search_text = st.text_input("Search comment text", placeholder="e.g. suami, puskesmas, aplikasi...")

        view = pred_df.copy()
        if only_errors and not only_exact:
            view = view[view["exact_match"] == 0]
        elif only_exact and not only_errors:
            view = view[view["exact_match"] == 1]

        if search_text.strip():
            view = view[
                view["text"].astype(str).str.contains(search_text.strip(), case=False, na=False)
            ]

        max_rows = min(max(len(view), 1), 200)
        n_show = st.slider("Rows to display", min_value=5, max_value=max(5, max_rows), value=min(25, max(5, max_rows)), step=5)

        display_cols = ["text", "actual_labels", "predicted_labels", "exact_match"]
        st.dataframe(
            view[display_cols].head(n_show),
            use_container_width=True,
            hide_index=True,
            column_config={
                "text": st.column_config.TextColumn("Comment", width="large"),
                "actual_labels": st.column_config.TextColumn("Actual", width="large"),
                "predicted_labels": st.column_config.TextColumn("Predicted", width="large"),
                "exact_match": st.column_config.NumberColumn("Exact", format="%d"),
            },
        )
        make_download(view, f"prediction_explorer_{selected_model.replace(' ', '_')}.csv", "Download filtered rows")


# ============================================================
# PAGE — METHOD
# ============================================================
elif page == "Method":
    page_header(
        "Experiment Design",
        "Method & Pipeline",
        "The supplementary benchmark keeps the final TA data pipeline fixed and changes "
        "the learner/representation required by each model family.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Target", "9 binary labels", "3 sentiment + 6 aspect labels")
    with c2:
        metric_card("Split", "80 / 10 / 10", "Stratified by sentiment label")
    with c3:
        metric_card("Random State", "42", "Fixed experiment seed")
    with c4:
        metric_card("Primary Metric", "F1-Macro", "All labels receive equal weight")

    section("Fixed upstream pipeline", "The model comparison does not rebuild a different data-processing protocol.")
    steps = [
        ("01", "Scraping", "122,282 raw TikTok comments"),
        ("02", "Preprocessing", "Cleaning, normalization, decomposition, filtering"),
        ("03", "Quality filtering", "31,780 comments retained for labeling"),
        ("04", "Silver labeling", "Sentiment + six aspects"),
        ("05", "Final transformation", "19,443 modeling samples · nine binary targets"),
        ("06", "Split", "15,554 train · 1,944 validation · 1,945 test"),
        ("07", "Representation", "TF-IDF for ML · IndoBERT tokenizer IDs for DL"),
        ("08", "Model training", "4 ML + 4 DL models"),
        ("09", "Evaluation", "F1-Macro primary, supporting multi-label metrics"),
    ]

    for num, title, desc in steps:
        st.markdown(
            f"""
            <div style="display:flex; gap:14px; align-items:flex-start;
                        border-bottom:1px solid #F1F1F3; padding:11px 0;">
                <div style="font-size:10px;color:#A1A1AA;width:28px;padding-top:2px;">{num}</div>
                <div>
                    <div style="font-size:12px;font-weight:650;color:#18181B;">{html.escape(title)}</div>
                    <div style="font-size:11px;color:#71717A;margin-top:3px;line-height:1.55;">{html.escape(desc)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    pipeline_img = ASSET_DIR / "01_pipeline_funnel_aktual_TA.png"
    if pipeline_img.exists():
        section("Pipeline funnel", "Original figure exported by the comparison notebook.")
        st.image(str(pipeline_img), use_container_width=True)

    section("Models in this benchmark")
    model_table = pd.DataFrame(
        [
            ["Machine Learning", "Logistic Regression", "TF-IDF", "One-vs-rest"],
            ["Machine Learning", "Multinomial Naive Bayes", "TF-IDF", "One-vs-rest"],
            ["Machine Learning", "Linear SVM", "TF-IDF", "One-vs-rest"],
            ["Machine Learning", "Random Forest", "TF-IDF", "One-vs-rest"],
            ["Deep Learning", "CNN", "IndoBERT tokenizer IDs", "Trainable embedding"],
            ["Deep Learning", "LSTM", "IndoBERT tokenizer IDs", "Trainable embedding"],
            ["Deep Learning", "BiLSTM", "IndoBERT tokenizer IDs", "Trainable embedding"],
            ["Deep Learning", "GRU", "IndoBERT tokenizer IDs", "Trainable embedding"],
        ],
        columns=["Family", "Model", "Input Representation", "Notes"],
    )
    st.dataframe(model_table, use_container_width=True, hide_index=True)

    section("Interpretation boundary")
    st.markdown(
        """
        <div class="note">
        <b>Silver-label benchmark.</b> These scores quantify agreement with the final
        silver labels used in the TA pipeline. They are not gold-label clinical accuracy
        and must not be interpreted as diagnosis, screening performance, or prevalence
        of perinatal mental-health conditions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="border-top:1px solid #F1F1F3;padding-top:15px;
                color:#A1A1AA;font-size:10px;display:flex;
                justify-content:space-between;gap:12px;flex-wrap:wrap;">
      <span>ML / DL Comparison Dashboard</span>
      <span>Read-only research visualization · 2026</span>
    </div>
    """,
    unsafe_allow_html=True,
)
