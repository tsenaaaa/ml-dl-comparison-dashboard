from pathlib import Path

import html
import re
import unicodedata

import joblib
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
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
ASSET_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR / "models"

RESULT_FILE = DATA_DIR / "hasil_komparasi_ML_vs_DL_lengkap.xlsx"

SVM_MODEL_FILE = MODEL_DIR / "Linear_SVM.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"


# ============================================================
# COLOR SYSTEM — BRIGHT / CLEAN / PASTEL
# ============================================================
COLORS = {
    "text": "#334155",
    "muted": "#64748B",
    "soft": "#94A3B8",
    "border": "#E2E8F0",
    "surface": "#F8FBFF",
    "surface2": "#F1F7FF",
    "white": "#FFFFFF",

    "blue": "#55A6FF",
    "blue_soft": "#EAF4FF",

    "mint": "#34D399",
    "mint_soft": "#ECFDF5",

    "yellow": "#FBBF24",
    "yellow_soft": "#FFFBEB",

    "coral": "#FB7185",
    "coral_soft": "#FFF1F2",

    "lavender": "#A78BFA",
    "lavender_soft": "#F5F3FF",

    "cyan": "#22D3EE",
    "cyan_soft": "#ECFEFF",

    "orange": "#FB923C",
    "orange_soft": "#FFF7ED",
}

PASTEL_SEQUENCE = [
    COLORS["blue"],
    COLORS["mint"],
    COLORS["yellow"],
    COLORS["coral"],
    COLORS["lavender"],
    COLORS["cyan"],
    COLORS["orange"],
    "#60A5FA",
]

FAMILY_COLORS = {
    "Machine Learning": COLORS["blue"],
    "Deep Learning": COLORS["mint"],
}

SPLIT_COLORS = {
    "Train": COLORS["blue"],
    "Validation": COLORS["lavender"],
    "Test": COLORS["mint"],
}


# ============================================================
# LABEL / MODEL METADATA
# ============================================================
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

TARGET_COLS = [
    "sentiment_positif",
    "sentiment_netral",
    "sentiment_negatif",
    "stress_kehamilan",
    "external_support",
    "local_wisdom",
    "finansial",
    "governance",
    "app_teknis",
]

SENTIMENT_DISPLAY = {
    "sentiment_positif": "Positive",
    "sentiment_netral": "Neutral",
    "sentiment_negatif": "Negative",
}

ASPECT_DISPLAY = {
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
# STYLING — CLEAN WHITE + BRIGHT ACCENTS
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --text: #334155;
        --muted: #64748B;
        --soft: #94A3B8;
        --border: #E2E8F0;
        --blue: #55A6FF;
        --mint: #34D399;
        --yellow: #FBBF24;
        --coral: #FB7185;
        --lavender: #A78BFA;
    }

    html, body, [class*="css"] {
        font-family:
            Inter,
            ui-sans-serif,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            Roboto,
            Helvetica,
            Arial,
            sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 7% 0%,
                rgba(85, 166, 255, 0.10),
                transparent 24rem
            ),
            radial-gradient(
                circle at 96% 8%,
                rgba(52, 211, 153, 0.09),
                transparent 24rem
            ),
            radial-gradient(
                circle at 78% 100%,
                rgba(167, 139, 250, 0.06),
                transparent 22rem
            ),
            #FFFFFF;
        color: #334155;
    }

    .block-container {
        max-width: 1420px;
        padding-top: 1.25rem;
        padding-bottom: 4rem;
        padding-left: 2.4rem;
        padding-right: 2.4rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0);
    }


    /* ==============================
       TOP BAR
    ============================== */

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;

        border-bottom: 1px solid #E8EEF6;

        padding-bottom: 17px;
        margin-bottom: 19px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
    }

    .brand-mark {
        width: 38px;
        height: 38px;

        border: 1px solid #CFE5FF;
        border-radius: 11px;

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 10px;
        letter-spacing: 0.035em;
        font-weight: 760;

        color: #3B82F6;

        background:
            linear-gradient(
                145deg,
                #EAF4FF,
                #F2FFFA
            );

        box-shadow:
            0 6px 18px
            rgba(85, 166, 255, 0.10);
    }

    .brand-title {
        color: #334155;
        font-size: 13px;
        font-weight: 700;
        line-height: 1.2;
    }

    .brand-sub {
        color: #94A3B8;
        font-size: 10.5px;
        margin-top: 2px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;

        padding: 6px 10px;

        border: 1px solid #CFF4E4;
        border-radius: 999px;

        color: #15803D;

        font-size: 10.5px;
        font-weight: 620;

        background: #F0FDF4;
    }

    .status-dot {
        width: 7px;
        height: 7px;

        border-radius: 50%;
        background: #34D399;

        display: inline-block;
    }


    /* ==============================
       PAGE HEADER
    ============================== */

    .eyebrow {
        color: #55A6FF;

        font-size: 10.5px;
        font-weight: 720;

        letter-spacing: 0.09em;
        text-transform: uppercase;

        margin-bottom: 9px;
    }

    .page-title {
        color: #334155;

        font-size: 36px;
        line-height: 1.05;

        font-weight: 700;
        letter-spacing: -0.04em;

        margin: 0;
    }

    .page-subtitle {
        color: #64748B;

        max-width: 850px;

        font-size: 13px;
        line-height: 1.7;

        margin-top: 11px;
        margin-bottom: 25px;
    }


    /* ==============================
       METRIC CARDS
    ============================== */

    .metric-card {
        background: rgba(255, 255, 255, 0.98);

        border: 1px solid #E2E8F0;
        border-radius: 14px;

        padding: 18px 19px;

        min-height: 118px;

        box-shadow:
            0 8px 26px
            rgba(148, 163, 184, 0.08);
    }

    .metric-card-blue {
        border-top: 3px solid #55A6FF;
    }

    .metric-card-mint {
        border-top: 3px solid #34D399;
    }

    .metric-card-yellow {
        border-top: 3px solid #FBBF24;
    }

    .metric-card-coral {
        border-top: 3px solid #FB7185;
    }

    .metric-card-lavender {
        border-top: 3px solid #A78BFA;
    }

    .metric-label {
        color: #64748B;

        font-size: 10.5px;
        font-weight: 620;

        margin-bottom: 18px;
    }

    .metric-value {
        color: #334155;

        font-size: 25px;
        line-height: 1.0;

        font-weight: 720;
        letter-spacing: -0.025em;

        word-break: break-word;
    }

    .metric-caption {
        color: #94A3B8;

        font-size: 10.5px;
        line-height: 1.45;

        margin-top: 9px;
    }


    /* ==============================
       SECTION
    ============================== */

    .section-head {
        margin-top: 31px;
        margin-bottom: 12px;
    }

    .section-title {
        color: #334155;

        font-size: 16px;
        font-weight: 700;

        letter-spacing: -0.015em;
    }

    .section-sub {
        color: #94A3B8;

        font-size: 11px;
        line-height: 1.5;

        margin-top: 3px;
    }


    /* ==============================
       NOTES & WINNERS
    ============================== */

    .note {
        padding: 14px 15px;

        background:
            linear-gradient(
                135deg,
                #F8FBFF,
                #F4FFFB
            );

        border: 1px solid #DCEAF8;
        border-radius: 11px;

        color: #64748B;

        font-size: 11px;
        line-height: 1.65;
    }

    .winner {
        padding: 18px 19px;

        border: 1px solid #DCEAF8;
        border-radius: 14px;

        background:
            linear-gradient(
                135deg,
                #FFFFFF,
                #F8FBFF
            );

        min-height: 132px;

        box-shadow:
            0 8px 26px
            rgba(148, 163, 184, 0.07);
    }

    .winner-family {
        font-size: 10px;

        color: #55A6FF;

        text-transform: uppercase;
        letter-spacing: 0.08em;

        font-weight: 720;
    }

    .winner-name {
        color: #334155;

        font-size: 20px;
        font-weight: 720;

        letter-spacing: -0.025em;

        margin-top: 10px;
    }

    .winner-meta {
        color: #64748B;

        font-size: 11px;
        line-height: 1.6;

        margin-top: 8px;
    }


    /* ==============================
       LIVE PREDICTION
    ============================== */

    .prediction-hero {
        padding: 21px;

        border-radius: 16px;

        background:
            linear-gradient(
                135deg,
                #EAF4FF 0%,
                #F4FFFB 50%,
                #FFF9E9 100%
            );

        border: 1px solid #D8EAFB;

        margin-bottom: 18px;

        box-shadow:
            0 10px 28px
            rgba(85, 166, 255, 0.08);
    }

    .prediction-title {
        color: #334155;

        font-size: 17px;
        font-weight: 720;
    }

    .prediction-sub {
        color: #64748B;

        font-size: 11px;
        line-height: 1.6;

        margin-top: 5px;
    }

    .chip-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;

        margin-top: 8px;
        margin-bottom: 6px;
    }

    .result-chip {
        display: inline-flex;
        align-items: center;

        padding: 7px 11px;

        border-radius: 999px;

        font-size: 11px;
        font-weight: 650;
    }

    .chip-blue {
        background: #EAF4FF;
        color: #2563EB;
        border: 1px solid #CFE5FF;
    }

    .chip-mint {
        background: #ECFDF5;
        color: #15803D;
        border: 1px solid #CFF4E4;
    }

    .chip-yellow {
        background: #FFFBEB;
        color: #A16207;
        border: 1px solid #FDE68A;
    }

    .chip-coral {
        background: #FFF1F2;
        color: #BE123C;
        border: 1px solid #FECDD3;
    }

    .chip-lavender {
        background: #F5F3FF;
        color: #7C3AED;
        border: 1px solid #DDD6FE;
    }


    /* ==============================
       STREAMLIT COMPONENTS
    ============================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 11px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #E2E8F0;
        border-radius: 11px;
        background: #FFFFFF;
    }

    div[role="radiogroup"] {
        gap: 4px;

        border-bottom: 1px solid #EFF4F8;

        padding-bottom: 10px;
        margin-bottom: 24px;
    }

    div[role="radiogroup"] label {
        background: transparent;

        border-radius: 8px;

        padding: 5px 9px;

        transition: background 0.12s ease;
    }

    div[role="radiogroup"] label:hover {
        background: #F3F9FF;
    }

    div[role="radiogroup"] label p {
        color: #64748B;

        font-size: 11px !important;
    }

    div[data-testid="stSelectbox"] label p,
    div[data-testid="stMultiSelect"] label p,
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stTextInput"] label p,
    div[data-testid="stTextArea"] label p,
    div[data-testid="stSlider"] label p {
        color: #64748B;

        font-size: 11px !important;
    }

    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] {
        border-color: #DCE6F1 !important;
        background: #FFFFFF !important;
        border-radius: 10px !important;
    }

    textarea {
        color: #334155 !important;

        font-size: 13px !important;
        line-height: 1.6 !important;
    }

    .stButton button,
    .stDownloadButton button {
        background:
            linear-gradient(
                135deg,
                #55A6FF,
                #60A5FA
            ) !important;

        color: #FFFFFF !important;

        border: none !important;
        border-radius: 10px !important;

        font-size: 11px !important;
        font-weight: 650 !important;

        box-shadow:
            0 7px 18px
            rgba(85, 166, 255, 0.18);
    }

    .stButton button:hover,
    .stDownloadButton button:hover {
        background:
            linear-gradient(
                135deg,
                #3B82F6,
                #55A6FF
            ) !important;
    }

    hr {
        border-color: #EFF4F8 !important;
    }


    /* ==============================
       MOBILE
    ============================== */

    @media (max-width: 760px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .page-title {
            font-size: 29px;
        }

        .status-pill {
            display: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADERS
# ============================================================
@st.cache_data(show_spinner=False)
def load_results():
    leaderboard = pd.read_excel(
        RESULT_FILE,
        sheet_name="Leaderboard Test",
    )

    split_results = pd.read_excel(
        RESULT_FILE,
        sheet_name="Train Val Test",
    )

    per_label = pd.read_excel(
        RESULT_FILE,
        sheet_name="Per Label",
    )

    prevalence = pd.read_excel(
        RESULT_FILE,
        sheet_name="Split Prevalence",
    )

    leaderboard["Model"] = leaderboard["Model"].astype(str)
    split_results["Model"] = split_results["Model"].astype(str)
    per_label["Model"] = per_label["Model"].astype(str)

    per_label["Label Display"] = (
        per_label["Label"]
        .map(LABEL_DISPLAY)
        .fillna(per_label["Label"])
    )

    prevalence["Label Display"] = (
        prevalence["Label"]
        .map(LABEL_DISPLAY)
        .fillna(prevalence["Label"])
    )

    return (
        leaderboard,
        split_results,
        per_label,
        prevalence,
    )


@st.cache_data(show_spinner=False)
def load_predictions():
    out = {}

    files = {
        "CNN":
            DATA_DIR
            / "predictions_test_CNN.csv",

        "Linear SVM":
            DATA_DIR
            / "predictions_test_Linear_SVM.csv",
    }

    for model, path in files.items():
        if path.exists():
            out[model] = pd.read_csv(path)

    return out


@st.cache_resource(
    show_spinner="Loading realtime prediction model..."
)
def load_live_predictor():
    model = joblib.load(
        SVM_MODEL_FILE
    )

    vectorizer = joblib.load(
        TFIDF_FILE
    )

    return (
        model,
        vectorizer,
    )


leaderboard, split_results, per_label, prevalence = (
    load_results()
)

prediction_files = load_predictions()


# ============================================================
# LIVE PREDICTION PREPROCESSING
# ============================================================
COMMON_SLANG = {
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "engga": "tidak",
    "enggak": "tidak",
    "gk": "tidak",
    "tdk": "tidak",

    "bgt": "banget",
    "yg": "yang",
    "dgn": "dengan",
    "utk": "untuk",

    "krn": "karena",
    "karna": "karena",

    "sm": "sama",

    "sy": "saya",
    "sya": "saya",
    "aq": "aku",

    "udh": "sudah",
    "udah": "sudah",

    "blm": "belum",
    "belom": "belum",

    "kalo": "kalau",

    "tp": "tapi",
    "tpi": "tapi",

    "trs": "terus",
    "trus": "terus",

    "jd": "jadi",
    "jdi": "jadi",

    "pgn": "ingin",
    "pngn": "ingin",

    "skrg": "sekarang",

    "lg": "lagi",
    "lgi": "lagi",

    "dr": "dari",
    "dri": "dari",

    "cape": "capek",

    "nangis": "menangis",

    "stress": "stres",
}


def preprocess_new_comment(text):
    """
    Lightweight runtime normalization for a new comment.

    The model artifacts were fitted on the final TA data.
    Therefore, this function only performs conservative structural
    cleaning and a compact slang normalization before TF-IDF transform.
    """

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text,
        flags=re.MULTILINE,
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        " ",
        text,
    )

    # Remove emoji / symbols
    text = "".join(
        ch
        for ch in text
        if unicodedata.category(ch)
        not in {
            "So",
            "Sk",
            "Cs",
        }
    )

    # ASCII normalization
    text = (
        text
        .encode(
            "ascii",
            "ignore",
        )
        .decode("ascii")
    )

    # Case folding
    text = text.lower()

    # Reduce extreme repeated characters
    text = re.sub(
        r"([a-z])\1{2,}",
        r"\1\1",
        text,
    )

    # Remove punctuation
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        " ",
        text,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # Compact slang normalization
    normalized_tokens = []

    for token in text.split():
        normalized_tokens.append(
            COMMON_SLANG.get(
                token,
                token,
            )
        )

    return " ".join(
        normalized_tokens
    )


def sigmoid_margin(x):
    """
    Transform an SVM decision margin into a 0-1 relative score.
    This is NOT a calibrated probability.
    """

    x = np.clip(
        np.asarray(
            x,
            dtype=float,
        ),
        -20,
        20,
    )

    return (
        1.0
        /
        (
            1.0
            + np.exp(-x)
        )
    )


def live_predictor_ready():
    return (
        SVM_MODEL_FILE.exists()
        and
        TFIDF_FILE.exists()
    )


def predict_new_comment(raw_text):
    model, vectorizer = (
        load_live_predictor()
    )

    clean_text = preprocess_new_comment(
        raw_text
    )

    if not clean_text.strip():
        raise ValueError(
            "Text becomes empty after preprocessing."
        )

    # Use the fitted TF-IDF vectorizer from the comparison run.
    x = vectorizer.transform(
        [clean_text]
    )

    # Nine binary outputs.
    binary_pred = (
        model
        .predict(x)
        .astype(int)[0]
    )

    # One-vs-rest SVM decision scores.
    if hasattr(
        model,
        "decision_function",
    ):
        margins = np.asarray(
            model.decision_function(x)
        )[0]

    else:
        margins = np.where(
            binary_pred == 1,
            1.0,
            -1.0,
        )

    relative_scores = sigmoid_margin(
        margins
    )

    # Sentiment is conceptually single-label.
    # Use the highest margin among the first 3 sentiment outputs.
    sentiment_index = int(
        np.argmax(
            margins[:3]
        )
    )

    sentiment_key = TARGET_COLS[
        sentiment_index
    ]

    # Aspect outputs remain multi-label.
    detected_aspects = [
        TARGET_COLS[j]

        for j
        in range(
            3,
            len(TARGET_COLS),
        )

        if binary_pred[j] == 1
    ]

    # Useful fallback if no aspect crosses the binary boundary.
    top_aspect_index = (
        3
        +
        int(
            np.argmax(
                margins[3:]
            )
        )
    )

    top_aspect_fallback = (
        TARGET_COLS[
            top_aspect_index
        ]
    )

    return {
        "clean_text":
            clean_text,

        "binary_pred":
            binary_pred,

        "margins":
            margins,

        "relative_scores":
            relative_scores,

        "sentiment_key":
            sentiment_key,

        "detected_aspects":
            detected_aspects,

        "top_aspect_fallback":
            top_aspect_fallback,
    }


# ============================================================
# GENERAL HELPERS
# ============================================================
def fmt4(v):
    if pd.isna(v):
        return "—"

    return f"{float(v):.4f}"


def fmt_pct(v):
    if pd.isna(v):
        return "—"

    return f"{float(v) * 100:.2f}%"


def metric_card(
    label,
    value,
    caption="",
    accent="blue",
):
    valid_accents = {
        "blue",
        "mint",
        "yellow",
        "coral",
        "lavender",
    }

    if accent not in valid_accents:
        accent = "blue"

    st.markdown(
        f"""
        <div class="
            metric-card
            metric-card-{accent}
        ">
          <div class="metric-label">
            {html.escape(str(label))}
          </div>

          <div class="metric-value">
            {html.escape(str(value))}
          </div>

          <div class="metric-caption">
            {html.escape(str(caption))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(
    title,
    subtitle="",
):
    st.markdown(
        f"""
        <div class="section-head">
          <div class="section-title">
            {html.escape(title)}
          </div>

          <div class="section-sub">
            {html.escape(subtitle)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(
    eyebrow,
    title,
    subtitle,
):
    st.markdown(
        f"""
        <div class="eyebrow">
            {html.escape(eyebrow)}
        </div>

        <div class="page-title">
            {html.escape(title)}
        </div>

        <div class="page-subtitle">
            {html.escape(subtitle)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def winner_card(
    family,
    name,
    f1,
    acc,
    text="",
):
    st.markdown(
        f"""
        <div class="winner">
          <div class="winner-family">
            {html.escape(family)}
          </div>

          <div class="winner-name">
            {html.escape(name)}
          </div>

          <div class="winner-meta">
            F1-Macro
            <b>{float(f1):.4f}</b>
            &nbsp;·&nbsp;
            Accuracy
            <b>{float(acc):.4f}</b>

            {
                ("<br>" + html.escape(text))
                if text
                else ""
            }
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_fig(
    fig,
    height=430,
    legend=True,
):
    fig.update_layout(
        height=height,

        margin=dict(
            l=18,
            r=18,
            t=48,
            b=20,
        ),

        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",

        font=dict(
            family="Inter, Arial, sans-serif",
            size=11,
            color=COLORS["text"],
        ),

        title_font=dict(
            size=14,
            color=COLORS["text"],
        ),

        legend=(
            dict(
                orientation="h",

                yanchor="bottom",
                y=1.02,

                xanchor="right",
                x=1,

                title=None,

                font=dict(
                    size=10
                ),
            )
            if legend

            else
            dict(
                visible=False
            )
        ),

        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_color="#334155",
            bordercolor="#CBD5E1",
        ),
    )

    fig.update_xaxes(
        showgrid=False,

        linecolor=COLORS["border"],

        tickfont=dict(
            size=10,
            color=COLORS["muted"],
        ),

        title_font=dict(
            size=10,
            color=COLORS["muted"],
        ),
    )

    fig.update_yaxes(
        gridcolor="#EEF4F9",
        zeroline=False,

        tickfont=dict(
            size=10,
            color=COLORS["muted"],
        ),

        title_font=dict(
            size=10,
            color=COLORS["muted"],
        ),
    )

    return fig


def make_download(
    df,
    filename,
    label,
):
    csv_bytes = (
        df
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )

    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


# ============================================================
# TOP BAR + NAVIGATION
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
Dashboard + live prediction
</div>
</div>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    "Navigation",

    [
        "Overview",
        "Performance",
        "Per Label",
        "Diagnostics",
        "Live Prediction",
        "Prediction Explorer",
        "Method",
    ],

    horizontal=True,
    label_visibility="collapsed",
)


# ============================================================
# CORE DERIVATIONS
# ============================================================
leaderboard_sorted = (
    leaderboard
    .sort_values(
        [
            "F1 Macro",
            "Accuracy",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)

best_overall = (
    leaderboard_sorted
    .iloc[0]
)

best_ml = (
    leaderboard_sorted[
        leaderboard_sorted["Family"]
        == "Machine Learning"
    ]
    .sort_values(
        [
            "F1 Macro",
            "Accuracy",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .iloc[0]
)

best_dl = (
    leaderboard_sorted[
        leaderboard_sorted["Family"]
        == "Deep Learning"
    ]
    .sort_values(
        [
            "F1 Macro",
            "Accuracy",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .iloc[0]
)

best_accuracy = (
    leaderboard_sorted
    .sort_values(
        "Accuracy",
        ascending=False,
    )
    .iloc[0]
)

delta_f1 = float(
    best_dl["F1 Macro"]
    -
    best_ml["F1 Macro"]
)

delta_acc = float(
    best_dl["Accuracy"]
    -
    best_ml["Accuracy"]
)


# ============================================================
# PAGE — OVERVIEW
# ============================================================
if page == "Overview":

    page_header(
        "Supplementary Benchmark",

        "Machine Learning vs Deep Learning",

        (
            "Eight models evaluated on the same final TA pipeline: "
            "19,443 modeling samples, nine binary targets, "
            "sentiment-stratified 80/10/10 split, random state 42, "
            "and F1-Macro as the primary metric."
        ),
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:
        metric_card(
            "Modeling Dataset",
            "19,443",

            (
                "15,554 train · "
                "1,944 validation · "
                "1,945 test"
            ),

            "blue",
        )


    with c2:
        metric_card(
            "Models Evaluated",
            str(
                len(
                    leaderboard_sorted
                )
            ),

            "4 Machine Learning · 4 Deep Learning",

            "mint",
        )


    with c3:
        metric_card(
            "Best Overall · F1",

            best_overall["Model"],

            (
                f'F1-Macro '
                f'{best_overall["F1 Macro"]:.4f}'
            ),

            "yellow",
        )


    with c4:
        metric_card(
            "Highest Accuracy",

            best_accuracy["Model"],

            (
                f'Accuracy '
                f'{best_accuracy["Accuracy"]:.4f}'
            ),

            "coral",
        )


    section(
        "Family winners",

        (
            "Best model in each family "
            "based on test F1-Macro."
        ),
    )


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


    section(
        "Best DL vs best ML",

        (
            "Difference between the family winners "
            "on the held-out test set."
        ),
    )


    c1, c2, c3 = st.columns(
        [
            1,
            1,
            2,
        ]
    )


    with c1:
        metric_card(
            "Δ F1-Macro",

            f"{delta_f1:+.4f}",

            (
                f'{best_dl["Model"]} '
                f'minus '
                f'{best_ml["Model"]}'
            ),

            "mint",
        )


    with c2:
        metric_card(
            "Δ Accuracy",

            f"{delta_acc:+.4f}",

            (
                f'{best_dl["Model"]} '
                f'minus '
                f'{best_ml["Model"]}'
            ),

            "blue",
        )


    with c3:
        st.markdown(
            """
            <div class="note">
            <b>Reading the benchmark.</b>
            CNN is selected as the overall winner because
            F1-Macro is the primary metric. The realtime
            prediction page uses Linear SVM — the best
            Machine Learning model — because it provides
            lightweight inference while the benchmark result
            itself remains unchanged.
            </div>
            """,

            unsafe_allow_html=True,
        )


    section(
        "Test-set ranking",

        (
            "Primary metric: F1-Macro. "
            "Accuracy is used as a supporting metric."
        ),
    )


    rank_plot = (
        leaderboard_sorted
        .copy()
    )


    fig = px.bar(
        rank_plot.sort_values(
            "F1 Macro"
        ),

        x="F1 Macro",
        y="Model",

        orientation="h",

        color="Family",

        color_discrete_map=
            FAMILY_COLORS,

        text="F1 Macro",

        hover_data={
            "Accuracy": ":.4f",
            "Precision Macro": ":.4f",
            "Recall Macro": ":.4f",
        },

        title="F1-Macro ranking",
    )


    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
        cliponaxis=False,
    )


    fig.update_xaxes(
        range=[
            max(
                0,
                rank_plot[
                    "F1 Macro"
                ].min()
                - 0.08,
            ),

            min(
                1.0,
                rank_plot[
                    "F1 Macro"
                ].max()
                + 0.08,
            ),
        ]
    )


    clean_fig(
        fig,
        height=460,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,

        config={
            "displayModeBar":
                False
        },
    )


    section(
        "Quick leaderboard",

        (
            "Core test metrics "
            "for all eight models."
        ),
    )


    table = (
        leaderboard_sorted
        .copy()
    )


    table["Rank"] = np.arange(
        1,
        len(table)
        + 1,
    )


    show_cols = [
        "Rank",
        "Model",
        "Family",
        "Accuracy",
        "Precision Macro",
        "Recall Macro",
        "F1 Macro",
        "F1 Micro",
        "Hamming Loss",
        "Jaccard Samples",
    ]


    st.dataframe(
        table[
            show_cols
        ]
        .style
        .format({
            "Accuracy":
                "{:.4f}",

            "Precision Macro":
                "{:.4f}",

            "Recall Macro":
                "{:.4f}",

            "F1 Macro":
                "{:.4f}",

            "F1 Micro":
                "{:.4f}",

            "Hamming Loss":
                "{:.4f}",

            "Jaccard Samples":
                "{:.4f}",
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

        (
            "Compare test performance across model families "
            "and metrics using a bright, clean analytical view."
        ),
    )


    selected_family = st.selectbox(
        "Model family",

        [
            "All",
            "Machine Learning",
            "Deep Learning",
        ],
    )


    perf_df = (
        leaderboard_sorted
        .copy()
    )


    if selected_family != "All":

        perf_df = (
            perf_df[
                perf_df["Family"]
                ==
                selected_family
            ]
            .copy()
        )


    metric_options = [
        "F1 Macro",
        "Accuracy",
        "Precision Macro",
        "Recall Macro",
        "F1 Micro",
        "F1 Weighted",
        "Jaccard Samples",
    ]


    selected_metrics = st.multiselect(
        "Metrics",

        metric_options,

        default=[
            "F1 Macro",
            "Accuracy",
            "Precision Macro",
            "Recall Macro",
        ],
    )


    if not selected_metrics:
        selected_metrics = [
            "F1 Macro"
        ]


    section(
        "Metric comparison",

        (
            "All scores are from "
            "the held-out test set."
        ),
    )


    melted = perf_df.melt(
        id_vars=[
            "Model",
            "Family",
        ],

        value_vars=
            selected_metrics,

        var_name="Metric",

        value_name="Score",
    )


    fig = px.bar(
        melted,

        x="Model",
        y="Score",

        color="Metric",

        barmode="group",

        color_discrete_sequence=
            PASTEL_SEQUENCE,

        title=
            "Selected test metrics",
    )


    fig.update_yaxes(
        range=[
            0,
            1.0,
        ]
    )


    clean_fig(
        fig,
        height=500,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,

        config={
            "displayModeBar":
                False
        },
    )


    section(
        "Accuracy vs F1-Macro",

        (
            "A model above and to the right "
            "is stronger on both dimensions."
        ),
    )


    fig = px.scatter(
        perf_df,

        x="Accuracy",
        y="F1 Macro",

        text="Model",

        color="Family",

        size="Jaccard Samples",

        size_max=24,

        color_discrete_map=
            FAMILY_COLORS,

        hover_data={
            "Precision Macro": ":.4f",
            "Recall Macro": ":.4f",
            "Hamming Loss": ":.4f",
            "Jaccard Samples": ":.4f",
        },

        title=
            "Accuracy–F1 performance space",
    )


    fig.update_traces(
        textposition=
            "top center"
    )


    clean_fig(
        fig,
        height=500,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,

        config={
            "displayModeBar":
                False
        },
    )


    section(
        "Full test leaderboard",

        "Downloadable benchmark table.",
    )


    show = perf_df[
        [
            "Model",
            "Family",
            "Accuracy",
            "Subset Accuracy",
            "Precision Macro",
            "Recall Macro",
            "F1 Macro",
            "F1 Micro",
            "F1 Weighted",
            "Hamming Loss",
            "Jaccard Samples",
            "Elapsed Seconds",
        ]
    ].copy()


    st.dataframe(
        show
        .style
        .format({
            "Accuracy":
                "{:.4f}",

            "Subset Accuracy":
                "{:.4f}",

            "Precision Macro":
                "{:.4f}",

            "Recall Macro":
                "{:.4f}",

            "F1 Macro":
                "{:.4f}",

            "F1 Micro":
                "{:.4f}",

            "F1 Weighted":
                "{:.4f}",

            "Hamming Loss":
                "{:.4f}",

            "Jaccard Samples":
                "{:.4f}",

            "Elapsed Seconds":
                "{:.2f}",
        }),

        use_container_width=True,
        hide_index=True,
    )


    make_download(
        show,

        "leaderboard_ml_dl.csv",

        "Download leaderboard CSV",
    )


# ============================================================
# PAGE — PER LABEL
# ============================================================
elif page == "Per Label":

    page_header(
        "Label-Level Evaluation",

        "Where does each model succeed?",

        (
            "Inspect precision, recall, and F1 "
            "for each of the nine binary targets."
        ),
    )


    models = list(
        leaderboard_sorted[
            "Model"
        ]
    )


    default_models = [
        best_ml["Model"],
        best_dl["Model"],
    ]


    chosen_models = st.multiselect(
        "Models",

        models,

        default=
            default_models,
    )


    if not chosen_models:
        chosen_models = (
            default_models
        )


    label_metric = st.selectbox(
        "Metric",

        [
            "F1",
            "Precision",
            "Recall",
        ],
    )


    plot_df = per_label[
        per_label["Model"]
        .isin(
            chosen_models
        )
    ].copy()


    section(
        f"{label_metric} by label",

        (
            "Direct comparison across "
            "the selected models."
        ),
    )


    fig = px.bar(
        plot_df,

        x="Label Display",
        y=label_metric,

        color="Model",

        barmode="group",

        color_discrete_sequence=
            PASTEL_SEQUENCE,

        hover_data={
            "Support": True
        },

        title=
            f"{label_metric} "
            f"across nine targets",
    )


    fig.update_yaxes(
        range=[
            0,
            1.0,
        ]
    )


    fig.update_xaxes(
        tickangle=-30
    )


    clean_fig(
        fig,
        height=520,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,

        config={
            "displayModeBar":
                False
        },
    )


    section(
        "Heatmap",

        (
            "F1-score for every model "
            "and target."
        ),
    )


    heat = (
        per_label
        .pivot(
            index="Model",

            columns=
                "Label Display",

            values="F1",
        )
        .reindex([
            model

            for model
            in MODEL_ORDER

            if model
            in set(
                per_label[
                    "Model"
                ]
            )
        ])
    )


    fig = go.Figure(
        data=
            go.Heatmap(
                z=
                    heat.values,

                x=
                    heat.columns
                    .tolist(),

                y=
                    heat.index
                    .tolist(),

                colorscale=[
                    [
                        0.0,
                        "#FFFFFF",
                    ],
                    [
                        0.35,
                        "#EAF4FF",
                    ],
                    [
                        0.65,
                        "#B8E8F5",
                    ],
                    [
                        1.0,
                        "#5BC6A8",
                    ],
                ],

                zmin=0,
                zmax=1,

                text=
                    np.round(
                        heat.values,
                        3,
                    ),

                texttemplate=
                    "%{text:.3f}",

                hovertemplate=(
                    "Model=%{y}"
                    "<br>"
                    "Label=%{x}"
                    "<br>"
                    "F1=%{z:.4f}"
                    "<extra></extra>"
                ),

                colorbar=dict(
                    title="F1",
                    thickness=11,
                ),
            )
    )


    fig.update_layout(
        title=
            "Per-label F1 heatmap"
    )


    clean_fig(
        fig,
        height=590,
        legend=False,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,

        config={
            "displayModeBar":
                False
        },
    )


    section(
        "Label prevalence",

        (
            "Positive-label prevalence "
            "across train, validation, and test."
        ),
    )


    prev_plot = (
        prevalence
        .copy()
    )


    fig = px.line(
        prev_plot,

        x="Label Display",
        y="Prevalence",

        color="Split",

        markers=True,

        color_discrete_map=
            SPLIT_COLORS,

        title=
            "Label prevalence by split",
    )


    fig.update_xaxes(
        tickangle=-30
    )


    clean_fig(
        fig,
        height=450,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,

        config={
            "displayModeBar":
                False
        },
    )


    section(
        "Detailed per-label table"
    )


    table = plot_df[
        [
            "Model",
            "Family",
            "Label Display",
            "Support",
            "Precision",
            "Recall",
            "F1",
        ]
    ].copy()


    st.dataframe(
        table
        .style
        .format({
            "Precision":
                "{:.4f}",

            "Recall":
                "{:.4f}",

            "F1":
                "{:.4f}",
        }),

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

        (
            "Compare train, validation, and test behavior, "
            "generalization gaps, runtime, learning curves, "
            "and binary confusion matrices."
        ),
    )


    selected_diag_metric = st.selectbox(
        "Split metric",

        [
            "F1 Macro",
            "Accuracy",
        ],
    )


    section(
        "Train / validation / test",

        (
            "Large train-to-validation gaps "
            "should be interpreted as weaker generalization."
        ),
    )


    fig = px.bar(
        split_results,

        x="Model",
        y=selected_diag_metric,

        color="Split",

        barmode="group",

        color_discrete_map=
            SPLIT_COLORS,

        title=
            f"{selected_diag_metric} "
            f"across data splits",
    )


    fig.update_yaxes(
        range=[
            0,
            1.0,
        ]
    )


    clean_fig(
        fig,
        height=500,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,

        config={
            "displayModeBar":
                False
        },
    )


    section(
        "Generalization gap",

        (
            "Train F1-Macro minus "
            "validation F1-Macro."
        ),
    )


    gap_df = leaderboard_sorted[
        [
            "Model",
            "Family",
            "Train-Val F1 Gap",
        ]
    ].copy()


    gap_df = (
        gap_df
        .sort_values(
            "Train-Val F1 Gap"
        )
    )


    fig = px.bar(
        gap_df,

        x="Train-Val F1 Gap",
        y="Model",

        orientation="h",

        color="Family",

        color_discrete_map=
            FAMILY_COLORS,

        text=
            "Train-Val F1 Gap",

        title=
            "Train–validation F1 gap",
    )


    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
        cliponaxis=False,
    )


    clean_fig(
        fig,
        height=470,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,

        config={
            "displayModeBar":
                False
        },
    )


    section(
        "Training efficiency",

        (
            "Elapsed training time is shown "
            "on a logarithmic scale."
        ),
    )


    eff = (
        leaderboard_sorted
        .dropna(
            subset=[
                "Elapsed Seconds"
            ]
        )
        .copy()
    )


    fig = px.scatter(
        eff,

        x="Elapsed Seconds",
        y="F1 Macro",

        text="Model",

        color="Family",

        size="Accuracy",

        size_max=22,

        log_x=True,

        color_discrete_map=
            FAMILY_COLORS,

        hover_data={
            "Accuracy": ":.4f",
            "Elapsed Seconds": ":.2f",
        },

        title=
            "Training time vs test F1-Macro",
    )


    fig.update_traces(
        textposition=
            "top center"
    )


    clean_fig(
        fig,
        height=480,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,

        config={
            "displayModeBar":
                False
        },
    )


    learning_path = (
        ASSET_DIR
        / "12_learning_curves_DL.png"
    )


    if learning_path.exists():

        section(
            "Deep-learning learning curves",

            (
                "Original visualization exported "
                "from the comparison notebook."
            ),
        )

        st.image(
            str(
                learning_path
            ),
            use_container_width=True,
        )


    c1, c2 = st.columns(2)


    with c1:

        cm_ml = (
            ASSET_DIR
            / "14_confusion_best_ML.png"
        )

        if cm_ml.exists():

            section(
                "Best ML · Binary confusion matrices",

                (
                    f'{best_ml["Model"]} '
                    f'on the test set.'
                ),
            )

            st.image(
                str(cm_ml),
                use_container_width=True,
            )


    with c2:

        cm_dl = (
            ASSET_DIR
            / "15_confusion_best_DL.png"
        )

        if cm_dl.exists():

            section(
                "Best DL · Binary confusion matrices",

                (
                    f'{best_dl["Model"]} '
                    f'on the test set.'
                ),
            )

            st.image(
                str(cm_dl),
                use_container_width=True,
            )


# ============================================================
# PAGE — LIVE PREDICTION
# ============================================================
elif page == "Live Prediction":

    page_header(
        "Realtime Inference",

        "Predict a New Comment",

        (
            "Enter a new Indonesian comment to obtain "
            "a sentiment prediction and one or more detected aspects. "
            "The realtime engine uses the saved Linear SVM + TF-IDF "
            "artifacts from the completed comparison run."
        ),
    )


    st.markdown(
        """
        <div class="prediction-hero">

          <div class="prediction-title">
            New-comment prediction machine
          </div>

          <div class="prediction-sub">
            The benchmark winner remains CNN based on F1-Macro.
            For realtime deployment, this page uses Linear SVM —
            the best Machine Learning model — because the fitted
            TF-IDF + SVM artifacts are compact and suitable for
            fast inference on Streamlit.
          </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    if not live_predictor_ready():

        st.error(
            (
                "Live prediction files are not complete. "
                "Add both files below to the project:\n\n"
                "models/Linear_SVM.joblib\n"
                "models/tfidf_vectorizer.joblib"
            )
        )

    else:

        sample_options = {
            "Custom":
                "",

            "Pregnancy stress":
                (
                    "aku akhir akhir ini capek banget "
                    "selama hamil sering menangis dan "
                    "takut menjelang lahiran"
                ),

            "External support":
                (
                    "selama hamil aku merasa mengurus "
                    "semuanya sendiri dan suami jarang membantu"
                ),

            "Financial":
                (
                    "aku khawatir biaya lahiran mahal "
                    "karena kondisi keuangan keluarga lagi sulit"
                ),

            "Governance":
                (
                    "aku bingung soal rujukan bpjs "
                    "dan pelayanan di puskesmas "
                    "untuk periksa kehamilan"
                ),

            "App technical":
                (
                    "aku sudah pakai aplikasi ini "
                    "tapi susah login dan "
                    "beberapa fiturnya sering error"
                ),
        }


        example = st.selectbox(
            "Quick example",

            list(
                sample_options
                .keys()
            ),
        )


        if (
            "live_comment"
            not in
            st.session_state
        ):
            st.session_state[
                "live_comment"
            ] = ""


        if (
            "last_example"
            not in
            st.session_state
        ):
            st.session_state[
                "last_example"
            ] = "Custom"


        if (
            example
            !=
            st.session_state[
                "last_example"
            ]
        ):

            st.session_state[
                "last_example"
            ] = example

            st.session_state[
                "live_comment"
            ] = (
                sample_options[
                    example
                ]
            )


        raw_comment = st.text_area(
            "New comment",

            key=
                "live_comment",

            height=150,

            placeholder=(
                "Contoh: aku akhir akhir ini "
                "capek banget selama hamil "
                "dan sering takut menjelang lahiran..."
            ),
        )


        c1, c2 = st.columns(
            [
                1,
                4,
            ]
        )


        with c1:
            predict_clicked = st.button(
                "Predict Comment",

                use_container_width=True,
            )


        with c2:
            st.caption(
                (
                    "Gunakan satu kalimat atau komentar "
                    "yang memiliki konteks cukup jelas."
                )
            )


        if predict_clicked:

            if not raw_comment.strip():

                st.warning(
                    (
                        "Masukkan komentar "
                        "terlebih dahulu."
                    )
                )

            else:

                try:

                    result = (
                        predict_new_comment(
                            raw_comment
                        )
                    )

                except Exception as e:

                    st.error(
                        (
                            "Prediction could not "
                            f"be generated: {e}"
                        )
                    )

                else:

                    sentiment = (
                        SENTIMENT_DISPLAY[
                            result[
                                "sentiment_key"
                            ]
                        ]
                    )

                    aspects = (
                        result[
                            "detected_aspects"
                        ]
                    )


                    section(
                        "Prediction result",

                        (
                            "Model decision on "
                            "the normalized new comment."
                        ),
                    )


                    c1, c2, c3 = (
                        st.columns(3)
                    )


                    with c1:
                        metric_card(
                            "Predicted Sentiment",

                            sentiment,

                            (
                                "Highest sentiment "
                                "decision margin"
                            ),

                            "blue",
                        )


                    with c2:
                        metric_card(
                            "Detected Aspects",

                            str(
                                len(
                                    aspects
                                )
                            ),

                            (
                                "Multi-label "
                                "aspect output"
                            ),

                            "mint",
                        )


                    with c3:
                        metric_card(
                            "Inference Engine",

                            "Linear SVM",

                            "Best ML model",

                            "yellow",
                        )


                    section(
                        "Detected aspect(s)",

                        (
                            "One comment may activate "
                            "more than one aspect."
                        ),
                    )


                    if aspects:

                        chip_html = (
                            '<div class="chip-wrap">'
                        )


                        chip_html += "".join(
                            (
                                '<span '
                                'class="result-chip chip-mint">'
                                f'{html.escape(ASPECT_DISPLAY[a])}'
                                '</span>'
                            )

                            for a
                            in aspects
                        )


                        chip_html += "</div>"


                        st.markdown(
                            chip_html,
                            unsafe_allow_html=True,
                        )


                    else:

                        fallback = (
                            result[
                                "top_aspect_fallback"
                            ]
                        )


                        st.markdown(
                            (
                                '<div class="chip-wrap">'

                                '<span '
                                'class="result-chip chip-yellow">'
                                'No aspect crossed the binary boundary'
                                '</span>'

                                '<span '
                                'class="result-chip chip-blue">'
                                'Closest: '
                                f'{html.escape(ASPECT_DISPLAY[fallback])}'
                                '</span>'

                                '</div>'
                            ),

                            unsafe_allow_html=True,
                        )


                    with st.expander(
                        "Show preprocessing and model details"
                    ):

                        st.markdown(
                            "**Normalized text used by the model**"
                        )


                        st.code(
                            result[
                                "clean_text"
                            ],

                            language=None,
                        )


                        score_df = pd.DataFrame({
                            "Label": [
                                LABEL_DISPLAY[
                                    x
                                ]

                                for x
                                in TARGET_COLS
                            ],

                            "Binary Prediction":
                                result[
                                    "binary_pred"
                                ],

                            "SVM Margin":
                                result[
                                    "margins"
                                ],

                            "Relative Score":
                                result[
                                    "relative_scores"
                                ],
                        })


                        score_df["Type"] = (
                            ["Sentiment"] * 3
                            +
                            ["Aspect"] * 6
                        )


                        st.dataframe(
                            score_df
                            .style
                            .format({
                                "SVM Margin":
                                    "{:+.4f}",

                                "Relative Score":
                                    "{:.3f}",
                            }),

                            use_container_width=True,
                            hide_index=True,
                        )


                        fig = px.bar(
                            score_df,

                            x=
                                "Relative Score",

                            y=
                                "Label",

                            orientation="h",

                            color="Type",

                            color_discrete_map={
                                "Sentiment":
                                    COLORS[
                                        "blue"
                                    ],

                                "Aspect":
                                    COLORS[
                                        "mint"
                                    ],
                            },

                            title=
                                "Relative model score",

                            range_x=[
                                0,
                                1,
                            ],
                        )


                        clean_fig(
                            fig,
                            height=500,
                        )


                        st.plotly_chart(
                            fig,

                            use_container_width=True,

                            config={
                                "displayModeBar":
                                    False
                            },
                        )


                        st.caption(
                            (
                                "Relative Score is derived "
                                "from the Linear SVM decision margin "
                                "with a sigmoid transform. "
                                "It is shown only for relative "
                                "visualization and is not a "
                                "calibrated probability."
                            )
                        )


                    st.markdown(
                        """
                        <div
                            class="note"
                            style="margin-top:14px;"
                        >
                            <b>Interpretation boundary:</b>
                            this predictor classifies text
                            according to the silver-label scheme
                            used in the research. It is not a
                            medical diagnosis, clinical screening
                            tool, or estimate of an individual's
                            psychological condition.
                        </div>
                        """,

                        unsafe_allow_html=True,
                    )


# ============================================================
# PAGE — PREDICTION EXPLORER
# ============================================================
elif page == "Prediction Explorer":

    page_header(
        "Held-out Test Set",

        "Prediction Explorer",

        (
            "Browse exported predictions for the "
            "best ML and best DL models. "
            "This page uses saved held-out predictions."
        ),
    )


    if not prediction_files:

        st.info(
            (
                "Prediction CSV files "
                "are not available."
            )
        )


    else:

        selected_model = st.selectbox(
            "Prediction file",

            list(
                prediction_files
                .keys()
            ),
        )


        pred_df = (
            prediction_files[
                selected_model
            ]
            .copy()
        )


        exact_rate = float(
            pred_df[
                "exact_match"
            ]
            .mean()
        )


        error_count = int(
            (
                pred_df[
                    "exact_match"
                ]
                ==
                0
            )
            .sum()
        )


        c1, c2, c3 = (
            st.columns(3)
        )


        with c1:
            metric_card(
                "Samples",

                f"{len(pred_df):,}",

                (
                    "Held-out "
                    "test examples"
                ),

                "blue",
            )


        with c2:
            metric_card(
                "Exact-match Rate",

                f"{exact_rate:.2%}",

                (
                    "All nine binary "
                    "decisions correct"
                ),

                "mint",
            )


        with c3:
            metric_card(
                "Exact-match Errors",

                f"{error_count:,}",

                (
                    "Samples with "
                    "≥1 mismatched label"
                ),

                "coral",
            )


        section(
            "Filter examples"
        )


        c1, c2, c3 = st.columns(
            [
                1,
                1,
                2,
            ]
        )


        with c1:
            only_errors = st.checkbox(
                "Errors only",
                value=True,
            )


        with c2:
            only_exact = st.checkbox(
                "Exact matches only",
                value=False,
            )


        with c3:
            search_text = st.text_input(
                "Search comment text",

                placeholder=(
                    "e.g. suami, "
                    "puskesmas, aplikasi..."
                ),
            )


        view = (
            pred_df
            .copy()
        )


        if (
            only_errors
            and
            not only_exact
        ):

            view = view[
                view[
                    "exact_match"
                ]
                ==
                0
            ]


        elif (
            only_exact
            and
            not only_errors
        ):

            view = view[
                view[
                    "exact_match"
                ]
                ==
                1
            ]


        if search_text.strip():

            view = view[
                view["text"]
                .astype(str)
                .str.contains(
                    search_text.strip(),

                    case=False,
                    na=False,
                )
            ]


        max_rows = min(
            max(
                len(view),
                1,
            ),
            200,
        )


        n_show = st.slider(
            "Rows to display",

            min_value=5,

            max_value=
                max(
                    5,
                    max_rows,
                ),

            value=
                min(
                    25,
                    max(
                        5,
                        max_rows,
                    ),
                ),

            step=5,
        )


        display_cols = [
            "text",
            "actual_labels",
            "predicted_labels",
            "exact_match",
        ]


        st.dataframe(
            view[
                display_cols
            ]
            .head(
                n_show
            ),

            use_container_width=True,
            hide_index=True,

            column_config={
                "text":
                    st.column_config.TextColumn(
                        "Comment",
                        width="large",
                    ),

                "actual_labels":
                    st.column_config.TextColumn(
                        "Actual",
                        width="large",
                    ),

                "predicted_labels":
                    st.column_config.TextColumn(
                        "Predicted",
                        width="large",
                    ),

                "exact_match":
                    st.column_config.NumberColumn(
                        "Exact",
                        format="%d",
                    ),
            },
        )


        make_download(
            view,

            (
                "prediction_explorer_"
                f'{selected_model.replace(" ", "_")}'
                ".csv"
            ),

            "Download filtered rows",
        )


# ============================================================
# PAGE — METHOD
# ============================================================
elif page == "Method":

    page_header(
        "Experiment Design",

        "Method & Pipeline",

        (
            "The supplementary benchmark keeps "
            "the final TA data pipeline fixed and "
            "changes the learner / representation "
            "required by each model family."
        ),
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:
        metric_card(
            "Target",

            "9 binary labels",

            (
                "3 sentiment + "
                "6 aspect labels"
            ),

            "blue",
        )


    with c2:
        metric_card(
            "Split",

            "80 / 10 / 10",

            (
                "Stratified by "
                "sentiment label"
            ),

            "mint",
        )


    with c3:
        metric_card(
            "Random State",

            "42",

            "Fixed experiment seed",

            "yellow",
        )


    with c4:
        metric_card(
            "Primary Metric",

            "F1-Macro",

            (
                "All labels receive "
                "equal weight"
            ),

            "coral",
        )


    section(
        "Fixed upstream pipeline",

        (
            "The model comparison does not rebuild "
            "a different data-processing protocol."
        ),
    )


    steps = [
        (
            "01",
            "Scraping",
            "122,282 raw TikTok comments",
        ),

        (
            "02",
            "Preprocessing",
            (
                "Cleaning, normalization, "
                "decomposition, filtering"
            ),
        ),

        (
            "03",
            "Quality filtering",
            (
                "31,780 comments retained "
                "for labeling"
            ),
        ),

        (
            "04",
            "Silver labeling",
            "Sentiment + six aspects",
        ),

        (
            "05",
            "Final transformation",
            (
                "19,443 modeling samples · "
                "nine binary targets"
            ),
        ),

        (
            "06",
            "Split",
            (
                "15,554 train · "
                "1,944 validation · "
                "1,945 test"
            ),
        ),

        (
            "07",
            "Representation",
            (
                "TF-IDF for ML · "
                "IndoBERT tokenizer IDs for DL"
            ),
        ),

        (
            "08",
            "Model training",
            "4 ML + 4 DL models",
        ),

        (
            "09",
            "Evaluation",
            (
                "F1-Macro primary, "
                "supporting multi-label metrics"
            ),
        ),
    ]


    for (
        num,
        title,
        desc,
    ) in steps:

        st.markdown(
            f"""
            <div style="
                display:flex;
                gap:14px;
                align-items:flex-start;
                border-bottom:1px solid #EFF4F8;
                padding:11px 0;
            ">

                <div style="
                    font-size:10px;
                    color:#55A6FF;
                    width:28px;
                    padding-top:2px;
                    font-weight:700;
                ">
                    {num}
                </div>

                <div>

                    <div style="
                        font-size:12px;
                        font-weight:680;
                        color:#334155;
                    ">
                        {html.escape(title)}
                    </div>

                    <div style="
                        font-size:11px;
                        color:#64748B;
                        margin-top:3px;
                        line-height:1.55;
                    ">
                        {html.escape(desc)}
                    </div>

                </div>
            </div>
            """,

            unsafe_allow_html=True,
        )


    pipeline_img = (
        ASSET_DIR
        / "01_pipeline_funnel_aktual_TA.png"
    )


    if pipeline_img.exists():

        section(
            "Pipeline funnel",

            (
                "Original figure exported "
                "by the comparison notebook."
            ),
        )

        st.image(
            str(
                pipeline_img
            ),

            use_container_width=True,
        )


    section(
        "Models in this benchmark"
    )


    model_table = pd.DataFrame(
        [
            [
                "Machine Learning",
                "Logistic Regression",
                "TF-IDF",
                "One-vs-rest",
            ],

            [
                "Machine Learning",
                "Multinomial Naive Bayes",
                "TF-IDF",
                "One-vs-rest",
            ],

            [
                "Machine Learning",
                "Linear SVM",
                "TF-IDF",
                "One-vs-rest",
            ],

            [
                "Machine Learning",
                "Random Forest",
                "TF-IDF",
                "One-vs-rest",
            ],

            [
                "Deep Learning",
                "CNN",
                "IndoBERT tokenizer IDs",
                "Trainable embedding",
            ],

            [
                "Deep Learning",
                "LSTM",
                "IndoBERT tokenizer IDs",
                "Trainable embedding",
            ],

            [
                "Deep Learning",
                "BiLSTM",
                "IndoBERT tokenizer IDs",
                "Trainable embedding",
            ],

            [
                "Deep Learning",
                "GRU",
                "IndoBERT tokenizer IDs",
                "Trainable embedding",
            ],
        ],

        columns=[
            "Family",
            "Model",
            "Input Representation",
            "Notes",
        ],
    )


    st.dataframe(
        model_table,

        use_container_width=True,
        hide_index=True,
    )


    section(
        "Realtime predictor"
    )


    st.markdown(
        f"""
        <div class="note">
        The live-prediction page uses the saved
        <b>Linear SVM</b> + fitted <b>TF-IDF</b>
        artifacts from the same benchmark.
        Linear SVM is the best Machine Learning
        model in the comparison
        (F1-Macro <b>{float(best_ml["F1 Macro"]):.4f}</b>;
        Accuracy <b>{float(best_ml["Accuracy"]):.4f}</b>).

        <br><br>

        The overall benchmark winner remains
        <b>{html.escape(str(best_overall["Model"]))}</b>
        based on the primary metric
        (F1-Macro
        <b>{float(best_overall["F1 Macro"]):.4f}</b>).
        </div>
        """,

        unsafe_allow_html=True,
    )


    section(
        "Interpretation boundary"
    )


    st.markdown(
        """
        <div class="note">
        <b>Silver-label benchmark.</b>
        These scores quantify agreement with the
        final silver labels used in the TA pipeline.
        They are not gold-label clinical accuracy
        and must not be interpreted as diagnosis,
        screening performance, or prevalence of
        perinatal mental-health conditions.
        </div>
        """,

        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    "<br><br>",
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div style="
        border-top:1px solid #EFF4F8;
        padding-top:15px;
        color:#94A3B8;
        font-size:10px;
        display:flex;
        justify-content:space-between;
        gap:12px;
        flex-wrap:wrap;
    ">
      <span>
        ML / DL Comparison Dashboard
      </span>

      <span>
        Bright analytics · realtime ML inference · 2026
      </span>
    </div>
    """,

    unsafe_allow_html=True,
)