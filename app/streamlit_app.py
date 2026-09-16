"""
Financial Assistant Web Application - Streamlit Frontend
Tabs: Chat | Data Browser | ML Predictions

Powered by Vertex AI (Gemini) function calling, Databricks SQL Warehouse,
Postgres on Supabase, AWS SageMaker, and AWS Bedrock.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import html
import time
import traceback
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

# Streamlit Cloud stores config in st.secrets, not os.environ — bridge them so
# the agent/tool modules (which only read os.environ) see the same values.
try:
    for _key, _value in st.secrets.items():
        if _key not in os.environ:
            os.environ[_key] = str(_value)
except Exception:
    pass

st.set_page_config(
    page_title="Prologis Financial Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent.parent
SEC_PATH = ROOT / "data" / "sec" / "prologis_financials.json"
PRESS_PATH = ROOT / "data" / "press_releases.json"

GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")
DATABRICKS_HOST = os.getenv("DATABRICKS_SERVER_HOSTNAME", "")
DATABRICKS_TABLE = "workspace.default.lease_transactions"
GENIE_SPACE = "Commercial Lease Analytics"

# --------------------------------------------------------------
# Static metadata about the agent's tools / data sources.
# Keep in sync with agent/agent.py TOOL_FUNCTIONS.
# --------------------------------------------------------------
TOOL_META = {
    "query_databricks": {
        "icon": "🧱", "label": "Databricks", "color": "#ff3621",
        "service": "Serverless SQL Warehouse · Delta table",
        "table": DATABRICKS_TABLE,
        "records": "25 lease transactions · 11 metros · 6 tenant industries",
        "ask": "tenants, rent per sq ft, annual rent, lease terms, lease types",
        "args": "metro_area · tenant_industry · min_annual_rent",
    },
    "query_postgres": {
        "icon": "🐘", "label": "Postgres", "color": "#3ecf8e",
        "service": "Supabase · Session pooler",
        "table": "properties ⋈ financials",
        "records": "20 properties · 11 metros · Industrial / Logistics / Warehouse",
        "ask": "property revenue, net income, expenses by metro or type",
        "args": "metro_area · property_type · min_revenue",
    },
    "query_sec_edgar": {
        "icon": "📑", "label": "SEC EDGAR", "color": "#60a5fa",
        "service": "XBRL Company Facts API · cached JSON",
        "table": "Prologis (NYSE: PLD) 10-K / 10-Q",
        "records": "revenue · net income · opex · total assets · total liabilities",
        "ask": "real, audited company-level financials (annual or quarterly)",
        "args": "metric · period",
    },
    "query_press_releases": {
        "icon": "📰", "label": "Press Releases", "color": "#fbbf24",
        "service": "JSON store",
        "table": "data/press_releases.json",
        "records": "10 releases · earnings / acquisition / expansion / sustainability",
        "ask": "announcements, deals, expansions, ESG updates",
        "args": "keywords · category · limit",
    },
    "summarize_with_bedrock": {
        "icon": "📝", "label": "AWS Bedrock", "color": "#f59e0b",
        "service": "Claude Haiku 4.5 · us-east-1 inference profile",
        "table": "summarization tool (no data of its own)",
        "records": "condenses any tool output to N words",
        "ask": "“summarize …”, “in 40 words”, “briefly”",
        "args": "text · max_words",
    },
}

# Clickable suggestions, grouped by the data source they exercise. Every group
# is always visible in the Chat tab — not just before the first question.
SUGGESTED_QUERIES = {
    "🧱 Databricks · Leases": [
        "Show me lease transactions in Chicago",
        "What's the average rent per square foot for Cold Storage tenants?",
        "Which leases have annual rent above $3M?",
        "List E-commerce tenant leases with their lease terms and lease types",
        "Compare total annual rent between Dallas and Los Angeles leases",
        "Which 3PL tenants have the largest square footage?",
    ],
    "🐘 Postgres · Properties": [
        "Show industrial properties in Chicago with revenue",
        "Compare property revenues between Dallas and Phoenix",
        "Which warehouse properties generate more than $5M in revenue?",
        "Which metro has the highest average revenue per property?",
        "List logistics properties in Los Angeles with net income",
        "What is the total revenue across all Seattle properties?",
    ],
    "📑 SEC EDGAR · Financials": [
        "What was Prologis' net income last year?",
        "What were Prologis' revenue and operating expenses in the latest 10-K?",
        "Show Prologis' total assets versus total liabilities",
        "What was Prologis' revenue in the most recent quarter?",
    ],
    "📰 Press Releases + Bedrock": [
        "Did Prologis announce any acquisitions recently?",
        "Summarize the most recent earnings press release in 40 words",
        "What sustainability initiatives has Prologis announced?",
        "Briefly summarize recent expansion announcements",
    ],
    "🔀 Multi-source": [
        "Compare Chicago property revenue in Postgres with Chicago lease rent in Databricks",
        "How does total lease rent in Dallas compare to Dallas property revenue?",
        "What was Prologis' net income last year, and what did the latest earnings release say? Keep it brief.",
        "Which metros appear in both the property database and the lease transactions?",
    ],
}

# --------------------------------------------------------------
# Custom CSS — futuristic, glassy, professional
# --------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

    /* Page background — radial gradient over deep navy */
    .stApp {
        background:
            radial-gradient(circle at 20% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 100%, rgba(255, 54, 33, 0.05) 0%, transparent 50%),
            #0a0e1a;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* ---------------------------------------------------------------
       Theme lock. The page background above is always dark, so text and
       surface colors must not follow Streamlit's Light theme (which the
       "Use system setting" option resolves to on a light OS). Pin them.
       --------------------------------------------------------------- */
    .stApp { color: #e2e8f0; color-scheme: dark; }

    /* Body text, lists, captions, headings */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessageContent"] p { color: #e2e8f0; }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p { color: #94a3b8 !important; }
    [data-testid="stHeading"] h2, [data-testid="stHeading"] h3,
    [data-testid="stHeading"] h4, [data-testid="stHeading"] h5,
    [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4, [data-testid="stMarkdownContainer"] h5 { color: #f0f9ff; }
    [data-testid="stMarkdownContainer"] a { color: #67e8f9; }
    [data-testid="stMarkdownContainer"] strong { color: #f0f9ff; }

    /* Inline code + code blocks */
    [data-testid="stMarkdownContainer"] code:not(pre code),
    [data-testid="stCaptionContainer"] code:not(pre code) {
        color: #67e8f9;
        background: rgba(34, 211, 238, 0.08);
        border: 1px solid rgba(34, 211, 238, 0.15);
        border-radius: 6px;
    }
    [data-testid="stCode"] pre, [data-testid="stCode"] code, .stCodeBlock pre {
        background: rgba(15, 23, 42, 0.85) !important;
        color: #67e8f9 !important;
    }
    [data-testid="stCode"] { border: 1px solid rgba(34, 211, 238, 0.15); border-radius: 8px; }

    /* Top header bar + toolbar icons (white in Light theme otherwise) */
    [data-testid="stHeader"] { background: rgba(10, 14, 26, 0.92); }
    [data-testid="stHeader"] button, [data-testid="stHeader"] a,
    [data-testid="stHeader"] span, [data-testid="stHeader"] svg,
    [data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebarCollapsedControl"] button, [data-testid="stSidebarCollapsedControl"] svg {
        color: #cbd5e1 !important; fill: #cbd5e1 !important;
    }

    /* Expanders — the inner <details> carries the theme's background */
    [data-testid="stExpander"] details,
    [data-testid="stExpanderDetails"] { background: transparent !important; border-color: rgba(148, 163, 184, 0.1) !important; }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary svg { color: #94a3b8 !important; fill: #94a3b8 !important; }
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:hover p { color: #e2e8f0 !important; }

    /* Widget labels, radios, metrics, dividers, chat avatars */
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
    .stRadio label p, .stSlider label, .stSelectbox label,
    .stNumberInput label, .stMultiSelect label { color: #cbd5e1 !important; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: #94a3b8 !important; }
    hr { border-color: rgba(148, 163, 184, 0.15) !important; }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] { background: rgba(99, 102, 241, 0.25); color: #f0f9ff; }

    /* Form inputs / selects (white boxes in Light theme otherwise) */
    [data-testid="stForm"] { border-color: rgba(148, 163, 184, 0.12); }
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div,
    [data-testid="stNumberInputContainer"],
    .stNumberInput input {
        background: rgba(30, 41, 59, 0.6) !important;
        border-color: rgba(129, 140, 248, 0.25) !important;
        color: #f0f9ff !important;
    }
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] input,
    .stMultiSelect [data-baseweb="select"] input,
    .stSelectbox [data-baseweb="select"] svg,
    .stMultiSelect [data-baseweb="select"] svg { color: #f0f9ff !important; fill: #cbd5e1 !important; }
    .stMultiSelect [data-baseweb="tag"] { background: rgba(99, 102, 241, 0.35) !important; color: #f0f9ff !important; }
    .stMultiSelect [data-baseweb="tag"] span { color: #f0f9ff !important; }

    /* Buttons and tabs keep their own colors regardless of markdown rules */
    .stButton button p, [data-testid="stFormSubmitButton"] button p,
    .stTabs [data-baseweb="tab"] p { color: inherit !important; }


    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(10, 14, 26, 0.98) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    [data-testid="stSidebar"] h1 {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #22d3ee 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.6rem;
        letter-spacing: -0.02em;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(148, 163, 184, 0.15);
        margin: 1.2rem 0;
    }
    [data-testid="stSidebar"] h3 {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.6rem;
    }
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] p { color: #e2e8f0; font-size: 0.88rem; margin-bottom: 0.15rem; }
    [data-testid="stSidebar"] code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #67e8f9;
        background: rgba(34, 211, 238, 0.08);
        border: 1px solid rgba(34, 211, 238, 0.15);
        border-radius: 6px;
    }

    /* Main heading gradient (Streamlit >=1.36 has no `.main` class — use test ids) */
    [data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
    [data-testid="stMain"] h4, [data-testid="stMain"] h5,
    .main h1, .main h2 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em; }
    [data-testid="stMain"] h1, .main h1 {
        background: linear-gradient(135deg, #f0f9ff 0%, #a5b4fc 50%, #67e8f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 24px;
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(8px);
        border-radius: 10px 10px 0 0;
        border: 1px solid rgba(148, 163, 184, 0.08);
        border-bottom: none;
        color: #94a3b8;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { background: rgba(99, 102, 241, 0.1); color: #e2e8f0; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(34, 211, 238, 0.15) 100%) !important;
        color: #f0f9ff !important;
        border-color: rgba(129, 140, 248, 0.3) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.15);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: linear-gradient(90deg, #6366f1 0%, #22d3ee 100%);
        height: 3px;
        border-radius: 3px 3px 0 0;
    }
    .stTabs [data-baseweb="tab-border"] { background: rgba(148, 163, 184, 0.15); }
    /* Nested tab strips (suggestion groups, data sources): denser, and wrap instead of clipping */
    .stTabs .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; row-gap: 4px; }
    .stTabs .stTabs [data-baseweb="tab"] { height: 40px; padding: 0 14px; font-size: 0.86rem; }

    /* Warehouse status pill */
    .wh { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; font-size: 0.8rem; color: #cbd5e1; }
    .wh .pill {
        display: inline-flex; align-items: center; gap: 0.3rem;
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600;
        border-radius: 999px; padding: 0.15rem 0.6rem;
        border: 1px solid var(--c); color: var(--c); background: rgba(255,255,255,0.03);
    }
    .wh .meta { color: #94a3b8; font-size: 0.74rem; }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }

    /* Primary buttons (Ask / Predict) — gradient */
    .stButton button[kind="primary"],
    .stButton [data-testid="baseButton-primary"],
    .stButton [data-testid="stBaseButton-primary"],
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        background-size: 200% 200%;
        color: white;
        border: none;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        padding: 0.6rem 1.6rem;
        border-radius: 10px;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25);
        transition: all 0.2s ease;
    }
    .stButton button[kind="primary"]:hover,
    .stButton [data-testid="baseButton-primary"]:hover,
    .stButton [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 28px rgba(99, 102, 241, 0.4);
        background-position: 100% 0%;
    }

    /* Secondary buttons (clickable suggested queries) — quiet glass chips */
    .stButton button[kind="secondary"],
    .stButton [data-testid="baseButton-secondary"],
    .stButton [data-testid="stBaseButton-secondary"] {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 10px;
        color: #cbd5e1;
        font-weight: 400;
        font-size: 0.86rem;
        text-align: left;
        justify-content: flex-start;
        padding: 0.5rem 0.9rem;
        min-height: 2.6rem;
        transition: all 0.15s ease;
    }
    .stButton button[kind="secondary"] p { text-align: left; }
    .stButton button[kind="secondary"]:hover {
        background: rgba(99, 102, 241, 0.16);
        border-color: rgba(129, 140, 248, 0.45);
        color: #f0f9ff;
        transform: translateY(-1px);
    }

    /* Form: text input on top (style the baseweb wrapper too — it carries the theme bg) */
    .stTextInput [data-baseweb="input"],
    .stTextInput [data-baseweb="base-input"] {
        background: rgba(30, 41, 59, 0.6) !important;
        border-color: rgba(129, 140, 248, 0.25) !important;
        border-radius: 12px !important;
    }
    .stTextInput input::placeholder { color: #64748b !important; }
    .stTextInput input {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(129, 140, 248, 0.25) !important;
        border-radius: 12px !important;
        color: #f0f9ff !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
    }
    .stTextInput input:focus {
        border-color: rgba(34, 211, 238, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.15) !important;
    }

    /* Horizontal radio (suggestion group picker) */
    .stRadio [role="radiogroup"] { gap: 0.4rem 1.1rem; flex-wrap: wrap; }
    .stRadio label { color: #cbd5e1; font-size: 0.86rem; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 14px 18px;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #67e8f9 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.3);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary { color: #94a3b8; font-size: 0.85rem; }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 10px;
        overflow: hidden;
    }

    code, pre { font-family: 'JetBrains Mono', monospace !important; }

    /* ---- Architecture / data-source cards ---- */
    .flow {
        display: flex; flex-wrap: wrap; align-items: center; gap: 0.45rem;
        font-family: 'JetBrains Mono', monospace; font-size: 0.74rem;
        color: #94a3b8; margin: 0.2rem 0 0.9rem 0;
    }
    .flow .node {
        background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px; padding: 0.28rem 0.6rem; color: #e2e8f0; white-space: nowrap;
    }
    .flow .node.brain {
        border-color: rgba(129, 140, 248, 0.5);
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(34, 211, 238, 0.15) 100%);
        color: #f0f9ff;
    }
    .flow .arrow { color: #64748b; }

    .src-grid {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
        gap: 0.7rem; margin-bottom: 0.4rem;
    }
    .src-card {
        background: rgba(30, 41, 59, 0.45); backdrop-filter: blur(8px);
        border: 1px solid rgba(148, 163, 184, 0.12); border-left: 3px solid var(--c);
        border-radius: 12px; padding: 0.8rem 0.95rem; min-height: 100%;
    }
    .src-card .name { font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #f0f9ff; font-size: 0.98rem; }
    .src-card .svc { color: #94a3b8; font-size: 0.74rem; margin-top: 0.1rem; }
    .src-card .tbl {
        font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--c);
        margin-top: 0.45rem; word-break: break-all;
    }
    .src-card .rec { color: #cbd5e1; font-size: 0.78rem; margin-top: 0.4rem; line-height: 1.35; }
    .src-card .ask { color: #94a3b8; font-size: 0.74rem; margin-top: 0.45rem; font-style: italic; line-height: 1.35; }
    .src-card .ask b { color: #cbd5e1; font-style: normal; }

    .genie {
        margin-top: 0.7rem; padding: 0.7rem 0.95rem; border-radius: 12px;
        background: linear-gradient(135deg, rgba(255, 54, 33, 0.10) 0%, rgba(30, 41, 59, 0.45) 100%);
        border: 1px solid rgba(255, 54, 33, 0.25); font-size: 0.8rem; color: #cbd5e1; line-height: 1.45;
    }
    .genie b { color: #f0f9ff; }
    .genie code { font-size: 0.72rem; color: #fca5a5; background: rgba(255, 54, 33, 0.1); border-radius: 5px; padding: 0 0.3rem; }

    /* ---- Source chips shown on each answer ---- */
    .chips { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.15rem 0 0.55rem 0; }
    .chip {
        display: inline-flex; align-items: center; gap: 0.3rem;
        font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
        color: var(--c); background: rgba(255,255,255,0.03);
        border: 1px solid var(--c); border-radius: 999px; padding: 0.15rem 0.6rem;
        opacity: 0.9;
    }

    /* ---- Sidebar connection status ---- */
    .status { display: grid; grid-template-columns: 1fr 1fr; gap: 0.3rem 0.6rem; font-size: 0.8rem; color: #cbd5e1; }
    .status .ok::before  { content: "●"; color: #34d399; margin-right: 0.4rem; }
    .status .off::before { content: "○"; color: #64748b; margin-right: 0.4rem; }
    .status .off { color: #64748b; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
@st.cache_resource
def get_db_engine():
    user = os.getenv("POSTGRES_USER", "postgres")
    pw = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "financial_assistant")
    return create_engine(f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}")


@st.cache_data(ttl=300, show_spinner=False)
def load_properties() -> pd.DataFrame:
    sql = """
        SELECT p.property_id, p.address, p.metro_area, p.sq_footage,
               p.property_type, f.revenue, f.net_income, f.expenses
        FROM properties p
        LEFT JOIN financials f ON p.property_id = f.property_id
        ORDER BY p.property_id
    """
    return pd.read_sql(sql, get_db_engine())


@st.cache_data(ttl=300, show_spinner=False)
def load_lease_transactions() -> pd.DataFrame:
    """Pull the full Databricks lease table via the same tool the agent uses."""
    from agent.tools import query_databricks
    result = query_databricks(limit=200)
    return pd.DataFrame(result.get("records", []))


@st.cache_data(show_spinner=False)
def load_json(path_str: str):
    p = Path(path_str)
    return json.loads(p.read_text()) if p.exists() else None


def invoke_sagemaker(endpoint_name, payload):
    import boto3
    region = os.getenv("AWS_REGION", "us-east-1")
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode())


def safe_md(text):
    if not text:
        return text
    return text.replace("$", "\\$")


def env_set(*names) -> bool:
    return all(os.getenv(n) for n in names)


def source_chip(tool_name: str) -> str:
    m = TOOL_META.get(tool_name, {"icon": "🔧", "label": tool_name, "color": "#94a3b8"})
    return (f'<span class="chip" style="--c:{m["color"]}">{m["icon"]} '
            f'{html.escape(m["label"])} · {html.escape(tool_name)}</span>')


def render_tool_result(result):
    """Show a tool's return value in the most readable form available."""
    if isinstance(result, str):
        st.markdown(safe_md(result))
        return
    if not isinstance(result, dict):
        st.json(result, expanded=False)
        return
    if "error" in result:
        st.error(result["error"])
        return
    rows = result.get("records") or result.get("properties") or result.get("results")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if result.get("summary"):
            st.json(result["summary"], expanded=False)
        return
    if "releases" in result:
        for r in result["releases"]:
            st.markdown(f"- **{r.get('date', '')}** · `{r.get('category', '')}` — {safe_md(r.get('title', ''))}")
        return
    st.json(result, expanded=False)


def run_query(prompt: str):
    """Send a question to the Vertex AI agent and prepend the exchange."""
    prompt = (prompt or "").strip()
    if not prompt:
        return
    try:
        from agent.agent import run_agent
        with st.spinner("Routing through Vertex AI Gemini…"):
            result = run_agent(prompt)
        st.session_state.messages.insert(0, {
            "user": prompt,
            "assistant": result["answer"],
            "tool_calls": result["tool_calls"],
        })
    except Exception as e:
        st.session_state.messages.insert(0, {
            "user": prompt,
            "assistant": f"⚠️ Agent error: {e}",
            "tool_calls": [],
            "traceback": traceback.format_exc(),
        })
    st.session_state.input_counter += 1
    st.rerun()


# --------------------------------------------------------------
# Databricks SQL Warehouse status / wake-up
# --------------------------------------------------------------
WH_COLORS = {"RUNNING": "#34d399", "STARTING": "#fbbf24", "STOPPED": "#94a3b8",
             "STOPPING": "#fb923c", "ERROR": "#f87171"}
DATABRICKS_CONFIGURED = env_set("DATABRICKS_SERVER_HOSTNAME", "DATABRICKS_HTTP_PATH", "DATABRICKS_ACCESS_TOKEN")


def refresh_warehouse_status() -> dict:
    """Query the warehouse state and stash it in session_state."""
    try:
        from agent.warehouse import get_warehouse_status
        status = get_warehouse_status()
    except Exception as e:
        status = {"state": "ERROR", "error": str(e), "checked_at": time.time()}
    st.session_state.wh_status = status
    return status


def warehouse_status() -> dict:
    """Cached-per-session status; checked once on first load."""
    if not DATABRICKS_CONFIGURED:
        return {"state": "UNCONFIGURED"}
    if "wh_status" not in st.session_state:
        refresh_warehouse_status()
    return st.session_state.wh_status


def wake_warehouse_ui(key: str):
    """Start the warehouse with live progress, then rerun with fresh status."""
    from agent.warehouse import wake_warehouse
    progress = st.empty()

    def on_update(status):
        note = status.get("note")
        progress.caption(f"⏳ {note}" if note else f"⏳ Warehouse state: **{status.get('state')}** — waiting…")

    try:
        with st.spinner("Waking the Databricks SQL Warehouse…"):
            status = wake_warehouse(on_update=on_update)
        st.session_state.wh_status = status
        st.session_state.wh_last_wake = time.time()
        if status.get("state") == "RUNNING":
            st.toast("🧱 Databricks warehouse is running", icon="✅")
        else:
            st.toast(f"Warehouse is {status.get('state')} — try again in a moment", icon="⚠️")
    except Exception as e:
        st.session_state.wh_status = {"state": "ERROR", "error": str(e), "checked_at": time.time()}
        st.toast(f"Wake-up failed: {e}", icon="❌")
    progress.empty()
    st.rerun()


def render_warehouse_panel(key: str, compact: bool = False):
    """Status pill + Check / Wake buttons. `key` must be unique per placement."""
    status = warehouse_status()
    state = status.get("state", "UNKNOWN")
    if state == "UNCONFIGURED":
        st.caption("🧱 Databricks credentials not set — warehouse status unavailable.")
        return
    from agent.warehouse import STATE_DISPLAY
    icon, label = STATE_DISPLAY.get(state, ("🔴", state.title() if state != "ERROR" else "Error"))
    color = WH_COLORS.get(state, "#f87171")

    meta = []
    if status.get("name"):
        meta.append(html.escape(status["name"]))
    if status.get("serverless"):
        meta.append("Serverless")
    if status.get("size"):
        meta.append(html.escape(str(status["size"])))
    if status.get("auto_stop_mins"):
        meta.append(f"auto-stop {status['auto_stop_mins']} min")
    if status.get("checked_at"):
        age = int(time.time() - status["checked_at"])
        meta.append("checked just now" if age < 5 else f"checked {age}s ago")
    if status.get("url"):
        meta.append(f'<a href="{html.escape(status["url"])}" target="_blank" style="color:#94a3b8">open ↗</a>')

    pill_html = (
        f'<div class="wh"><span class="pill" style="--c:{color}">{icon} {html.escape(label.upper())}</span>'
        + (f'<span class="meta">{" · ".join(meta)}</span>' if meta and not compact else "")
        + "</div>"
    )
    hint = None
    if state == "ERROR":
        hint = f"⚠️ {status.get('error', 'unknown error')}"
    elif state == "STOPPED":
        hint = "Asleep — the first Databricks query will wait for start-up (often 30–90 s). Wake it now to avoid the delay."
    elif state == "STARTING":
        hint = "Starting — Databricks queries will run once it reaches RUNNING."

    # Compact: pill + buttons on one row. Full: pill/meta on top, buttons below.
    if compact:
        c0, c1, c2 = st.columns([3.2, 1.1, 1.1], vertical_alignment="center")
        with c0:
            st.markdown(pill_html + (f'<div class="wh meta">{html.escape(hint)}</div>' if hint else ""),
                        unsafe_allow_html=True)
    else:
        st.markdown(pill_html, unsafe_allow_html=True)
        if hint:
            st.caption(hint)
        c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Check status", key=f"{key}_check", use_container_width=True, type="secondary"):
            refresh_warehouse_status()
            st.rerun()
    with c2:
        wake_label = "✅ Awake" if state == "RUNNING" else "⚡ Wake up"
        if st.button(wake_label, key=f"{key}_wake", use_container_width=True,
                     type="secondary" if state == "RUNNING" else "primary",
                     disabled=(state == "RUNNING")):
            wake_warehouse_ui(key)


# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
with st.sidebar:
    st.title("🏢 Prologis FinAssist")
    st.caption("AI-powered financial, property & lease insights for an industrial REIT — Databricks edition")

    st.divider()
    st.markdown("### 📂 Data Sources")
    st.markdown(f"""
- 🧱 **Databricks** — `lease_transactions` (Delta, SQL Warehouse)
- 🐘 **Postgres** — `properties` + `financials` (Supabase)
- 📑 **SEC EDGAR** — 10-K / 10-Q company facts
- 📰 **Press Releases** — JSON store
""")

    st.divider()
    st.markdown("### ☁️ Cloud Services")
    st.markdown(f"""
- 🤖 **Vertex AI** — `{GEMINI_MODEL}` agent (function calling)
- 🧱 **Databricks** — Serverless SQL Warehouse + Genie
- 🔮 **AWS SageMaker** — 2 hosted ML endpoints
- 📝 **AWS Bedrock** — Claude Haiku 4.5 summarization
- 🐘 **Supabase** — managed Postgres
""")

    st.divider()
    st.markdown("### 🔌 Connections")
    st.caption("Credentials present in this deployment")
    statuses = [
        ("Vertex AI", env_set("GOOGLE_API_KEY") or env_set("GEMINI_API_KEY")),
        ("Databricks", env_set("DATABRICKS_SERVER_HOSTNAME", "DATABRICKS_HTTP_PATH", "DATABRICKS_ACCESS_TOKEN")),
        ("Postgres", env_set("POSTGRES_HOST", "POSTGRES_PASSWORD")),
        ("AWS", env_set("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")),
        ("SageMaker reg", env_set("SAGEMAKER_REGRESSION_ENDPOINT")),
        ("SageMaker clf", env_set("SAGEMAKER_CLASSIFICATION_ENDPOINT")),
    ]
    st.markdown(
        '<div class="status">' +
        "".join(f'<span class="{"ok" if ok else "off"}">{name}</span>' for name, ok in statuses) +
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### 🧱 Databricks Warehouse")
    render_warehouse_panel(key="sb_wh")

    st.divider()
    st.markdown("### 🚀 ML Endpoints")
    reg_ep = os.getenv("SAGEMAKER_REGRESSION_ENDPOINT", "(not deployed)")
    clf_ep = os.getenv("SAGEMAKER_CLASSIFICATION_ENDPOINT", "(not deployed)")
    st.code(f"reg: {reg_ep}\nclf: {clf_ep}", language=None)

    st.divider()
    st.caption("Multi-cloud AI assignment · GCP + Databricks + AWS")

# --------------------------------------------------------------
# Main header
# --------------------------------------------------------------
st.title("🏢 Prologis Financial Assistant")
st.caption(
    "End-to-end multi-cloud AI system: Vertex AI Gemini agent routing across "
    "Databricks (leases), Postgres (properties), SEC EDGAR (filings) and press releases — "
    "with AWS Bedrock summarization and AWS SageMaker ML endpoints."
)

# --------------------------------------------------------------
# Tabs
# --------------------------------------------------------------
tab_chat, tab_data, tab_ml = st.tabs(["💬 Chat", "📊 Data", "🤖 ML Predictions"])

# ============================================================
# TAB 1: CHAT — newest-on-top, input pinned at top
# ============================================================
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0

    st.subheader("Conversational Assistant")
    st.caption(
        f"Ask about leases, properties, financials, or press releases. Vertex AI **{GEMINI_MODEL}** "
        "reads your question, picks the right tool(s) via function calling, and composes an answer "
        "grounded in the returned data."
    )

    # ---- Architecture & data-source overview ----
    with st.expander("🧭 How it works — architecture & data sources",
                     expanded=not st.session_state.messages):
        st.markdown(
            '<div class="flow">'
            '<span class="node">💬 your question</span><span class="arrow">→</span>'
            f'<span class="node brain">🤖 Vertex AI · {html.escape(GEMINI_MODEL)} · function calling</span>'
            '<span class="arrow">→</span><span class="node">🔧 up to 6 tool-calling turns</span>'
            '<span class="arrow">→</span><span class="node">🧱 🐘 📑 📰 📝 tools</span>'
            '<span class="arrow">→</span><span class="node">✍️ grounded answer</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        cards = []
        for tool, m in TOOL_META.items():
            cards.append(
                f'<div class="src-card" style="--c:{m["color"]}">'
                f'<div class="name">{m["icon"]} {html.escape(m["label"])}</div>'
                f'<div class="svc">{html.escape(m["service"])}</div>'
                f'<div class="tbl">{html.escape(tool)}<br/>{html.escape(m["table"])}</div>'
                f'<div class="rec">{html.escape(m["records"])}</div>'
                f'<div class="ask"><b>Ask about:</b> {html.escape(m["ask"])}<br/>'
                f'<b>Filters:</b> {html.escape(m["args"])}</div>'
                f'</div>'
            )
        st.markdown('<div class="src-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)

        genie_link = (f'<a href="https://{html.escape(DATABRICKS_HOST)}" target="_blank" '
                      f'style="color:#fca5a5">{html.escape(DATABRICKS_HOST)}</a>'
                      if DATABRICKS_HOST else "your Databricks workspace")
        st.markdown(
            f'<div class="genie">🧞 <b>Databricks Genie — “{GENIE_SPACE}”</b> · '
            f'The same <code>{DATABRICKS_TABLE}</code> table also has a Genie space in {genie_link}, '
            f'so anyone with workspace access can ask plain-English questions and get generated SQL, '
            f'a chart and a written summary — Databricks-native text-to-SQL alongside the custom '
            f'<code>query_databricks</code> tool used here. Verified Genie prompts: '
            f'<i>“What are the counts of leases by lease type?”</i>, '
            f'<i>“What is the distribution of lease terms in months?”</i>, '
            f'<i>“What is the monthly count of lease starts?”</i></div>',
            unsafe_allow_html=True,
        )

    # ---- Input form at the top — using a form so Enter submits ----
    with st.form(key=f"chat_form_{st.session_state.input_counter}", clear_on_submit=True):
        col_input, col_submit = st.columns([6, 1])
        with col_input:
            user_input = st.text_input(
                "Ask a question",
                placeholder="Ask about leases, properties, financials, or press releases…",
                label_visibility="collapsed",
            )
        with col_submit:
            submitted = st.form_submit_button("Ask 🚀", use_container_width=True, type="primary")

    if submitted and user_input.strip():
        run_query(user_input)

    # ---- Suggested queries — always visible, click to run ----
    st.markdown("##### 💡 Suggested queries — pick a data source, click a question to run it")
    sug_tabs = st.tabs(list(SUGGESTED_QUERIES.keys()))
    for tab, (group, queries) in zip(sug_tabs, SUGGESTED_QUERIES.items()):
        with tab:
            if group.startswith("🧱") or group.startswith("🔀"):
                # These hit the Databricks warehouse — surface its state right here.
                render_warehouse_panel(key=f"sug_wh_{group[:2]}", compact=True)
            sug_cols = st.columns(2)
            for i, q in enumerate(queries):
                with sug_cols[i % 2]:
                    if st.button(q, key=f"sug_{group}_{i}", use_container_width=True, type="secondary"):
                        run_query(q)

    # ---- Conversation (newest first) ----
    if st.session_state.messages:
        st.divider()
        hdr_col, clr_col = st.columns([5, 1])
        with hdr_col:
            st.markdown(f"##### 🗂️ Conversation · {len(st.session_state.messages)} exchange(s) — newest first")
        with clr_col:
            if st.button("🗑️ Clear chat", key="clear_chat", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.rerun()

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message("user"):
            st.markdown(safe_md(msg["user"]))
        with st.chat_message("assistant"):
            calls = msg.get("tool_calls") or []
            if calls:
                st.markdown('<div class="chips">' + "".join(source_chip(c["name"]) for c in calls) + "</div>",
                            unsafe_allow_html=True)
            st.markdown(safe_md(msg["assistant"]))
            if calls:
                with st.expander(f"🔧 {len(calls)} tool call(s) — what the agent fetched"):
                    for c in calls:
                        m = TOOL_META.get(c["name"], {"icon": "🔧", "label": c["name"]})
                        st.markdown(f"**{m['icon']} {m['label']}** — `{c['name']}({c['args']})`")
                        render_tool_result(c["result"])
                        st.markdown("")
            if msg.get("traceback"):
                with st.expander("Traceback"):
                    st.code(msg["traceback"])
        if idx < len(st.session_state.messages) - 1:
            st.markdown("<hr style='border-color: rgba(148,163,184,0.08); margin: 0.5rem 0;'/>",
                        unsafe_allow_html=True)

# ============================================================
# TAB 2: DATA BROWSER
# ============================================================
with tab_data:
    st.subheader("Data Browser")
    st.caption("Browse every source the agent can query — the same tables and files the tools read.")
    sub_pg, sub_dbx, sub_sec, sub_pr = st.tabs([
        "🐘 Properties (Postgres)", "🧱 Lease Transactions (Databricks)",
        "📑 SEC Filings", "📰 Press Releases",
    ])

    with sub_pg:
        st.markdown("#### Properties & Financials (Postgres on Supabase)")
        st.caption("`properties ⋈ financials` — 20 synthetic properties across 11 US metros. Queried by `query_postgres`.")
        try:
            with st.spinner("Querying Postgres…"):
                df = load_properties()
            col1, col2 = st.columns(2)
            with col1:
                metro_filter = st.multiselect(
                    "Filter by metro",
                    options=sorted(df["metro_area"].unique()),
                    key="pg_metro",
                )
            with col2:
                type_filter = st.multiselect(
                    "Filter by type",
                    options=sorted(df["property_type"].unique()),
                    key="pg_type",
                )
            view = df.copy()
            if metro_filter:
                view = view[view["metro_area"].isin(metro_filter)]
            if type_filter:
                view = view[view["property_type"].isin(type_filter)]
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Properties", len(view))
            mcol2.metric("Total revenue", f"${view['revenue'].sum()/1e6:.1f}M")
            mcol3.metric("Total net income", f"${view['net_income'].sum()/1e6:.1f}M")
            mcol4.metric("Avg revenue", f"${view['revenue'].mean()/1e6:.1f}M" if len(view) else "—")
            st.dataframe(view, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"DB connection failed: {e}")

    with sub_dbx:
        st.markdown("#### Lease Transactions (Databricks SQL Warehouse)")
        st.caption(f"`{DATABRICKS_TABLE}` — Delta table on a Serverless SQL Warehouse. Queried by `query_databricks` "
                   f"and by the **{GENIE_SPACE}** Genie space.")
        render_warehouse_panel(key="data_wh")
        wh_state = warehouse_status().get("state")
        # A cold warehouse makes the first query block for a minute or more, so
        # don't auto-load the table until it is running (or the user insists).
        warehouse_cold = wh_state in ("STOPPED", "STARTING", "STOPPING")
        try:
            if warehouse_cold and not st.session_state.get("dbx_force_load"):
                st.info("The warehouse is not running, so the lease table isn't loaded automatically. "
                        "Wake it up above, or load anyway and wait for start-up.")
                if st.button("Load anyway (waits for the warehouse)", key="dbx_force_load_btn", type="secondary"):
                    st.session_state.dbx_force_load = True
                    st.rerun()
                leases = None
            else:
                with st.spinner("Querying Databricks SQL Warehouse…"):
                    leases = load_lease_transactions()
            if leases is None:
                pass  # skipped — warehouse cold
            elif leases.empty:
                st.warning("The lease_transactions table returned no rows.")
            else:
                f1, f2, f3 = st.columns(3)
                with f1:
                    dbx_metro = st.multiselect("Filter by metro", sorted(leases["metro_area"].unique()), key="dbx_metro")
                with f2:
                    dbx_ind = st.multiselect("Filter by tenant industry", sorted(leases["tenant_industry"].unique()), key="dbx_ind")
                with f3:
                    dbx_type = st.multiselect("Filter by lease type", sorted(leases["lease_type"].unique()), key="dbx_type")
                lv = leases.copy()
                if dbx_metro:
                    lv = lv[lv["metro_area"].isin(dbx_metro)]
                if dbx_ind:
                    lv = lv[lv["tenant_industry"].isin(dbx_ind)]
                if dbx_type:
                    lv = lv[lv["lease_type"].isin(dbx_type)]
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Leases", len(lv))
                k2.metric("Total annual rent", f"${lv['annual_rent'].sum()/1e6:.1f}M")
                k3.metric("Avg rent / sq ft", f"${lv['rent_per_sqft'].mean():.2f}" if len(lv) else "—")
                k4.metric("Total sq ft", f"{lv['sq_footage'].sum()/1e6:.2f}M")
                st.dataframe(
                    lv.sort_values("annual_rent", ascending=False),
                    use_container_width=True, hide_index=True,
                )
                if st.button("↻ Refresh from Databricks", key="dbx_refresh", type="secondary"):
                    load_lease_transactions.clear()
                    refresh_warehouse_status()
                    st.rerun()
        except Exception as e:
            st.error(f"Databricks query failed: {e}")
            st.caption("If the warehouse has been idle, it may need a moment (or a manual restart) before queries succeed.")

    with sub_sec:
        st.markdown("#### SEC EDGAR — Prologis (NYSE: PLD)")
        st.caption("Real figures from the SEC XBRL Company Facts API, cached in `data/sec/`. Queried by `query_sec_edgar`.")
        data = load_json(str(SEC_PATH))
        if data:
            rows = []
            for name, m in data.get("metrics", {}).items():
                latest = m.get("latest_annual") or {}
                quarterly = m.get("latest_quarterly") or {}
                rows.append({
                    "Metric": name,
                    "Latest annual (USD)": f"${latest.get('val', 0):,}",
                    "FY end": latest.get("end", "—"),
                    "Form": latest.get("form", "—"),
                    "Latest quarterly (USD)": f"${quarterly.get('val', 0):,}" if quarterly else "—",
                    "Quarter end": quarterly.get("end", "—") if quarterly else "—",
                })
            st.table(rows)
            with st.expander("View raw JSON"):
                st.json(data, expanded=False)
        else:
            st.warning("Run `python scripts/fetch_sec.py` to populate SEC data.")

    with sub_pr:
        st.markdown("#### Recent Press Releases")
        st.caption("Mocked releases in `data/press_releases.json`. Queried by `query_press_releases`; "
                   "summaries go through AWS Bedrock (Claude Haiku 4.5).")
        releases = load_json(str(PRESS_PATH))
        if releases:
            categories = sorted(set(r["category"] for r in releases))
            cat_filter = st.multiselect("Filter by category", options=categories, key="pr_cat")
            filtered = [r for r in releases if not cat_filter or r["category"] in cat_filter]
            st.caption(f"Showing {len(filtered)} of {len(releases)} releases")
            for pr in filtered:
                with st.expander(f"📰 {pr['date']} — {pr['title']}"):
                    st.markdown(f"**Category:** `{pr['category']}`")
                    st.markdown(safe_md(pr["content"]))
        else:
            st.warning("Press releases file not found.")


# ============================================================
# TAB 3: ML PREDICTIONS
# ============================================================
with tab_ml:
    st.subheader("ML Model Predictions")
    st.caption("Both models are deployed as live SageMaker endpoints and called over HTTPS.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 🏠 Housing Price")
        st.caption("**Random Forest** on California Housing — predicts median house value")

        med_inc = st.slider("Median Income (10k USD)", 0.5, 15.0, 5.0)
        house_age = st.slider("House Age (years)", 1, 52, 25)
        avg_rooms = st.slider("Avg Rooms", 1.0, 10.0, 5.0)
        avg_bedrms = st.slider("Avg Bedrooms", 0.5, 3.0, 1.0)
        population = st.slider("Population", 100, 5000, 1500)
        avg_occup = st.slider("Avg Occupancy", 1.0, 6.0, 3.0)
        latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
        longitude = st.slider("Longitude", -125.0, -114.0, -118.0)

        if st.button("🎯 Predict House Value", key="reg_btn", use_container_width=True, type="primary"):
            payload = {
                "MedInc": med_inc, "HouseAge": house_age,
                "AveRooms": avg_rooms, "AveBedrms": avg_bedrms,
                "Population": population, "AveOccup": avg_occup,
                "Latitude": latitude, "Longitude": longitude,
            }
            ep = os.getenv("SAGEMAKER_REGRESSION_ENDPOINT")
            if not ep:
                st.error("SAGEMAKER_REGRESSION_ENDPOINT not set")
            else:
                try:
                    with st.spinner("Calling SageMaker..."):
                        result = invoke_sagemaker(ep, payload)
                    val = result[0]["predicted_value_usd"]
                    st.success(f"### 💰 ${val:,.0f}")
                    st.caption("Predicted median house value")
                    with st.expander("Raw response"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Endpoint call failed: {e}")

    with col2:
        st.markdown("### 🏦 Subscription Likelihood")
        st.caption("**Logistic Regression** on UCI Bank Marketing — predicts subscription")

        age = st.number_input("Age", 18, 95, 35)
        job = st.selectbox("Job", [
            "admin.", "blue-collar", "technician", "services", "management",
            "retired", "self-employed", "entrepreneur", "unemployed",
            "housemaid", "student", "unknown"
        ])
        marital = st.selectbox("Marital", ["married", "single", "divorced"])
        education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
        default = st.selectbox("Has credit in default?", ["no", "yes"])
        housing = st.selectbox("Has housing loan?", ["no", "yes"])
        loan = st.selectbox("Has personal loan?", ["no", "yes"])
        contact = st.selectbox("Contact type", ["cellular", "telephone", "unknown"])
        month = st.selectbox("Last contact month", [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ])
        poutcome = st.selectbox("Previous outcome", ["unknown", "failure", "other", "success"])
        balance = st.number_input("Balance (EUR)", -5000, 100000, 1500)
        duration = st.number_input("Last Contact Duration (s)", 0, 5000, 200)
        campaign = st.number_input("# contacts this campaign", 1, 50, 1)
        pdays = st.number_input("Days since last contact (-1 = never)", -1, 1000, -1)
        previous = st.number_input("# contacts before this campaign", 0, 50, 0)

        if st.button("🎯 Predict Subscription", key="clf_btn", use_container_width=True, type="primary"):
            payload = {
                "age": age, "job": job, "marital": marital, "education": education,
                "default": default, "housing": housing, "loan": loan,
                "contact": contact, "month": month, "poutcome": poutcome,
                "balance": balance, "duration": duration,
                "campaign": campaign, "pdays": pdays, "previous": previous,
            }
            ep = os.getenv("SAGEMAKER_CLASSIFICATION_ENDPOINT")
            if not ep:
                st.error("SAGEMAKER_CLASSIFICATION_ENDPOINT not set")
            else:
                try:
                    with st.spinner("Calling SageMaker..."):
                        result = invoke_sagemaker(ep, payload)
                    pred = result[0]
                    label = pred["label"]
                    prob = pred["probability"]
                    if label == "yes":
                        st.success(f"### ✅ Will subscribe")
                        st.caption(f"Confidence: {prob:.1%}")
                    else:
                        st.warning(f"### ❌ Will NOT subscribe")
                        st.caption(f"Probability of yes: {prob:.1%}")
                    with st.expander("Raw response"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Endpoint call failed: {e}")
