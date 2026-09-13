"""
Financial Assistant Web Application - Streamlit Frontend
Tabs: Chat | Data Browser | ML Predictions

Powered by Vertex AI (Gemini) function calling, AWS SageMaker, AWS Bedrock,
and Postgres on Supabase.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

st.set_page_config(
    page_title="Prologis Financial Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
            radial-gradient(circle at 80% 100%, rgba(34, 211, 238, 0.06) 0%, transparent 50%),
            #0a0e1a;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

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
    [data-testid="stSidebar"] li { color: #e2e8f0; font-size: 0.9rem; }
    [data-testid="stSidebar"] code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #67e8f9;
        background: rgba(34, 211, 238, 0.08);
        border: 1px solid rgba(34, 211, 238, 0.15);
        border-radius: 6px;
    }

    /* Main heading gradient */
    .main h1, .main h2 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em; }
    .main h1 {
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

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }

    /* Buttons */
    .stButton button {
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
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 28px rgba(99, 102, 241, 0.4);
        background-position: 100% 0%;
    }

    /* Form: text input on top */
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

    /* Newest message highlight */
    .latest-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(34, 211, 238, 0.08) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 0 24px rgba(99, 102, 241, 0.1);
    }
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


# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
with st.sidebar:
    st.title("🏢 Prologis FinAssist")
    st.caption("AI-powered financial & property insights for an industrial REIT")
    st.divider()
    st.markdown("### 📂 Data Sources")
    st.markdown("""
- **SEC EDGAR** — 10-K / 10-Q filings
- **Postgres** — properties + financials
- **Press Releases** — JSON store
""")
    st.divider()
    st.markdown("### ☁️ Cloud Services")
    st.markdown("""
- 🤖 **Vertex AI** — Gemini 2.5 Flash agent (function calling)
- 🔮 **AWS SageMaker** — ML model endpoints
- 📝 **AWS Bedrock** — Claude Haiku summarization
""")
    st.divider()
    st.markdown("### 🚀 ML Endpoints")
    reg_ep = os.getenv("SAGEMAKER_REGRESSION_ENDPOINT", "(not deployed)")
    clf_ep = os.getenv("SAGEMAKER_CLASSIFICATION_ENDPOINT", "(not deployed)")
    st.code(f"reg: {reg_ep}\nclf: {clf_ep}", language=None)
    st.divider()
    st.caption("Multi-cloud AI assignment · GCP + AWS")

# --------------------------------------------------------------
# Main header
# --------------------------------------------------------------
st.title("🏢 Prologis Financial Assistant")
st.caption("End-to-end AI system: structured data + classic ML + generative AI on Postgres, AWS SageMaker, AWS Bedrock, and Google Vertex AI.")

# --------------------------------------------------------------
# Tabs
# --------------------------------------------------------------
tab_chat, tab_data, tab_ml = st.tabs(["💬 Chat", "📊 Data", "🤖 ML Predictions"])

# ============================================================
# TAB 1: CHAT — newest-on-top, input pinned at top
# ============================================================
with tab_chat:
    st.subheader("Conversational Assistant")
    st.caption("Ask about financials, properties, or recent press releases. Powered by Vertex AI Gemini function calling — automatically routes your question across Postgres, SEC EDGAR, press releases, and AWS Bedrock.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0

    # Input form at the top — using a form so Enter submits
    with st.form(key=f"chat_form_{st.session_state.input_counter}", clear_on_submit=True):
        col_input, col_submit = st.columns([6, 1])
        with col_input:
            user_input = st.text_input(
                "Ask a question",
                placeholder="Ask a question about Prologis...",
                label_visibility="collapsed",
            )
        with col_submit:
            submitted = st.form_submit_button("Ask 🚀", use_container_width=True)

    if st.session_state.messages:
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    # Process new message
    if submitted and user_input.strip():
        prompt = user_input.strip()
        try:
            from agent.agent import run_agent
            with st.spinner("Thinking..."):
                result = run_agent(prompt)
            # Insert at the FRONT so newest renders on top
            st.session_state.messages.insert(0, {
                "user": prompt,
                "assistant": result["answer"],
                "tool_calls": result["tool_calls"],
            })
        except Exception as e:
            import traceback
            st.session_state.messages.insert(0, {
                "user": prompt,
                "assistant": f"⚠️ Agent error: {e}",
                "tool_calls": [],
                "traceback": traceback.format_exc(),
            })
        st.session_state.input_counter += 1
        st.rerun()

    # Example queries (only when no messages yet)
    if not st.session_state.messages:
        with st.expander("💡 Example queries", expanded=True):
            cols = st.columns(2)
            examples = [
                "What was Prologis' net income last year?",
                "Show industrial properties in Chicago with revenue.",
                "Did Prologis announce any acquisitions recently?",
                "Summarize the most recent earnings press release.",
                "Compare property revenues between Dallas and Phoenix.",
                "Which metro has the highest average revenue per property?",
            ]
            for i, ex in enumerate(examples):
                with cols[i % 2]:
                    st.markdown(f"- _{ex}_")

    # Render messages — newest first (already inserted at front)
    for idx, msg in enumerate(st.session_state.messages):
        # Highlight only the most recent (idx == 0)
        is_latest = (idx == 0)
        wrapper_class = "latest-card" if is_latest else ""
        if is_latest:
            st.markdown('<div class="latest-card">', unsafe_allow_html=True)

        with st.chat_message("user"):
            st.markdown(safe_md(msg["user"]))
        with st.chat_message("assistant"):
            st.markdown(safe_md(msg["assistant"]))
            if msg.get("tool_calls"):
                with st.expander(f"🔧 {len(msg['tool_calls'])} tool call(s)"):
                    for c in msg["tool_calls"]:
                        st.markdown(f"**`{c['name']}`**(`{c['args']}`)")
                        st.json(c["result"], expanded=False)
            if msg.get("traceback"):
                with st.expander("Traceback"):
                    st.code(msg["traceback"])

        if is_latest:
            st.markdown('</div>', unsafe_allow_html=True)

        if not is_latest:
            st.markdown("<hr style='border-color: rgba(148,163,184,0.08); margin: 0.5rem 0;'/>", unsafe_allow_html=True)

# ============================================================
# TAB 2: DATA BROWSER
# ============================================================
with tab_data:
    st.subheader("Data Browser")
    sub1, sub2, sub3 = st.tabs(["🏢 Properties", "📑 SEC Filings", "📰 Press Releases"])

    with sub1:
        st.markdown("#### Properties & Financials (Postgres)")
        try:
            engine = get_db_engine()
            sql = """
                SELECT p.property_id, p.address, p.metro_area, p.sq_footage,
                       p.property_type, f.revenue, f.net_income, f.expenses
                FROM properties p
                LEFT JOIN financials f ON p.property_id = f.property_id
                ORDER BY p.property_id
            """
            df = pd.read_sql(sql, engine)
            col1, col2 = st.columns(2)
            with col1:
                metro_filter = st.multiselect(
                    "Filter by metro",
                    options=sorted(df["metro_area"].unique()),
                )
            with col2:
                type_filter = st.multiselect(
                    "Filter by type",
                    options=sorted(df["property_type"].unique()),
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

    with sub2:
        st.markdown("#### SEC EDGAR — Prologis (NYSE: PLD)")
        sec_path = Path(__file__).parent.parent / "data" / "sec" / "prologis_financials.json"
        if sec_path.exists():
            data = json.loads(sec_path.read_text())
            rows = []
            for name, m in data.get("metrics", {}).items():
                latest = m.get("latest_annual") or {}
                rows.append({
                    "Metric": name,
                    "Value (USD)": f"${latest.get('val', 0):,}",
                    "FY End": latest.get("end", "—"),
                    "Form": latest.get("form", "—"),
                })
            st.table(rows)
            with st.expander("View raw JSON"):
                st.json(data, expanded=False)
        else:
            st.warning("Run `python scripts/fetch_sec.py` to populate SEC data.")

    with sub3:
        st.markdown("#### Recent Press Releases")
        pr_path = Path(__file__).parent.parent / "data" / "press_releases.json"
        if pr_path.exists():
            releases = json.loads(pr_path.read_text())
            categories = sorted(set(r["category"] for r in releases))
            cat_filter = st.multiselect("Filter by category", options=categories)
            filtered = [r for r in releases if not cat_filter or r["category"] in cat_filter]
            st.caption(f"Showing {len(filtered)} of {len(releases)} releases")
            for pr in filtered:
                with st.expander(f"📰 {pr['date']} — {pr['title']}"):
                    st.markdown(f"**Category:** `{pr['category']}`")
                    st.markdown(pr["content"])


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

        if st.button("🎯 Predict House Value", key="reg_btn", use_container_width=True):
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

        if st.button("🎯 Predict Subscription", key="clf_btn", use_container_width=True):
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
