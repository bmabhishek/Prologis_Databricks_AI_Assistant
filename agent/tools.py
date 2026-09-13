"""
Tools available to the Gemini agent. Each function queries one data source
(Postgres, SEC EDGAR JSON, or press releases JSON) and returns structured data
that the LLM can summarize.
"""
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
SEC_PATH = ROOT / "data" / "sec" / "prologis_financials.json"
PRESS_PATH = ROOT / "data" / "press_releases.json"


# --------------------------------------------------------------
# DB connection (cached at module level)
# --------------------------------------------------------------
def _get_engine():
    user = os.getenv("POSTGRES_USER", "postgres")
    pw = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "financial_assistant")
    return create_engine(f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}")


# ============================================================
# TOOL 1: Postgres property query
# ============================================================
def query_postgres(
    metro_area: Optional[str] = None,
    property_type: Optional[str] = None,
    min_revenue: Optional[float] = None,
    limit: int = 20,
) -> dict:
    """
    Query the properties+financials database. Filter by metro, type, or
    minimum revenue. Returns up to `limit` properties with their financials.

    Args:
        metro_area: city name, e.g. "Chicago", "Dallas", "Los Angeles"
        property_type: "Industrial", "Logistics", or "Warehouse"
        min_revenue: only return properties with revenue >= this value (USD)
        limit: max rows to return

    Returns:
        {"count": int, "properties": [...], "summary": {...}}
    """
    engine = _get_engine()
    sql = """
        SELECT p.property_id, p.address, p.metro_area, p.sq_footage,
               p.property_type, f.revenue, f.net_income, f.expenses
        FROM properties p
        LEFT JOIN financials f ON p.property_id = f.property_id
        WHERE 1=1
    """
    params = {}
    if metro_area:
        sql += " AND LOWER(p.metro_area) = LOWER(:metro)"
        params["metro"] = metro_area
    if property_type:
        sql += " AND LOWER(p.property_type) = LOWER(:ptype)"
        params["ptype"] = property_type
    if min_revenue is not None:
        sql += " AND f.revenue >= :minrev"
        params["minrev"] = min_revenue
    sql += " ORDER BY f.revenue DESC NULLS LAST LIMIT :lim"
    params["lim"] = limit

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    if df.empty:
        return {"count": 0, "properties": [], "summary": {}}

    # Cast numerics for JSON serialization
    for col in ["revenue", "net_income", "expenses"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return {
        "count": len(df),
        "properties": df.to_dict("records"),
        "summary": {
            "total_revenue": float(df["revenue"].sum()),
            "total_net_income": float(df["net_income"].sum()),
            "avg_revenue": float(df["revenue"].mean()),
            "metros": sorted(df["metro_area"].unique().tolist()),
            "types": sorted(df["property_type"].unique().tolist()),
        },
    }


# ============================================================
# TOOL 2: SEC EDGAR financial metrics
# ============================================================
def query_sec_edgar(metric: Optional[str] = None, period: str = "annual") -> dict:
    """
    Look up Prologis financial metrics from SEC EDGAR data.

    Args:
        metric: one of "revenue", "net_income", "operating_expenses",
                "total_assets", "total_liabilities". If None, returns all.
        period: "annual" (10-K) or "quarterly" (10-Q)

    Returns:
        {"company": str, "results": [...]}
    """
    if not SEC_PATH.exists():
        return {"error": "SEC data not found. Run scripts/fetch_sec.py first."}

    data = json.loads(SEC_PATH.read_text())
    metrics = data.get("metrics", {})

    if metric and metric not in metrics:
        return {
            "error": f"Unknown metric '{metric}'.",
            "available_metrics": list(metrics.keys()),
        }

    target_metrics = [metric] if metric else list(metrics.keys())
    out = {"company": data.get("company", "Prologis"), "results": []}

    for m in target_metrics:
        m_data = metrics.get(m, {})
        if period == "annual":
            entry = m_data.get("latest_annual")
        else:
            entry = m_data.get("latest_quarterly")
        if entry:
            out["results"].append({
                "metric": m,
                "value_usd": entry.get("val"),
                "period_end": entry.get("end"),
                "form": entry.get("form"),
                "fiscal_year": entry.get("fy"),
                "fiscal_period": entry.get("fp"),
            })
    return out


# ============================================================
# TOOL 3: Press releases search
# ============================================================
def query_press_releases(
    keywords: Optional[list] = None,
    category: Optional[str] = None,
    limit: int = 5,
) -> dict:
    """
    Search Prologis press releases by keywords (case-insensitive substring
    match across title and content) and/or category.

    Args:
        keywords: list of keywords like ["acquisition", "Dallas"]
        category: one of "earnings", "acquisition", "expansion", "sustainability"
        limit: max releases to return

    Returns:
        {"count": int, "releases": [{"date", "title", "category", "content"}]}
    """
    limit = int(limit)
    if not PRESS_PATH.exists():
        return {"error": "Press releases file not found."}

    releases = json.loads(PRESS_PATH.read_text())

    def match(r: dict) -> bool:
        if category and r.get("category", "").lower() != category.lower():
            return False
        if keywords:
            text_blob = (r.get("title", "") + " " + r.get("content", "")).lower()
            for kw in keywords:
                if kw.lower() not in text_blob:
                    return False
        return True

    matched = [r for r in releases if match(r)]
    matched.sort(key=lambda r: r.get("date", ""), reverse=True)
    return {"count": len(matched), "releases": matched[:limit]}


# Quick test when run directly
if __name__ == "__main__":
    print("=== query_postgres(metro_area='Chicago') ===")
    print(json.dumps(query_postgres(metro_area="Chicago"), indent=2, default=str))
    print("\n=== query_sec_edgar(metric='revenue') ===")
    print(json.dumps(query_sec_edgar(metric="revenue"), indent=2, default=str))
    print("\n=== query_press_releases(category='acquisition') ===")
    print(json.dumps(query_press_releases(category="acquisition"), indent=2, default=str))
