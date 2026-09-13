"""
Fetch SEC EDGAR financial data for Prologis (CIK 0001045609).
Uses SEC's free company facts API — no auth needed, just a User-Agent.

Outputs: data/sec/prologis_financials.json
"""
import json
import os
import requests
from pathlib import Path

# Prologis CIK (Central Index Key) — padded to 10 digits for the API
PROLOGIS_CIK = "0001045609"

HEADERS = {
    # SEC requires a User-Agent with contact info — use yours
    "User-Agent": "Rahul Nayak rahulnayak70400@gmail.com"
}

OUT_DIR = Path(__file__).parent.parent / "data" / "sec"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Key metrics we want from the company facts API
# These are GAAP concept names used in SEC filings
TARGET_CONCEPTS = {
    "Revenues": "revenue",
    "NetIncomeLoss": "net_income",
    "OperatingExpenses": "operating_expenses",
    "Assets": "total_assets",
    "Liabilities": "total_liabilities",
}


def fetch_company_facts(cik: str) -> dict:
    """Fetch all reported facts for a company from SEC EDGAR."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    print(f"Fetching {url}")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_metrics(facts: dict) -> dict:
    """Pull out the latest annual + quarterly values for our target concepts."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    out = {"company": facts.get("entityName", "Prologis"), "metrics": {}}

    for concept, friendly in TARGET_CONCEPTS.items():
        if concept not in us_gaap:
            continue
        # Each concept has values reported in different units (USD, shares, etc.)
        units = us_gaap[concept].get("units", {})
        usd_values = units.get("USD", [])
        if not usd_values:
            continue

        # Filter to 10-K (annual) and 10-Q (quarterly)
        annual = [v for v in usd_values if v.get("form") == "10-K"]
        quarterly = [v for v in usd_values if v.get("form") == "10-Q"]

        # Sort by end date descending, take most recent
        annual.sort(key=lambda v: v.get("end", ""), reverse=True)
        quarterly.sort(key=lambda v: v.get("end", ""), reverse=True)

        out["metrics"][friendly] = {
            "latest_annual": annual[0] if annual else None,
            "latest_quarterly": quarterly[0] if quarterly else None,
            "annual_history": annual[:5],   # last 5 years
            "quarterly_history": quarterly[:8],  # last 8 quarters
        }
    return out


def main():
    facts = fetch_company_facts(PROLOGIS_CIK)
    metrics = extract_metrics(facts)

    out_path = OUT_DIR / "prologis_financials.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    # Print a quick summary
    print("\n--- Summary ---")
    print(f"Company: {metrics['company']}")
    for name, data in metrics["metrics"].items():
        latest = data.get("latest_annual")
        if latest:
            val = latest.get("val", 0)
            end = latest.get("end", "?")
            print(f"  {name:20s}: ${val:>18,} (FY ending {end})")


if __name__ == "__main__":
    main()
