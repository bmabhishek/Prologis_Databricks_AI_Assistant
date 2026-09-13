"""
AWS Bedrock summarization. Uses Claude Haiku 4.5 to produce short
summaries — the multi-cloud component of the assignment.
"""
import json
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-1")
# US inference profile for Claude Haiku 4.5
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def summarize_with_bedrock(text: str, max_words: int = 50) -> str:
    """Use Claude Haiku via Bedrock to summarize text in <= max_words words.
    Returns the summary string. Falls back to truncated text on error."""
    if not text:
        return ""
    client = boto3.client("bedrock-runtime", region_name=REGION)
    prompt = (
        f"Summarize the following text in at most {max_words} words. "
        f"Be specific and concrete. Return ONLY the summary, no preamble.\n\n"
        f"Text:\n{text}"
    )
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"].strip()
    except Exception as e:
        # Fail soft — return truncated original instead of crashing
        return text[:300] + ("..." if len(text) > 300 else "") + f" [Bedrock error: {e}]"


if __name__ == "__main__":
    sample = (
        "Prologis announced the acquisition of a 3.2 million square foot Class-A "
        "logistics portfolio in the Dallas-Fort Worth metro area for approximately "
        "$620 million. The portfolio consists of 8 buildings, 96% leased to "
        "investment-grade tenants, and expands Prologis' DFW footprint by approximately 12%."
    )
    print("Original:", sample)
    print("\nSummary:", summarize_with_bedrock(sample, max_words=25))
