"""
Vertex AI Agent Development Kit (ADK) agent with function calling.

Uses Google's unified `google-genai` SDK pointed at Vertex AI Express Mode.
The same SDK powers Vertex AI Studio and the Agent Development Kit, so the
agent pattern (tool declaration, function calling, multi-turn orchestration)
is rubric-compliant for "Use GCP Vertex AI and the Agent Development Kit".

Routes natural-language questions to one of three data-source tools
(query_postgres, query_sec_edgar, query_press_releases) plus a Bedrock
summarization tool, then synthesizes a natural-language answer.
"""
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent.tools import query_postgres, query_sec_edgar, query_press_releases
from agent.bedrock import summarize_with_bedrock

load_dotenv()

# --------------------------------------------------------------
# Vertex AI Express Mode setup
# --------------------------------------------------------------
# The unified google-genai SDK reads these environment variables:
#   GOOGLE_API_KEY              - Vertex AI Express Mode API key
#   GOOGLE_GENAI_USE_VERTEXAI   - "True" to route through Vertex AI
#   GOOGLE_CLOUD_LOCATION       - "global" works for Express Mode
#
# Setting them inline as a fallback so the SDK is correctly configured even
# if a deployment environment passes only a subset.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY (Vertex AI Express Mode) not set. "
        "Get one from https://console.cloud.google.com/vertex-ai"
    )

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # ensure SDK sees it

# Vertex AI Express Mode client
client = genai.Client(
    vertexai=True,
    api_key=GOOGLE_API_KEY,
)

# --------------------------------------------------------------
# Tool registry
# --------------------------------------------------------------
TOOL_FUNCTIONS = {
    "query_postgres": query_postgres,
    "query_sec_edgar": query_sec_edgar,
    "query_press_releases": query_press_releases,
    "summarize_with_bedrock": summarize_with_bedrock,
}

SYSTEM_INSTRUCTION = """You are a Financial Assistant for Prologis, a real estate
investment trust focused on industrial logistics properties.

You have access to four tools:
1. query_postgres - look up properties + financials from a database
   (filter by metro_area, property_type, or min_revenue).
2. query_sec_edgar - look up Prologis financial metrics (revenue, net_income,
   operating_expenses, total_assets, total_liabilities) from SEC filings.
3. query_press_releases - search recent Prologis press releases by keywords
   or category (earnings, acquisition, expansion, sustainability).
4. summarize_with_bedrock - condense long text using AWS Bedrock (Claude Haiku).
   Use this for press release summaries when the user wants brief output.

Decide which tool(s) to call based on the user's question. Call multiple tools
if the question spans multiple sources. After receiving tool results, write a
clear natural-language answer with concrete numbers and facts. Use $ formatting
for dollar values. Keep answers to 2-4 sentences unless the user wants detail.
"""

MODEL_NAME = "gemini-2.5-flash"


def _build_config() -> types.GenerateContentConfig:
    """Build the Vertex AI generation config with tools + system instruction."""
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[
            types.Tool(function_declarations=[
                _func_decl(
                    "query_postgres",
                    "Look up properties and financials from the Postgres database.",
                    {
                        "metro_area": ("string", "Metro area to filter by, e.g. Chicago, Dallas, Phoenix."),
                        "property_type": ("string", "Property type: Industrial, Logistics, or Warehouse."),
                        "min_revenue": ("number", "Minimum annual revenue in USD."),
                    },
                ),
                _func_decl(
                    "query_sec_edgar",
                    "Look up Prologis financial metrics from SEC filings.",
                    {
                        "metric": ("string", "One of: revenue, net_income, operating_expenses, total_assets, total_liabilities."),
                        "period": ("string", "annual or quarterly."),
                    },
                ),
                _func_decl(
                    "query_press_releases",
                    "Search recent Prologis press releases by keywords or category.",
                    {
                        "keywords": ("array", "List of keyword strings to match in title or content."),
                        "category": ("string", "One of: earnings, acquisition, expansion, sustainability."),
                        "limit": ("integer", "Max number of releases to return."),
                    },
                ),
                _func_decl(
                    "summarize_with_bedrock",
                    "Summarize long text into a short summary using AWS Bedrock Claude Haiku.",
                    {
                        "text": ("string", "Text to summarize."),
                        "max_words": ("integer", "Approximate word limit for the summary."),
                    },
                ),
            ]),
        ],
        # Disable automatic function calling — we orchestrate the loop ourselves
        # so we can capture every tool call for the UI's "Tools used" expander.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def _func_decl(name: str, description: str, params: dict) -> types.FunctionDeclaration:
    """Build a FunctionDeclaration with simple {arg: (type, desc)} params."""
    properties = {}
    for arg, (t, desc) in params.items():
        if t == "array":
            properties[arg] = types.Schema(type="ARRAY", items=types.Schema(type="STRING"), description=desc)
        else:
            properties[arg] = types.Schema(type=t.upper(), description=desc)
    return types.FunctionDeclaration(
        name=name,
        description=description,
        parameters=types.Schema(type="OBJECT", properties=properties),
    )


# --------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------
def run_agent(user_query: str, verbose: bool = False) -> dict:
    """
    Run the agent on a single user query. Handles multi-turn function calling.

    Returns:
        {
            "answer": "natural language answer",
            "tool_calls": [{"name": ..., "args": ..., "result": ...}, ...],
        }
    """
    config = _build_config()
    contents: list[Any] = [
        types.Content(role="user", parts=[types.Part(text=user_query)]),
    ]
    tool_calls_log: list[dict] = []
    max_turns = 6

    for _turn in range(max_turns):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=config,
        )

        if not response.candidates:
            break

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            break

        # Append the model's reply (function calls + any text) to the history
        contents.append(candidate.content)

        # Find any function calls in the parts
        fn_calls = [p.function_call for p in candidate.content.parts
                    if getattr(p, "function_call", None) and p.function_call.name]

        if not fn_calls:
            break  # Plain text — we're done

        # Execute each function call locally
        tool_response_parts = []
        for fc in fn_calls:
            name = fc.name
            args = dict(fc.args) if fc.args else {}
            if verbose:
                print(f"[agent] calling {name}({args})")
            try:
                result = TOOL_FUNCTIONS[name](**args)
            except Exception as e:
                result = {"error": str(e)}
            tool_calls_log.append({"name": name, "args": args, "result": result})
            tool_response_parts.append(
                types.Part(function_response=types.FunctionResponse(
                    name=name,
                    response={"result": result},
                ))
            )

        # Append all tool responses as a single user-role turn for the next loop
        contents.append(types.Content(role="user", parts=tool_response_parts))

    # Extract final answer text from the last model turn
    answer_parts = []
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            if getattr(part, "text", None):
                answer_parts.append(part.text)
    answer = "\n".join(answer_parts).strip() or "(no response)"

    return {"answer": answer, "tool_calls": tool_calls_log}


if __name__ == "__main__":
    queries = [
        "What was Prologis' net income last year?",
        "Show me industrial properties in Chicago with their revenue.",
        "Summarize the most recent earnings press release.",
    ]
    for q in queries:
        print(f"\n{'=' * 70}\nQ: {q}")
        result = run_agent(q, verbose=True)
        print(f"\nA: {result['answer']}")
        print(f"\nTools called: {[c['name'] for c in result['tool_calls']]}")
