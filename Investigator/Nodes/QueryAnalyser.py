import json
from dotenv import load_dotenv
from typing import TypedDict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

Gemini_model = "gemini-3.1-flash-lite"

# ── 1. Define the Graph State ─────────────────────────────────────────────────

class GraphState(TypedDict):
    query: str
    news: str
    sebi: str
    company_name: str
    raw_response: dict
    error: str | None

# ── 2. Initialize Gemini LLM ──────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(model=Gemini_model, temperature=0.2)


# ── Helper: safely extract text from response.content ────────────────────────

def extract_text(content) -> str:
    """
    response.content can be:
      - a plain str  (older langchain-google-genai)
      - a list of dicts like [{"type": "text", "text": "..."}]  (newer versions)
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif hasattr(block, "text"):          # Pydantic content block
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "".join(parts)

    return str(content)


# ── Helper: strip markdown fences ────────────────────────────────────────────

def strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrapping that Gemini sometimes adds."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first line (```json or ```) and last line (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    return text


# ── 3. The Node Function ──────────────────────────────────────────────────────

def query_gemini_node(state: GraphState) -> GraphState:
    query = state["query"]

    print("\n" + "━" * 52)
    print("🚀  Node: gemini_extractor  —  starting")
    print("━" * 52)
    print(f"📨  Query received  : {query}")
    print("🔄  Sending query to Gemini LLM …\n")

    system_prompt = """You are a financial data extraction assistant.
    Given a user query about a company or financial topic, extract the following fields
    and return ONLY a valid JSON object — no explanation, no markdown, no extra text.

    JSON schema:
    {
    "news"        : "Return a list about what to search on web search tools to find relevant news articles. If the query is not about news, return an empty string. minimum 4 keywords.",
    "sebi"        : "Return a list of what to search on SEBI vector database to find relevant filings or documents. If the query is not about a company, return an empty string. minimum 4 keywords.",
    "company_name": "<name of the company mentioned or inferred>"   
    }

    If a field cannot be determined from the query, set its value to null."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    try:
        response = llm.invoke(messages)

        # ── FIX: handle both str and list content ──
        raw_text = extract_text(response.content)
        print("✅  Response received from Gemini.")
        print(f"📄  Raw response preview : {raw_text[:120]} …\n")

        # ── Strip markdown fences if present ──
        clean_text = strip_fences(raw_text)
        print("🧹  Cleaned response (fences stripped):")
        print(f"    {clean_text[:200]}\n")

        # ── Parse JSON ──
        print("🔍  Parsing JSON …")
        parsed: dict = json.loads(clean_text)
        print("✅  JSON parsed successfully!\n")

        result: GraphState = {
            "query":        state["query"],
            "news":         parsed.get("news") or "",
            "sebi":         parsed.get("sebi") or "",
            "company_name": parsed.get("company_name") or "",
            "raw_response": parsed,
            "error":        None,
        }

        print("📦  Extracted fields:")
        print(f"    🏢  Company  : {result['company_name']}")
        print(f"    📰  News     : {result['news'][:100]}{'…' if len(result['news']) > 100 else ''}")
        print(f"    🏛️  SEBI     : {result['sebi'][:100]}{'…' if len(result['sebi']) > 100 else ''}")
        print("\n" + "━" * 52)
        print("🏁  Node: gemini_extractor  —  done")
        print("━" * 52)

        return result

    except json.JSONDecodeError as e:
        print(f"❌  JSON parse error: {e}")
        print(f"    Offending text: {clean_text[:300]}")
        return {**state, "error": f"JSON parse error: {e}"}

    except Exception as e:
        print(f"❌  Unexpected error: {type(e).__name__}: {e}")
        return {**state, "error": f"LLM error: {type(e).__name__}: {e}"}

