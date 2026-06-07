# requirements:
# pip install langgraph langchain-google-genai langchain-core

import json
from dotenv import load_dotenv
from typing import TypedDict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from QueryAnalyser import query_gemini_node


class GraphState(TypedDict):
    query: str
    news: str
    sebi: str
    company_name: str
    raw_response: dict
    error: str | None



# ── 4. Build the LangGraph ────────────────────────────────────────────────────

def build_graph() -> Any:
    print("\n🔧  Building LangGraph …")
    graph = StateGraph(GraphState)
    graph.add_node("gemini_extractor", query_gemini_node)
    graph.set_entry_point("gemini_extractor")
    graph.add_edge("gemini_extractor", END)
    compiled = graph.compile()
    print("✅  Graph compiled successfully!\n")
    return compiled


# ── 5. Run it ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("  💼  AI Financial Investigation System")
    print("  📊  Powered by Gemini + LangGraph")
    print("=" * 52)

    app = build_graph()

    initial_state: GraphState = {
        "query": "What is the latest SEBI news about Reliance Industries?",
        "news": "",
        "sebi": "",
        "company_name": "",
        "raw_response": {},
        "error": None,
    }

    print(f"🎯  Running graph with query:\n    \"{initial_state['query']}\"\n")
    result = app.invoke(initial_state)

    print("\n" + "=" * 52)
    print("  📋  FINAL RESULTS")
    print("=" * 52)

    if result["error"]:
        print(f"  ❌  Error : {result['error']}")
    else:
        print(f"  🏢  Company  : {result['company_name']}")
        print(f"  📰  News     : {result['news']}")
        print(f"  🏛️  SEBI     : {result['sebi']}")
        print(f"\n  📦  Full JSON:\n")
        print(json.dumps(result["raw_response"], indent=4))

    print("=" * 52)
    print("  ✅  Done!")
    print("=" * 52)