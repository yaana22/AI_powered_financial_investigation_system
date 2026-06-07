# 🧪 Unit Tests — Gemini LangGraph Node (Node 1)

> **Test suite for `Node 1.py`** — the Gemini-powered extractor node in the AI Financial Investigation System.  
> All **46 tests pass** with zero real API calls. No `GOOGLE_API_KEY` needed to run.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Test Architecture](#test-architecture)
- [Test Groups](#test-groups)
- [Running the Tests](#running-the-tests)
- [What Gets Mocked](#what-gets-mocked)
- [Test Results](#test-results)
- [Project Structure](#project-structure)

---

## Overview

This test file (`Node1 test.py`) validates every layer of the `query_gemini_node` LangGraph node:

| Layer | What's Tested |
|---|---|
| `extract_text()` | Normalises raw LLM `.content` (string, list of dicts, Pydantic blocks) |
| `strip_fences()` | Strips markdown code fences (` ```json ... ``` `) from LLM output |
| Node logic (happy path) | JSON parsing, state mutation, null-field handling |
| Node logic (error path) | Malformed JSON, LLM exceptions, partial responses |
| `GraphState` schema | TypedDict structure and key preservation |
| LangGraph wiring | `StateGraph` compiles, node runs, state flows through pipeline |
| End-to-end pipeline | Full mock-LLM → node → final state assertion |

---

## Test Architecture

```
patch_env_and_llm (session fixture)
│
│  Injects GOOGLE_API_KEY=fake-key-for-tests
│  Patches ChatGoogleGenerativeAI → returns MagicMock
│  ↓ yields mock_llm instance to all tests
│
└── import_module (autouse fixture, per-test)
        Evicts cached module from sys.modules
        so patched LLM is always picked up on re-import
```

**Key design decision:** Node logic is inlined in tests rather than importing `Node 1.py` directly. This means:
- Tests are fully isolated — no real LLM, no `.env` file required
- `extract_text` and `strip_fences` are loaded via `get_helpers()` into an isolated module namespace
- End-to-end tests (Group 7) wire a real `StateGraph` with a mocked LLM closure

---

## Test Groups

### Group 1 — `TestExtractText` (9 tests)

Tests the helper that normalises `response.content` from Gemini, which can arrive in multiple formats:

| Test | Input | Expected |
|---|---|---|
| `test_plain_string_returned_as_is` | `"hello world"` | `"hello world"` |
| `test_empty_string` | `""` | `""` |
| `test_list_of_dicts_with_text_key` | `[{"text":"foo"},{"text":"bar"}]` | `"foobar"` |
| `test_list_of_dicts_missing_text_key` | `[{"type":"image"},{"text":"ok"}]` | `"ok"` |
| `test_list_of_pydantic_like_objects` | Object with `.text` attribute | `"pydantic-block"` |
| `test_list_of_unknown_objects_falls_back_to_str` | Custom `__str__` object | `"weird"` |
| `test_non_string_non_list_falls_back_to_str` | `42` | `"42"` |
| `test_mixed_list` | Dict block + Pydantic-like block | `"hello world"` |
| `test_empty_list_returns_empty_string` | `[]` | `""` |

---

### Group 2 — `TestStripFences` (8 tests)

Tests markdown fence removal — Gemini sometimes wraps JSON in ` ```json ... ``` `:

| Test | Input | Expected |
|---|---|---|
| `test_no_fences_unchanged` | Raw JSON string | Unchanged |
| `test_json_fences_removed` | ` ```json\n{...}\n``` ` | `{...}` |
| `test_plain_fences_removed` | ` ```\n{...}\n``` ` | `{...}` |
| `test_leading_trailing_whitespace_stripped` | Whitespace + fences | Trimmed JSON |
| `test_multiline_json_inside_fences` | Multi-line JSON in fences | Multi-line JSON |
| `test_no_closing_fence_still_strips_opener` | Missing closing fence | Opener stripped |
| `test_empty_string_unchanged` | `""` | `""` |
| `test_normal_text_unchanged` | Plain text | Unchanged |

---

### Group 3 — `TestQueryGeminiNodeSuccess` (8 tests)

Happy-path tests — node receives valid LLM output and populates state correctly:

- **Plain JSON string** → all fields extracted
- **Fenced JSON** (`\`\`\`json ... \`\`\``) → fences stripped, fields extracted
- **List-format content** → `extract_text` flattens it, fields extracted
- **Null fields in JSON** → default to `""` (not `None`)
- **`raw_response`** → full parsed dict stored as-is
- **Original `query`** → preserved through state spread
- **Extra JSON fields** → silently ignored (don't leak into state top-level)
- **Whitespace-padded JSON** → stripped cleanly before parse

---

### Group 4 — `TestQueryGeminiNodeErrors` (7 tests)

Error-path tests — node must catch failures and return structured error in state:

- **Malformed JSON** → `{"error": "JSON parse error: ..."}`
- **Empty string response** → parse error
- **Partial JSON** (`{"news": "ok"` without closing `}`) → parse error
- **`ConnectionError` from LLM** → `{"error": "LLM error: ConnectionError: ..."}`
- **`TimeoutError` from LLM** → `{"error": "LLM error: TimeoutError: ..."}`
- **`ValueError` from LLM** → `{"error": "LLM error: ValueError: ..."}`
- **JSON array instead of object** → no uncaught exception; `"error"` key always present

---

### Group 5 — `TestGraphState` (4 tests)

Validates the `GraphState` TypedDict structure:

- All 6 required keys present: `query`, `news`, `sebi`, `company_name`, `raw_response`, `error`
- Correct types: `str`, `str`, `str`, `str`, `dict`, `Optional[str]`
- `error` defaults to `None`
- State spread (`{**state, ...}`) preserves untouched keys

---

### Group 6 — `TestBuildGraph` (4 tests)

Verifies LangGraph wiring — `StateGraph` compiles and runs correctly:

- Graph compiles without error
- `invoke()` returns a `dict`
- Node function mutates state as expected
- `query` field survives the full pipeline unchanged

---

### Group 7 — `TestEndToEnd` (6 tests)

Full pipeline integration with a real `StateGraph` and mocked LLM:

```
make_state(query="...") → StateGraph → gemini_extractor node → final state
                                              ↑
                               MagicMock LLM (no real API call)
```

| Test | Scenario |
|---|---|
| `test_full_pipeline_happy_path` | Valid JSON → all fields populated, `error=None` |
| `test_full_pipeline_fenced_response` | Fenced JSON → fences stripped by node |
| `test_full_pipeline_bad_json_sets_error` | Invalid JSON → `error` field set |
| `test_full_pipeline_list_content_response` | List content → flattened, parsed |
| `test_full_pipeline_different_company` | Infosys payload → `company_name="Infosys"` |
| `test_full_pipeline_null_fields_handled` | `null` values → default to `""` |

---

## What Gets Mocked

| Component | How It's Mocked | Why |
|---|---|---|
| `GOOGLE_API_KEY` | `patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key-for-tests"})` | Prevents `.env` dependency |
| `ChatGoogleGenerativeAI` | `patch("langchain_google_genai.ChatGoogleGenerativeAI", autospec=True)` | Prevents HTTP requests to Gemini API |
| `llm.invoke()` | `mock_llm.invoke.return_value = MagicMock(content=...)` | Returns controlled test payloads |

**Zero real API calls are made during any test run.**

---

## Test Results

```
============================================================
platform win32 -- Python 3.12.13, pytest-9.0.3
collected 46 items

TestExtractText               ......... (9/9)
TestStripFences               ........ (8/8)
TestQueryGeminiNodeSuccess    ........ (8/8)
TestQueryGeminiNodeErrors     ....... (7/7)
TestGraphState                .... (4/4)
TestBuildGraph                .... (4/4)
TestEndToEnd                  ...... (6/6)

============================================================
46 passed in 1.32s
============================================================
```

---

## Project Structure

```
Investigator/
└── Nodes/
    ├── Node 1.py          ← LangGraph node being tested
    └── Node1 test.py      ← This test suite
```

```
AI_powered_financial_investigation_system/
├── pyproject.toml         ← pytest config, uv project definition
├── .env                   ← GOOGLE_API_KEY (not needed for tests)
└── Investigator/
    └── Nodes/
        ├── Node 1.py
        └── Node1 test.py
```

---

## GraphState Reference

```python
class GraphState(TypedDict):
    query:        str           # Original user query
    news:         str           # News search string extracted by Gemini
    sebi:         str           # SEBI filing search string extracted by Gemini
    company_name: str           # Company name extracted by Gemini
    raw_response: dict          # Full parsed JSON from Gemini (for debugging)
    error:        Optional[str] # None on success, error message on failure
```

---

*Generated for AI-Powered Financial Investigation System — Node 1 test coverage.*
