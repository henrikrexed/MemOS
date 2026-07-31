#!/usr/bin/env python3
"""
CrewAI + MemOS Real Agent Demo (memory-semconv v0.1.0)

Real architecture:
  CrewAI Agent (Ollama qwen3.6:latest LLM @ 10.0.0.185)
    → decides to call MemOS memory tools
    → MemOS stores/retrieves memories using Ollama nomic-embed-text embeddings
    → all operations emit real OTel spans → Dynatrace

Requirements:
  - Ollama at OLLAMA_HOST (default: http://10.0.0.185:11434) with qwen3.6:latest + nomic-embed-text
  - otelcol-contrib running on localhost:4317
  - NAS venv: /mnt/nas/tools/memos-crewai-env

Usage:
  /mnt/nas/tools/memos-crewai-env/bin/python3.13 examples/crewai_memos_real_demo.py
  OLLAMA_HOST=http://10.0.0.185:11434 OLLAMA_CHAT_MODEL=qwen3.6:latest ... (above)
"""
import logging
import os
import sys
import time
import importlib.util
import pathlib
import json
import math

# ---------------------------------------------------------------------------
# 0. Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ---------------------------------------------------------------------------
# 1. Configure OTel BEFORE anything imports it
# ---------------------------------------------------------------------------
_tel_path = pathlib.Path(__file__).parent.parent / "src" / "memos" / "telemetry.py"
_spec = importlib.util.spec_from_file_location("memos.telemetry", _tel_path)
tel = importlib.util.module_from_spec(_spec)
sys.modules["memos.telemetry"] = tel
_spec.loader.exec_module(tel)

OTLP_ENDPOINT = os.environ.get("OTLP_ENDPOINT", "http://localhost:4317")
tel.configure(
    service_name="memos-crewai-real",
    service_version="1.0.0",
    otlp_endpoint=OTLP_ENDPOINT,
    export_interval_ms=2000,
)
print(f"[OTel] Configured: service=memos-crewai-real endpoint={OTLP_ENDPOINT}")

# ---------------------------------------------------------------------------
# 2. CrewAI OTel instrumentation
# ---------------------------------------------------------------------------
try:
    from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
    CrewAIInstrumentor().instrument()
    print("[OTel] CrewAIInstrumentor active")
except ImportError:
    print("[OTel] CrewAIInstrumentor not available")

from opentelemetry import trace
tracer = tel.get_tracer()

# ---------------------------------------------------------------------------
# 2b. Qwen3.6 compatibility hook
#     Qwen3.6-nothink returns empty when "Observation:" is appended to
#     assistant messages (CrewAI ReAct default). This hook moves each
#     Observation into a separate user message so the model responds correctly.
# ---------------------------------------------------------------------------
from crewai.hooks.llm_hooks import register_before_llm_call_hook


def _qwen36_observation_hook(ctx) -> None:
    """Fix qwen3.6 ReAct incompatibility: the model returns empty when:
      1. A user message ends with 'Thought: ...' (CrewAI's initial prompt suffix)
      2. An assistant message contains 'Observation:' (should be a user message)
    Both fixes move those fragments to the correct role.
    """
    reformatted = []
    changed = False
    for msg in ctx.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and "\n\nThought:" in content:
            # Move trailing "Thought: ..." into a separate assistant message
            parts = content.split("\n\nThought:", 1)
            reformatted.append({"role": "user", "content": parts[0].rstrip()})
            reformatted.append({"role": "assistant", "content": "Thought:" + parts[1]})
            changed = True
        elif role == "assistant" and "\nObservation:" in content:
            # Move "Observation: ..." into a user message
            parts = content.split("\nObservation:", 1)
            reformatted.append({"role": "assistant", "content": parts[0].rstrip()})
            reformatted.append({"role": "user", "content": "Observation: " + parts[1].lstrip()})
            changed = True
        else:
            reformatted.append(msg)
    if changed:
        ctx.messages[:] = reformatted


register_before_llm_call_hook(_qwen36_observation_hook)
print("[Hook] Qwen3.6 observation-reformat hook registered")

# ---------------------------------------------------------------------------
# 3. Lightweight in-process MemOS memory backend
#    Uses Ollama for embeddings (nomic-embed-text) + semantic cosine search
#    No qdrant/sentence_transformers needed
# ---------------------------------------------------------------------------
import ollama as _ollama_lib

def _ollama_client():
    return _ollama_lib.Client(host=OLLAMA_HOST)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://10.0.0.185:11434")
# qwen3.6-nothink is qwen3.6:latest with PARAMETER think=false so CrewAI
# can parse responses without stripping <think>...</think> tokens.
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen3.6-nothink")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# In-memory store: list of {id, content, embedding, metadata}
_memory_store: list[dict] = []
_memory_id_counter = 0


def _embed(text: str) -> list[float]:
    resp = _ollama_client().embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
    return resp["embedding"]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-9)


def memos_add(content: str, user_id: str = "crewai-agent", cube_id: str = "research-cube") -> str:
    """Store memory item with OTel instrumentation."""
    global _memory_id_counter
    _memory_id_counter += 1
    memory_id = f"mem-{_memory_id_counter:04d}"

    with tel.memory_span("add", tel.TIER_TEXTUAL, {
        tel.MEMORY_USER_ID: user_id,
        tel.MEMORY_CUBE_ID: cube_id,
        tel.MEMORY_ITEM_COUNT: 1,
        tel.MEMORY_SESSION_ID: "real-agent-session",
        "memory.backend": "ollama",
    }) as span:
        embedding = _embed(content)
        _memory_store.append({
            "id": memory_id,
            "content": content,
            "embedding": embedding,
            "metadata": {"user_id": user_id, "cube_id": cube_id},
        })
        span.set_attribute("memory.stored.id", memory_id)
        span.set_attribute("memory.content.length", len(content))
        span.set_attribute("memory.embedding.dims", len(embedding))
        tel.emit_op_log(
            logging.INFO, "add", tel.TIER_TEXTUAL,
            f"stored memory {memory_id} ({len(content)} chars)",
            {tel.MEMORY_USER_ID: user_id, "memory.stored.id": memory_id},
        )
    return memory_id


def memos_search(query: str, user_id: str = "crewai-agent", top_k: int = 3) -> list[dict]:
    """Semantic search with cosine similarity + OTel instrumentation."""
    with tel.memory_span("search", tel.TIER_TEXTUAL, {
        tel.MEMORY_USER_ID: user_id,
        tel.MEMORY_QUERY_TEXT: query[:256],
        tel.MEMORY_TOP_K: top_k,
        "memory.backend": "ollama",
        "memory.total_indexed": len(_memory_store),
    }) as span:
        if not _memory_store:
            span.set_attribute(tel.MEMORY_RESULT_COUNT, 0)
            return []

        query_embedding = _embed(query)
        scored = [
            {"item": m, "score": _cosine_sim(query_embedding, m["embedding"])}
            for m in _memory_store
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:top_k]

        span.set_attribute(tel.MEMORY_RESULT_COUNT, len(results))
        if results:
            span.set_attribute("memory.result.top_score", round(results[0]["score"], 4))
        tel.record_result_count(len(results), tel.TIER_TEXTUAL)
        tel.emit_op_log(
            logging.INFO, "search", tel.TIER_TEXTUAL,
            f"search returned {len(results)}/{len(_memory_store)} memories",
            {tel.MEMORY_RESULT_COUNT: len(results), "memory.query.text": query[:80]},
        )
    return [{"id": r["item"]["id"], "content": r["item"]["content"], "score": r["score"]}
            for r in results]


def memos_update(memory_id: str, new_content: str, user_id: str = "crewai-agent") -> bool:
    """Update memory content + re-embed + OTel instrumentation."""
    with tel.memory_span("update", tel.TIER_TEXTUAL, {
        tel.MEMORY_USER_ID: user_id,
        tel.MEMORY_CUBE_ID: "research-cube",
        "memory.backend": "ollama",
    }) as span:
        for item in _memory_store:
            if item["id"] == memory_id:
                item["content"] = new_content
                item["embedding"] = _embed(new_content)
                span.set_attribute("memory.updated.id", memory_id)
                span.set_attribute("memory.content.length", len(new_content))
                tel.emit_op_log(
                    logging.INFO, "update", tel.TIER_TEXTUAL,
                    f"updated memory {memory_id}",
                    {"memory.updated.id": memory_id},
                )
                return True
        span.set_attribute("memory.updated.id", memory_id)
        span.set_attribute("memory.update.found", False)
    return False


def memos_delete(memory_id: str, user_id: str = "crewai-agent") -> bool:
    """Delete memory + OTel instrumentation."""
    global _memory_store
    with tel.memory_span("delete", tel.TIER_TEXTUAL, {
        tel.MEMORY_USER_ID: user_id,
        tel.MEMORY_CUBE_ID: "research-cube",
        "memory.backend": "ollama",
    }) as span:
        before = len(_memory_store)
        _memory_store = [m for m in _memory_store if m["id"] != memory_id]
        deleted = before > len(_memory_store)
        span.set_attribute("memory.deleted.id", memory_id)
        span.set_attribute("memory.delete.success", deleted)
        tel.emit_op_log(
            logging.INFO, "delete", tel.TIER_TEXTUAL,
            f"{'deleted' if deleted else 'not found'}: {memory_id}",
            {"memory.deleted.id": memory_id},
        )
    return deleted


# ---------------------------------------------------------------------------
# 4. CrewAI Tools wrapping the real MemOS functions above
# ---------------------------------------------------------------------------
from crewai.tools import BaseTool
from typing import Optional


class MemOSStoreTool(BaseTool):
    name: str = "store_in_memory"
    description: str = (
        "Store a new fact or finding in MemOS long-term memory. "
        "Use this when you learn something important to remember for later. "
        "Input: the text content to store."
    )

    def _run(self, content: str) -> str:
        mid = memos_add(content.strip(), user_id="researcher-agent")
        return f"Stored in MemOS with id={mid}. Content: {content[:60]}..."


class MemOSSearchTool(BaseTool):
    name: str = "search_memory"
    description: str = (
        "Search MemOS long-term memory for relevant information using semantic similarity. "
        "Use this to retrieve previously stored knowledge. "
        "Input: your search query."
    )

    def _run(self, query: str) -> str:
        results = memos_search(query.strip(), user_id="researcher-agent", top_k=3)
        if not results:
            return "No memories found."
        lines = [f"[{r['id']}] (score={r['score']:.3f}) {r['content']}" for r in results]
        return "Found memories:\n" + "\n".join(lines)


class MemOSUpdateTool(BaseTool):
    name: str = "update_memory"
    description: str = (
        "Update an existing memory entry with new or enriched content. "
        "Input: JSON with 'id' (the memory id) and 'content' (the new text)."
    )

    def _run(self, input_json: str) -> str:
        try:
            data = json.loads(input_json)
            mid = data["id"]
            content = data["content"]
        except (json.JSONDecodeError, KeyError):
            # Fallback: treat as plain text update for the last stored memory
            if _memory_store:
                mid = _memory_store[-1]["id"]
                content = input_json
            else:
                return "No memories to update."
        ok = memos_update(mid, content, user_id="analyst-agent")
        return f"{'Updated' if ok else 'Not found'}: memory {mid}"


class MemOSDeleteTool(BaseTool):
    name: str = "delete_memory"
    description: str = (
        "Delete a memory entry by its id when it is stale or incorrect. "
        "Input: the memory id to delete."
    )

    def _run(self, memory_id: str) -> str:
        ok = memos_delete(memory_id.strip(), user_id="analyst-agent")
        return f"{'Deleted' if ok else 'Not found'}: {memory_id.strip()}"


# ---------------------------------------------------------------------------
# 5. Ollama LLM wrapper for CrewAI
#    CrewAI expects an OpenAI-compatible endpoint; Ollama provides one at /v1
# ---------------------------------------------------------------------------
from crewai import LLM

ollama_llm = LLM(
    model=f"ollama/{OLLAMA_CHAT_MODEL}",
    base_url=OLLAMA_HOST,
    timeout=600,  # qwen3.6 (22GB) needs ~2 min per call on CPU
)

# ---------------------------------------------------------------------------
# 6. CrewAI Agents
# ---------------------------------------------------------------------------
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Memory Research Specialist",
    goal=(
        "Research and store key facts about AI memory systems in MemOS. "
        "Always store findings in memory, then verify retrieval works."
    ),
    backstory=(
        "You are an AI researcher specializing in agent memory systems. "
        "You use MemOS as your persistent knowledge store. "
        "You carefully store research findings and verify they can be retrieved."
    ),
    tools=[MemOSStoreTool(), MemOSSearchTool()],
    llm=ollama_llm,
    verbose=True,
    max_iter=6,
    allow_delegation=False,
)

analyst = Agent(
    role="Memory Quality Analyst",
    goal=(
        "Search MemOS for stored research findings, update them with enriched details, "
        "and prune any outdated entries to maintain memory quality."
    ),
    backstory=(
        "You are a quality analyst who reviews and curates the MemOS knowledge base. "
        "You search for stored memories, update them with enriched information, "
        "and delete stale entries."
    ),
    tools=[MemOSSearchTool(), MemOSUpdateTool(), MemOSDeleteTool()],
    llm=ollama_llm,
    verbose=True,
    max_iter=6,
    allow_delegation=False,
)

# ---------------------------------------------------------------------------
# 7. Tasks
# ---------------------------------------------------------------------------
research_task = Task(
    description=(
        "Use your tools to do the following:\n"
        "1. Store this fact in MemOS: 'MemOS implements L1/L2/L3 memory tiers: "
        "textual (episodic/semantic), activation (KV-cache), and parametric (model weights). "
        "Memory operations are instrumented with OpenTelemetry (memory-semconv v0.1.0) "
        "to emit spans, metrics and logs to Dynatrace.'\n"
        "2. Search MemOS for 'memory tiers' to verify it was stored correctly.\n"
        "Report the memory id of what you stored and the search result score."
    ),
    expected_output=(
        "The memory id of the stored fact and confirmation that search retrieved it "
        "with a similarity score above 0.7."
    ),
    agent=researcher,
)

analysis_task = Task(
    description=(
        "Use your tools to do the following:\n"
        "1. Search MemOS for 'OpenTelemetry memory instrumentation'.\n"
        "2. Update the top result with this enriched content: "
        "'MemOS OTel instrumentation (memory-semconv v0.1.0): spans on add/search/update/delete, "
        "memory.tier attribute (textual/activation/parametric/preference), "
        "W3C TraceContext propagation for end-to-end Dynatrace traces. "
        "CrewAI agents use Ollama (qwen2.5:0.5b) and emit real telemetry.'\n"
        "Report what you found and what you updated."
    ),
    expected_output=(
        "Summary of what was found in memory, which entry was updated, and confirmation "
        "of the update with the enriched content."
    ),
    agent=analyst,
)

# ---------------------------------------------------------------------------
# 8. Run the crew
# ---------------------------------------------------------------------------
crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task],
    process=Process.sequential,
    verbose=True,
)

print("\n" + "=" * 66)
print("  CrewAI + MemOS Real Agent Demo")
print(f"  LLM:       Ollama {OLLAMA_CHAT_MODEL} @ {OLLAMA_HOST}")
print(f"  Embedder:  Ollama {OLLAMA_EMBED_MODEL} @ {OLLAMA_HOST}")
print(f"  OTel:      {OTLP_ENDPOINT} → Dynatrace")
print("=" * 66 + "\n")

# Run inside a root span so the whole session is one trace in Dynatrace
with tracer.start_as_current_span(
    "crewai.session",
    kind=trace.SpanKind.SERVER,
    attributes={
        "crewai.crew.name": "memos-research-crew",
        "crewai.llm.model": OLLAMA_CHAT_MODEL,
        "crewai.llm.provider": "ollama",
        "crewai.llm.host": OLLAMA_HOST,
        "memory.backend": "ollama",
        "memory.embed_model": OLLAMA_EMBED_MODEL,
    },
) as root_span:
    result = crew.kickoff()
    root_span.set_attribute("crewai.session.status", "completed")
    root_span.set_attribute("crewai.memory.total_items", len(_memory_store))

print("\n" + "=" * 66)
print("  Crew result:")
print(result)
print()
print(f"  Memory store now has {len(_memory_store)} item(s)")
print()
print("  Flushing OTel data to Dynatrace (waiting 5s)...")
time.sleep(5)
print("  Done. In Dynatrace, filter by:")
print("    service.name = memos-crewai-real")
print("    OR memory.tier = textual")
print("    OR crewai.llm.model = qwen2.5:0.5b")
