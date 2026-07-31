#!/usr/bin/env python3
"""
CrewAI + MemOS OTel Demo (memory-semconv v0.1.0)

Generates real end-to-end traces in Dynatrace:
  crewai.crew span
    └── crewai.task (research)
          └── crewai.agent (Memory Research Specialist)
                ├── memory.add   (store quantum computing facts)
                └── memory.search (verify retrieval)
    └── crewai.task (analysis)
          └── crewai.agent (Memory Quality Analyst)
                ├── memory.search (find relevant memories)
                ├── memory.update (refresh knowledge)
                └── memory.delete (prune stale entry)

Usage:
    OTLP_ENDPOINT=http://localhost:4317 python3 examples/crewai_memos_demo.py

No LLM API key required — agent decision-making is scripted for the demo.
Real memory.* OTel spans flow to the OTLP collector and appear in Dynatrace.
"""
import logging
import os
import sys
import time
import importlib.util
import pathlib

# ---------------------------------------------------------------------------
# 0. Add src/ to path so memos.telemetry is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ---------------------------------------------------------------------------
# 1. Load memos.telemetry and configure OTel BEFORE anything else
# ---------------------------------------------------------------------------
_tel_path = pathlib.Path(__file__).parent.parent / "src" / "memos" / "telemetry.py"
_spec = importlib.util.spec_from_file_location("memos.telemetry", _tel_path)
tel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tel)

OTLP_ENDPOINT = os.environ.get("OTLP_ENDPOINT", "http://localhost:4317")
tel.configure(
    service_name="memos-crewai-demo",
    service_version="1.0.0",
    otlp_endpoint=OTLP_ENDPOINT,
    export_interval_ms=2000,
)
print(f"[OTel] Configured: service=memos-crewai-demo endpoint={OTLP_ENDPOINT}")

# ---------------------------------------------------------------------------
# 2. Install CrewAI instrumentation to capture agent/task spans automatically
# ---------------------------------------------------------------------------
try:
    from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
    CrewAIInstrumentor().instrument()
    print("[OTel] CrewAIInstrumentor active")
except ImportError:
    print("[OTel] opentelemetry-instrumentation-crewai not installed; using manual spans")

from opentelemetry import trace
tracer = tel.get_tracer()

# ---------------------------------------------------------------------------
# 3. Memory helper functions — emit real memory.* OTel spans
# ---------------------------------------------------------------------------

def memory_add(content: str, user_id: str, cube_id: str) -> str:
    with tel.memory_span("add", tel.TIER_TEXTUAL, {
        tel.MEMORY_USER_ID: user_id,
        tel.MEMORY_CUBE_ID: cube_id,
        tel.MEMORY_ITEM_COUNT: 1,
        tel.MEMORY_SESSION_ID: "crewai-demo-session",
    }) as span:
        time.sleep(0.06)
        memory_id = f"mem-{int(time.time() * 1000) % 100000:05d}"
        span.set_attribute("memory.stored.id", memory_id)
        span.set_attribute("memory.content.length", len(content))
        tel.emit_op_log(
            logging.INFO, "add", tel.TIER_TEXTUAL,
            "agent stored memory item",
            {tel.MEMORY_USER_ID: user_id, "memory.stored.id": memory_id},
        )
    return memory_id


def memory_search(query: str, user_id: str, cube_id: str, top_k: int = 5) -> list:
    with tel.memory_span("search", tel.TIER_TEXTUAL, {
        tel.MEMORY_USER_ID: user_id,
        tel.MEMORY_CUBE_ID: cube_id,
        tel.MEMORY_QUERY_TEXT: query[:256],
        tel.MEMORY_TOP_K: top_k,
        tel.MEMORY_SESSION_ID: "crewai-demo-session",
    }) as span:
        time.sleep(0.09)
        results = [
            {"id": "mem-00001", "score": 0.95, "content": "Quantum computing: qubits enable superposition-based parallel computation"},
            {"id": "mem-00002", "score": 0.88, "content": "Google achieved 1000-qubit milestone in 2025"},
            {"id": "mem-00003", "score": 0.82, "content": "IBM roadmap: 100k qubits by 2033; applications in cryptography and drug discovery"},
        ]
        span.set_attribute(tel.MEMORY_RESULT_COUNT, len(results))
        span.set_attribute("memory.result.top_score", results[0]["score"])
        tel.record_result_count(len(results), tel.TIER_TEXTUAL)
        tel.emit_op_log(
            logging.INFO, "search", tel.TIER_TEXTUAL,
            f"agent search returned {len(results)} results",
            {tel.MEMORY_RESULT_COUNT: len(results), tel.MEMORY_USER_ID: user_id},
        )
    return results


def memory_update(memory_id: str, content: str, user_id: str, cube_id: str) -> bool:
    with tel.memory_span("update", tel.TIER_TEXTUAL, {
        tel.MEMORY_USER_ID: user_id,
        tel.MEMORY_CUBE_ID: cube_id,
        tel.MEMORY_SESSION_ID: "crewai-demo-session",
    }) as span:
        time.sleep(0.04)
        span.set_attribute("memory.updated.id", memory_id)
        span.set_attribute("memory.content.length", len(content))
        tel.emit_op_log(
            logging.INFO, "update", tel.TIER_TEXTUAL,
            f"agent updated memory: {memory_id}",
            {"memory.updated.id": memory_id, tel.MEMORY_USER_ID: user_id},
        )
    return True


def memory_delete(memory_id: str, user_id: str, cube_id: str) -> bool:
    with tel.memory_span("delete", tel.TIER_TEXTUAL, {
        tel.MEMORY_USER_ID: user_id,
        tel.MEMORY_CUBE_ID: cube_id,
        tel.MEMORY_SESSION_ID: "crewai-demo-session",
    }) as span:
        time.sleep(0.02)
        span.set_attribute("memory.deleted.id", memory_id)
        tel.emit_op_log(
            logging.INFO, "delete", tel.TIER_TEXTUAL,
            f"agent deleted stale memory: {memory_id}",
            {"memory.deleted.id": memory_id, tel.MEMORY_USER_ID: user_id},
        )
    return True


# ---------------------------------------------------------------------------
# 4. Simulate CrewAI crew execution with nested spans
#    Hierarchy: crewai.crew -> crewai.task -> crewai.agent -> memory.*
# ---------------------------------------------------------------------------

print("\n" + "=" * 62)
print("  CrewAI + MemOS OTel Demo — sending telemetry to Dynatrace")
print(f"  OTLP endpoint: {OTLP_ENDPOINT}")
print("=" * 62 + "\n")

with tracer.start_as_current_span(
    "crewai.crew",
    kind=trace.SpanKind.INTERNAL,
    attributes={
        "crewai.crew.name": "memos-research-crew",
        "crewai.crew.process": "sequential",
        "crewai.crew.agent_count": 2,
        "crewai.crew.task_count": 2,
    },
) as crew_span:
    print("[Crew] memos-research-crew started")

    # ---- Task 1: Research Agent stores and retrieves knowledge ----
    with tracer.start_as_current_span(
        "crewai.task",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "crewai.task.name": "research_quantum_computing",
            "crewai.task.description": (
                "Store quantum computing research in MemOS and verify retrieval."
            ),
        },
    ) as task1_span:
        print("\n[Task 1] Research Quantum Computing")

        with tracer.start_as_current_span(
            "crewai.agent",
            kind=trace.SpanKind.INTERNAL,
            attributes={
                "crewai.agent.role": "Memory Research Specialist",
                "crewai.agent.goal": "Ingest research findings into MemOS and validate retrieval",
            },
        ) as agent1_span:
            print("  [Agent] Memory Research Specialist")
            time.sleep(0.1)

            print("  [Tool] store_memory — quantum computing facts")
            mid = memory_add(
                "Quantum computing 2025: Google achieved 1000-qubit milestone. "
                "IBM roadmap: 100k qubits by 2033. Applications: cryptography, "
                "drug discovery, financial optimization.",
                user_id="researcher-agent",
                cube_id="research-cube",
            )
            agent1_span.set_attribute("crewai.agent.tool_calls", 1)
            print(f"    stored as {mid}")

            print("  [Tool] search_memory — verify storage")
            results = memory_search(
                "quantum computing milestones 2025",
                user_id="researcher-agent",
                cube_id="research-cube",
            )
            agent1_span.set_attribute("crewai.agent.tool_calls", 2)
            agent1_span.set_attribute("crewai.agent.search_results", len(results))
            print(f"    found {len(results)} memories (top score: {results[0]['score']})")
            agent1_span.set_attribute("crewai.agent.status", "completed")

        task1_span.set_attribute("crewai.task.status", "completed")
        task1_span.set_attribute("crewai.task.memories_stored", 1)
        task1_span.set_attribute("crewai.task.memories_retrieved", len(results))
        print("[Task 1] Completed")

    # ---- Task 2: Analyst Agent searches, updates and prunes memories ----
    with tracer.start_as_current_span(
        "crewai.task",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "crewai.task.name": "analyse_and_curate_memory",
            "crewai.task.description": (
                "Search, update and prune quantum computing memories."
            ),
        },
    ) as task2_span:
        print("\n[Task 2] Analyse and Curate Memory")

        with tracer.start_as_current_span(
            "crewai.agent",
            kind=trace.SpanKind.INTERNAL,
            attributes={
                "crewai.agent.role": "Memory Quality Analyst",
                "crewai.agent.goal": "Validate memory relevance and prune stale entries",
            },
        ) as agent2_span:
            print("  [Agent] Memory Quality Analyst")
            time.sleep(0.1)

            print("  [Tool] search_memory — find quantum computing entries")
            results2 = memory_search(
                "quantum computing recent developments IBM Google",
                user_id="analyst-agent",
                cube_id="research-cube",
                top_k=10,
            )
            agent2_span.set_attribute("crewai.agent.tool_calls", 1)
            print(f"    found {len(results2)} memories to analyse")

            print("  [Tool] update_memory — enrich top result")
            updated = memory_update(
                results2[0]["id"],
                "Quantum computing 2025 [ENRICHED]: Google Willow processor 1000 qubits "
                "with error correction breakthrough. IBM Eagle: 127 qubits production-ready.",
                user_id="analyst-agent",
                cube_id="research-cube",
            )
            agent2_span.set_attribute("crewai.agent.tool_calls", 2)
            print(f"    updated {results2[0]['id']}: {updated}")

            print("  [Tool] delete_memory — prune stale entry")
            deleted = memory_delete(
                "mem-stale-001",
                user_id="analyst-agent",
                cube_id="research-cube",
            )
            agent2_span.set_attribute("crewai.agent.tool_calls", 3)
            print(f"    deleted mem-stale-001: {deleted}")
            agent2_span.set_attribute("crewai.agent.status", "completed")

        task2_span.set_attribute("crewai.task.status", "completed")
        task2_span.set_attribute("crewai.task.memories_analysed", len(results2))
        task2_span.set_attribute("crewai.task.memories_updated", 1)
        task2_span.set_attribute("crewai.task.memories_deleted", 1)
        print("[Task 2] Completed")

    crew_span.set_attribute("crewai.crew.status", "success")
    crew_span.set_attribute("crewai.crew.total_memory_ops", 5)

print("\n" + "=" * 62)
print("  Crew done — 5 memory operations across 2 agents:")
print("    1x memory.add    researcher stores quantum facts")
print("    2x memory.search researcher verifies + analyst queries")
print("    1x memory.update analyst enriches top memory")
print("    1x memory.delete analyst prunes stale entry")
print()
print("  Flushing OTel data to Dynatrace (waiting 4s)...")
time.sleep(4)
print("  Done. Filter in Dynatrace by:")
print("    service.name = memos-crewai-demo")
print("    OR span.name starts with 'crewai.'")
print("    OR memory.tier = textual")
