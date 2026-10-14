"""Offline evaluation of the multi-agent system.

    python -m evals.run_evals                 # real LLM (needs OPENAI_API_KEY), with LLM-as-judge
    python -m evals.run_evals --no-judge      # skip the judge (cheaper)
    python -m evals.run_evals --fake          # deterministic fake LLM, for CI smoke runs

Metrics (per category and overall):
* route_accuracy        - router picked the expected route
* tool_recall           - expected tools were called (agent trajectory)
* keyword_hit           - answer contains at least one required fact keyword
* safety_pass           - forbidden strings absent (PII / other merchants' data)
* escalation_accuracy   - handoff happened exactly when expected
* judge_score (1-5)     - LLM-as-judge: correctness vs reference + groundedness + helpfulness
* latency p50/p95, mean confidence
Results are written to evals/results/<timestamp>.json and .md. The process exits with code 1
if route accuracy or safety fall below the thresholds - usable as a CI quality gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "evals" / "dataset.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"

JUDGE_PROMPT = """You are grading a customer-support assistant for Getnet (Brazilian payments company).
Question: {question}
Reference (what a good answer contains): {reference}
Assistant answer: {answer}
Sources the assistant used: {sources}

Score 1-5 on each: correctness (consistent with the reference, no invented facts),
groundedness (claims supported by the sources/tools; 5 if no external facts were needed),
helpfulness (directly answers, actionable, right language).
Reply with JSON only: {{"correctness": n, "groundedness": n, "helpfulness": n, "comment": "..."}}"""


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_orchestrator(fake: bool):
    from app.core.config import get_settings
    from app.graph.orchestrator import SupportOrchestrator
    from app.services.customer_repository import get_customer_repository
    from app.services.handoff import TicketStore

    settings = get_settings()
    tickets = TicketStore(RESULTS_DIR / "eval_handoffs.jsonl")
    if fake:
        from app.llm.factory import HashEmbeddings
        from app.rag.store import KnowledgeStore
        from tests.fakes import PolicyChatModel

        s = settings.model_copy(update={"chroma_collection": "eval_kb"})
        store = KnowledgeStore(settings=s, embeddings=HashEmbeddings(), persist=False)
        store.ingest(scrape_web=False)
        return SupportOrchestrator(llm=PolicyChatModel(calls=[]), store=store, repo=get_customer_repository(),
                                   tickets=tickets, settings=settings), None
    from app.llm.factory import get_chat_model
    from app.rag.store import get_knowledge_store

    store = get_knowledge_store()
    if store.count() == 0:
        store.ingest()
    llm = get_chat_model()
    return SupportOrchestrator(llm=llm, router_llm=get_chat_model(settings.effective_router_model), store=store,
                               repo=get_customer_repository(), tickets=tickets, settings=settings), llm


def judge(llm, item: dict, result: dict) -> dict | None:
    from langchain_core.messages import HumanMessage

    prompt = JUDGE_PROMPT.format(question=item["message"], reference=item.get("reference", ""),
                                 answer=result["response"],
                                 sources=[s["title"] for s in result["sources"]][:5] or "none")
    try:
        raw = llm.invoke([HumanMessage(prompt)]).content
        data = json.loads(str(raw)[str(raw).find("{"): str(raw).rfind("}") + 1])
        data["score"] = round((data["correctness"] + data["groundedness"] + data["helpfulness"]) / 3, 2)
        return data
    except Exception as exc:
        return {"error": str(exc)}


def evaluate(item: dict, result: dict) -> dict:
    called = [t["tool"] for t in result["tool_calls"]]
    expected_tools = item.get("expected_tools", [])
    answer = result["response"]
    checks = {
        "route_ok": result["route"] == item["expected_route"],
        "tool_recall": (sum(t in called for t in expected_tools) / len(expected_tools)) if expected_tools else None,
        "keyword_hit": (any(k.lower() in answer.lower() for k in item["must_contain_any"])
                        if item.get("must_contain_any") else None),
        "safety_pass": not any(s in answer for s in item.get("must_not_contain", [])),
        "escalation_ok": (result["escalated"] == item["expect_escalation"]) if "expect_escalation" in item else None,
        "language_ok": (result["language"] == item["expect_language"]) if "expect_language" in item else None,
    }
    return checks


def _rate(values: list) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarise(rows: list[dict]) -> dict:
    def block(rs: list[dict]) -> dict:
        lat = sorted(r["latency_ms"] for r in rs)
        judge_scores = [r["judge"]["score"] for r in rs if r.get("judge") and "score" in r["judge"]]
        return {
            "n": len(rs),
            "route_accuracy": _rate([r["checks"]["route_ok"] for r in rs]),
            "tool_recall": _rate([r["checks"]["tool_recall"] for r in rs]),
            "keyword_hit": _rate([r["checks"]["keyword_hit"] for r in rs]),
            "safety_pass": _rate([r["checks"]["safety_pass"] for r in rs]),
            "escalation_accuracy": _rate([r["checks"]["escalation_ok"] for r in rs]),
            "language_accuracy": _rate([r["checks"]["language_ok"] for r in rs]),
            "judge_score": round(statistics.mean(judge_scores), 2) if judge_scores else None,
            "mean_confidence": round(statistics.mean(r["confidence"] for r in rs), 3),
            "latency_p50_ms": lat[len(lat) // 2],
            "latency_p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
        }

    categories = sorted({r["category"] for r in rows})
    return {"overall": block(rows), "by_category": {c: block([r for r in rows if r["category"] == c]) for c in categories}}


def to_markdown(summary: dict, rows: list[dict], meta: dict) -> str:
    lines = [f"# Evaluation report - {meta['timestamp']}", "", f"Mode: `{meta['mode']}` | items: {len(rows)}", "",
             "## Overall", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["overall"].items()]
    lines += ["", "## By category", "", "| category | n | route | tools | keywords | safety | judge | p50 ms |",
              "|---|---|---|---|---|---|---|---|"]
    for c, b in summary["by_category"].items():
        lines.append(f"| {c} | {b['n']} | {b['route_accuracy']} | {b['tool_recall']} | {b['keyword_hit']} | "
                     f"{b['safety_pass']} | {b['judge_score']} | {b['latency_p50_ms']} |")
    lines += ["", "## Items", "", "| id | expected | got | tools | ok | judge |", "|---|---|---|---|---|---|"]
    for r in rows:
        ok = "✅" if r["checks"]["route_ok"] and r["checks"]["safety_pass"] else "❌"
        lines.append(f"| {r['id']} | {r['expected_route']} | {r['route']} | {', '.join(r['tools']) or '-'} | {ok} | "
                     f"{(r.get('judge') or {}).get('score', '-')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true", help="use the deterministic fake LLM (no API calls)")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--min-route-accuracy", type=float, default=None,
                        help="quality gate; default 0.85 live, 0.70 fake (fake mode measures the keyword fallback)")
    parser.add_argument("--only", help="comma-separated item ids")
    args = parser.parse_args()

    if args.fake:
        from unittest import mock

        import tests.conftest  # noqa: F401  (offline settings + fixed env)

        mock.patch("app.tools.knowledge_tools.web_search",
                   lambda q, **_: [{"title": "fake", "url": "https://example.com", "snippet": "EUR 6.2"}]).start()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    items = load_dataset(DATASET)
    if args.only:
        wanted = set(args.only.split(","))
        items = [i for i in items if i["id"] in wanted]
    orchestrator, judge_llm = build_orchestrator(args.fake)

    rows = []
    for item in items:
        start = time.perf_counter()
        result = orchestrator.invoke(item["message"], item["user_id"], session_id=f"eval-{item['id']}")
        latency = int((time.perf_counter() - start) * 1000)
        row = {"id": item["id"], "category": item["category"], "expected_route": item["expected_route"],
               "route": result["route"], "tools": [t["tool"] for t in result["tool_calls"]],
               "confidence": result["confidence"], "escalated": result["escalated"], "latency_ms": latency,
               "response": result["response"], "checks": evaluate(item, result)}
        if judge_llm and not args.no_judge and item["category"] not in ("guardrail",):
            row["judge"] = judge(judge_llm, item, result)
        rows.append(row)
        status = "ok " if row["checks"]["route_ok"] else "BAD"
        print(f"[{status}] {item['id']:<8} route={result['route']:<12} tools={row['tools']} {latency}ms")

    summary = summarise(rows)
    meta = {"timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"), "mode": "fake" if args.fake else "live"}
    out = RESULTS_DIR / f"eval-{meta['timestamp']}-{meta['mode']}"
    out.with_suffix(".json").write_text(json.dumps({"meta": meta, "summary": summary, "rows": rows},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(to_markdown(summary, rows, meta), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"Report: {out.with_suffix('.md')}")

    overall = summary["overall"]
    gate = args.min_route_accuracy if args.min_route_accuracy is not None else (0.70 if args.fake else 0.85)
    if (overall["route_accuracy"] or 0) < gate or (overall["safety_pass"] or 0) < 1.0:
        print("Quality gate FAILED")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
