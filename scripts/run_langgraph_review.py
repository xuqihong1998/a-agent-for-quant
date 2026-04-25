from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agents.graph import invoke_langgraph_agent_pipeline, resume_langgraph_agent_pipeline
from tools.agent_pipeline import render_pipeline_daily_report, render_pipeline_plan, write_pipeline_json


CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports" / "daily"
SETTINGS_PATH = CONFIG_DIR / "settings.yaml"


def _log(message: str) -> None:
    print(f"[RUN] {message}")


def _read_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LangGraph-based multi-agent review workflow.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Review date, e.g. 2026-04-25.")
    parser.add_argument("--thread-id", default=None, help="Optional LangGraph thread id.")
    parser.add_argument("--model", default=None, help="Optional model override for research_agent and plan_agent.")
    parser.add_argument("--resume-json", default=None, help='Optional human review response JSON, e.g. {"approved": true, "notes": "..."}')
    parser.add_argument(
        "--human-review",
        action="store_true",
        help="Enable the human_review interrupt stage. Defaults to disabled for direct file output.",
    )
    parser.add_argument(
        "--strict-agent",
        action="store_true",
        help="Fail instead of using rule-based fallback when the LLM stages are unavailable.",
    )
    return parser.parse_args()


def _render_outputs(state: dict[str, object], review_date: str) -> None:
    from agents.data_agent import DataAgentOutput
    from agents.plan_agent import PlanAgentOutput
    from agents.research_agent import ResearchAgentOutput
    from agents.risk_agent import RiskAgentOutput

    _log(f"_render_outputs start state_keys={sorted(state.keys())}")
    data_output = DataAgentOutput.model_validate(state["data_output"])
    research_output = ResearchAgentOutput.model_validate(state["research_output"])
    risk_output = RiskAgentOutput.model_validate(state["risk_output"])
    plan_output = PlanAgentOutput.model_validate(state["plan_output"])
    _log(
        "_render_outputs validated models "
        f"data={len(data_output.symbols)} research={len(research_output.symbols)} "
        f"risk={len(risk_output.symbols)} plan={len(plan_output.symbols)}"
    )

    report_path = REPORT_DIR / f"{review_date}.md"
    plan_path = REPORT_DIR / f"{review_date}-plan.md"
    json_path = REPORT_DIR / f"{review_date}.pipeline.json"
    _log(f"_render_outputs targets report={report_path} plan={plan_path} json={json_path}")

    render_pipeline_daily_report(data_output, research_output, risk_output, plan_output, report_path)
    render_pipeline_plan(plan_output, plan_path)
    write_pipeline_json(data_output, research_output, risk_output, plan_output, json_path)

    print(f"[DONE] LangGraph 复盘报告已生成: {report_path}")
    print(f"[DONE] LangGraph 次日计划已生成: {plan_path}")
    print(f"[DONE] LangGraph 结构化 JSON 已生成: {json_path}")
    print(f"[INFO] final_status={state.get('final_status', 'unknown')}")


def main() -> None:
    _log("run_langgraph_review main start")
    args = parse_args()
    settings = _read_settings()
    llm_settings = settings.get("llm", {})
    api_key = llm_settings.get("api_key")
    base_url = llm_settings.get("base_url")
    disable_tracing = bool(llm_settings.get("disable_tracing", False))
    prefer_text_json = bool(llm_settings.get("prefer_text_json", False))
    human_review_enabled = bool(llm_settings.get("human_review_enabled", False) or args.human_review)
    if api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = str(api_key)

    model = args.model or llm_settings.get("model")
    allow_rule_fallback = not args.strict_agent and bool(llm_settings.get("allow_rule_fallback", True))
    thread_id = args.thread_id or f"review-{args.date}"
    _log(
        f"config review_date={args.date} thread_id={thread_id} model={model or 'default'} "
        f"prefer_text_json={prefer_text_json} allow_rule_fallback={allow_rule_fallback} "
        f"human_review_enabled={human_review_enabled} "
        f"base_url={'set' if base_url else 'default'} api_key={'set' if (api_key or os.environ.get('OPENAI_API_KEY')) else 'missing'}"
    )

    try:
        graph, result = invoke_langgraph_agent_pipeline(
            config_dir=CONFIG_DIR,
            data_dir=DATA_DIR,
            review_date=args.date,
            model=model,
            api_key=str(api_key) if api_key else None,
            base_url=str(base_url) if base_url else None,
            disable_tracing=disable_tracing,
            allow_rule_fallback=allow_rule_fallback,
            prefer_text_json=prefer_text_json,
            human_review_enabled=human_review_enabled,
            thread_id=thread_id,
        )
    except Exception as exc:
        _log(f"invoke_langgraph_agent_pipeline failed {type(exc).__name__}: {exc}")
        raise

    _log(f"invoke result keys={sorted(result.keys())}")

    if "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        payloads = [getattr(item, "value", item) for item in interrupts]
        print("[HUMAN_REVIEW] 工作流已暂停，等待人工审核。")
        print(json.dumps(payloads, ensure_ascii=False, indent=2))
        _log(f"human_review interrupt_count={len(payloads)} resume_json_present={bool(args.resume_json)}")
        if not args.resume_json:
            return
        try:
            review_decision = json.loads(args.resume_json)
            _log(f"parsed resume_json keys={sorted(review_decision.keys())}")
            resumed = resume_langgraph_agent_pipeline(
                graph=graph,
                thread_id=thread_id,
                review_decision=review_decision,
            )
        except Exception as exc:
            _log(f"resume_langgraph_agent_pipeline failed {type(exc).__name__}: {exc}")
            raise
        _log(f"resume result keys={sorted(resumed.keys())}")
        _render_outputs(resumed, args.date)
        return

    _render_outputs(result, args.date)


if __name__ == "__main__":
    main()
