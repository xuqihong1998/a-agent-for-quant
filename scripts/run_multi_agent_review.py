from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agents.graph import run_sequential_agent_pipeline
from tools.agent_pipeline import render_pipeline_daily_report, render_pipeline_plan, write_pipeline_json


CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports" / "daily"
SETTINGS_PATH = CONFIG_DIR / "settings.yaml"


def _read_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sequential multi-agent daily review pipeline.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Review date, e.g. 2026-04-25.")
    parser.add_argument("--model", default=None, help="Optional model override for research_agent and plan_agent.")
    parser.add_argument(
        "--strict-agent",
        action="store_true",
        help="Fail instead of using rule-based fallback when the LLM stages are unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = _read_settings()
    llm_settings = settings.get("llm", {})
    api_key = llm_settings.get("api_key")
    base_url = llm_settings.get("base_url")
    disable_tracing = bool(llm_settings.get("disable_tracing", False))
    prefer_text_json = bool(llm_settings.get("prefer_text_json", False))
    if api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = str(api_key)

    model = args.model or llm_settings.get("model")
    allow_rule_fallback = not args.strict_agent and bool(llm_settings.get("allow_rule_fallback", True))

    data_output, research_output, risk_output, plan_output = run_sequential_agent_pipeline(
        config_dir=CONFIG_DIR,
        data_dir=DATA_DIR,
        review_date=args.date,
        model=model,
        api_key=str(api_key) if api_key else None,
        base_url=str(base_url) if base_url else None,
        disable_tracing=disable_tracing,
        allow_rule_fallback=allow_rule_fallback,
        prefer_text_json=prefer_text_json,
    )

    report_path = REPORT_DIR / f"{args.date}.md"
    plan_path = REPORT_DIR / f"{args.date}-plan.md"
    json_path = REPORT_DIR / f"{args.date}.pipeline.json"

    render_pipeline_daily_report(data_output, research_output, risk_output, plan_output, report_path)
    render_pipeline_plan(plan_output, plan_path)
    write_pipeline_json(data_output, research_output, risk_output, plan_output, json_path)

    print(f"[DONE] 多 Agent 复盘报告已生成: {report_path}")
    print(f"[DONE] 多 Agent 次日计划已生成: {plan_path}")
    print(f"[DONE] 多 Agent 结构化 JSON 已生成: {json_path}")
    print(f"[INFO] research_agent={research_output.run_mode}")
    print(f"[INFO] plan_agent={plan_output.run_mode}")


if __name__ == "__main__":
    main()
