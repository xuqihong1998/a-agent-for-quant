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

from agents.review_agent import run_review_agent
from tools.agent_review import render_agent_daily_report, write_agent_review_json


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
    parser = argparse.ArgumentParser(description="Run the single-agent daily review assistant.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Review date, e.g. 2026-04-24.")
    parser.add_argument("--model", default=None, help="Optional OpenAI model override.")
    parser.add_argument(
        "--strict-agent",
        action="store_true",
        help="Fail instead of using the rule-based fallback when the SDK or model call is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = _read_settings()
    llm_settings = settings.get("llm", {})
    api_key = llm_settings.get("api_key")
    base_url = llm_settings.get("base_url")
    disable_tracing = bool(llm_settings.get("disable_tracing", False))
    if api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = str(api_key)
    model = args.model or llm_settings.get("model")
    allow_rule_fallback = not args.strict_agent and bool(llm_settings.get("allow_rule_fallback", True))
    print(f"[INFO] review_date={args.date}")
    print(f"[INFO] model={model or 'default'}")
    print(f"[INFO] base_url={base_url or 'default'}")
    print(f"[INFO] tracing_disabled={disable_tracing}")
    print(f"[INFO] rule_fallback_enabled={allow_rule_fallback}")
    print(f"[INFO] api_key_present={bool(api_key or os.environ.get('OPENAI_API_KEY'))}")

    review_output, base_results = run_review_agent(
        config_dir=CONFIG_DIR,
        data_dir=DATA_DIR,
        review_date=args.date,
        model=model,
        api_key=str(api_key) if api_key else None,
        base_url=str(base_url) if base_url else None,
        disable_tracing=disable_tracing,
        allow_rule_fallback=allow_rule_fallback,
    )
    print("[INFO] final_review_output:")
    print(json.dumps(review_output.model_dump(), ensure_ascii=False, indent=2))

    report_path = REPORT_DIR / f"{args.date}.md"
    json_path = REPORT_DIR / f"{args.date}.json"

    render_agent_daily_report(base_results, review_output.model_dump(), report_path)
    write_agent_review_json(base_results, review_output.model_dump(), json_path)

    print(f"[DONE] AI 复盘报告已生成: {report_path}")
    print(f"[DONE] 结构化 JSON 已生成: {json_path}")
    print(f"[INFO] 运行模式: {review_output.run_mode}")


if __name__ == "__main__":
    main()
