from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from tools.agent_review import build_review_context
from tools.risk_checks import classify_plan_action


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_STANCES = {"继续持有", "谨慎持有", "重点观察", "普通观察", "减仓观察", "暂不关注"}


def _normalize_stance(value: Any) -> str:
    mapping = {
        "持有": "继续持有",
        "继续持仓": "继续持有",
        "观察": "普通观察",
    }
    normalized = mapping.get(str(value), str(value))
    return normalized if normalized in VALID_STANCES else "普通观察"


class ReviewedSymbol(BaseModel):
    symbol: str = Field(description="Six-digit A-share symbol.")
    name: str = Field(description="Readable security name.")
    category: Literal["position", "watchlist"] = Field(description="Whether the symbol comes from holdings or watchlist.")
    stance: Literal["继续持有", "谨慎持有", "重点观察", "普通观察", "减仓观察", "暂不关注"]
    headline: str = Field(description="Short analyst-style headline.")
    thesis: str = Field(description="One concise research conclusion grounded in the provided signals.")
    key_signals: list[str] = Field(default_factory=list, description="Two to four concise signals that support the conclusion.")
    risk_flags: list[str] = Field(default_factory=list, description="Risk reminders that matter for the next trading day.")
    next_day_plan: str = Field(description="Concrete plan for the next trading day.")
    focus_price: str = Field(description="Levels or moving-average area to watch.")
    invalidation: str = Field(description="What would invalidate the current view.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1.")


class DailyResearchReview(BaseModel):
    review_date: str
    run_mode: Literal["openai_agent", "rule_fallback"]
    market_summary: str
    portfolio_summary: str
    next_day_overview: str
    symbols: list[ReviewedSymbol]
    warnings: list[str] = Field(default_factory=list)


def _import_openai_agents_sdk() -> Any:
    original_package = sys.modules.get("agents")
    removed_paths: list[tuple[int, str]] = []

    for index in range(len(sys.path) - 1, -1, -1):
        candidate = sys.path[index]
        resolved = Path(candidate or ".").resolve()
        if resolved == PROJECT_ROOT:
            removed_paths.append((index, candidate))
            sys.path.pop(index)

    if original_package is not None:
        package_file = getattr(original_package, "__file__", None)
        package_path = Path(package_file).resolve() if package_file else None
        package_paths = [Path(path).resolve() for path in getattr(original_package, "__path__", [])]
        is_local_package = bool(package_path and PROJECT_ROOT in package_path.parents) or any(
            PROJECT_ROOT in path.parents or path == PROJECT_ROOT / "agents" for path in package_paths
        )
        if is_local_package:
            del sys.modules["agents"]

    try:
        return importlib.import_module("agents")
    finally:
        for index, candidate in sorted(removed_paths, key=lambda item: item[0]):
            sys.path.insert(index, candidate)
        if original_package is not None and "agents" not in sys.modules:
            sys.modules["agents"] = original_package


def _build_signal_list(item: dict[str, Any]) -> list[str]:
    signals = [
        f"收盘 {item['close']:.2f}，MA5/MA10/MA20 为 {item['ma5']:.2f}/{item['ma10']:.2f}/{item['ma20']:.2f}",
        f"量能状态为 {item['volume_status']}，5日量比 {item['amount_ratio_5d']:.2f}",
    ]
    if item.get("pct_chg") is not None:
        signals.append(f"当日涨跌幅 {item['pct_chg'] * 100:.2f}%")
    if item.get("is_position"):
        signals.append(f"持仓状态 {item.get('position_action', '持有')}，浮盈亏 {item.get('pnl_pct', 0.0) * 100:.2f}%")
    return signals[:4]


def _fallback_review_for_item(item: dict[str, Any]) -> ReviewedSymbol:
    category: Literal["position", "watchlist"] = "position" if item.get("is_position") else "watchlist"
    stance = _normalize_stance(item.get("position_action") if item.get("is_position") else classify_plan_action(item))
    risk_flags = list(item.get("risk_notes", [])) if item.get("is_position") else []
    if not risk_flags:
        risk_flags = ["暂无新增风险提示，继续跟踪量价和均线关系。"]

    return ReviewedSymbol(
        symbol=item["symbol"],
        name=item["name"],
        category=category,
        stance=stance,
        headline=item["comment"],
        thesis=f"{item['comment']}。当前核心观察点是 {item['next_focus']}。",
        key_signals=_build_signal_list(item),
        risk_flags=risk_flags,
        next_day_plan=item["action_hint"],
        focus_price=item["next_focus"],
        invalidation="若跌破关键均线并出现放量走弱，则需要下调评级。",
        confidence=0.55,
    )


def _normalize_review_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    normalized = dict(payload)
    symbols = normalized.get("symbols")
    if not isinstance(symbols, list):
        return normalized

    normalized_symbols: list[Any] = []
    for item in symbols:
        if isinstance(item, dict):
            symbol_item = dict(item)
            if "stance" in symbol_item:
                symbol_item["stance"] = _normalize_stance(symbol_item["stance"])
            normalized_symbols.append(symbol_item)
        else:
            normalized_symbols.append(item)
    normalized["symbols"] = normalized_symbols
    return normalized


def _stringify_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [f"{key}: {_stringify_value(item)}" for key, item in value.items()]
        return "; ".join(parts)
    if isinstance(value, list):
        return "; ".join(_stringify_value(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _coerce_review_payload(payload: Any, base_results: list[dict[str, Any]]) -> Any:
    normalized = _normalize_review_payload(payload)
    if not isinstance(normalized, dict):
        return normalized

    summary_map = {str(item.get("symbol", "")).zfill(6): item for item in base_results if item.get("symbol")}
    coerced = dict(normalized)
    coerced["review_date"] = _stringify_value(coerced.get("review_date"))
    coerced["run_mode"] = "openai_agent"
    coerced["market_summary"] = _stringify_value(coerced.get("market_summary"))
    coerced["portfolio_summary"] = _stringify_value(coerced.get("portfolio_summary"))
    coerced["next_day_overview"] = _stringify_value(coerced.get("next_day_overview"))
    coerced["warnings"] = [_stringify_value(item) for item in (coerced.get("warnings") or [])]

    symbols = coerced.get("symbols")
    if not isinstance(symbols, list):
        coerced["symbols"] = []
        return coerced

    coerced_symbols: list[dict[str, Any]] = []
    for item in symbols:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).zfill(6)
        source_item = summary_map.get(symbol, {})
        coerced_item = dict(item)
        coerced_item["symbol"] = symbol
        coerced_item["name"] = _stringify_value(coerced_item.get("name") or source_item.get("name") or symbol)
        coerced_item["category"] = "position" if source_item.get("is_position") else "watchlist"
        coerced_item["stance"] = _normalize_stance(
            coerced_item.get("stance")
            or source_item.get("position_action")
            or classify_plan_action(source_item)
            if source_item
            else coerced_item.get("stance")
        )
        coerced_item["headline"] = _stringify_value(coerced_item.get("headline"))
        coerced_item["thesis"] = _stringify_value(coerced_item.get("thesis"))
        coerced_item["key_signals"] = [_stringify_value(v) for v in (coerced_item.get("key_signals") or [])]
        coerced_item["risk_flags"] = [_stringify_value(v) for v in (coerced_item.get("risk_flags") or [])]
        coerced_item["next_day_plan"] = _stringify_value(coerced_item.get("next_day_plan"))
        coerced_item["focus_price"] = _stringify_value(coerced_item.get("focus_price"))
        coerced_item["invalidation"] = _stringify_value(coerced_item.get("invalidation"))
        try:
            coerced_item["confidence"] = float(coerced_item.get("confidence", 0.5))
        except (TypeError, ValueError):
            coerced_item["confidence"] = 0.5
        coerced_symbols.append(coerced_item)

    coerced["symbols"] = coerced_symbols
    return coerced


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def _is_json_mode_unsupported_error(exc: Exception) -> bool:
    return "json mode is not supported" in str(exc).lower()


def build_rule_fallback_review(
    base_results: list[dict[str, Any]],
    review_date: str,
    warning: str | None = None,
) -> DailyResearchReview:
    positions = [item for item in base_results if item.get("is_position")]
    watch_only = [item for item in base_results if not item.get("is_position")]
    warnings = [warning] if warning else []

    if not positions:
        portfolio_summary = "当前无持仓，重点关注观察池中量价结构改善的标的。"
    else:
        weak_positions = [item for item in positions if item.get("position_action") == "减仓观察"]
        cautious_positions = [item for item in positions if item.get("position_action") == "谨慎持有"]
        if weak_positions:
            portfolio_summary = "持仓中已有转弱标的，次日优先处理风险暴露。"
        elif cautious_positions:
            portfolio_summary = "持仓整体仍可跟踪，但已有标的靠近关键均线，需观察承接。"
        else:
            portfolio_summary = "持仓结构总体稳定，重点看强势股能否延续并守住短期均线。"

    if watch_only:
        next_day_overview = "先按原有规则观察候选标的是否在 MA5/MA10 区域获得承接，再决定是否升级关注。"
    else:
        next_day_overview = "观察池暂时为空，次日以持仓风险控制和趋势跟踪为主。"

    return DailyResearchReview(
        review_date=review_date,
        run_mode="rule_fallback",
        market_summary="当前为规则兜底模式，市场总结基于既有趋势、量能和风险规则生成。",
        portfolio_summary=portfolio_summary,
        next_day_overview=next_day_overview,
        symbols=[_fallback_review_for_item(item) for item in base_results],
        warnings=warnings,
    )


def create_review_agent(
    context_payload: dict[str, Any],
    model: str | None = None,
    structured_output: bool = True,
) -> Any:
    sdk = _import_openai_agents_sdk()
    Agent = getattr(sdk, "Agent")
    function_tool = getattr(sdk, "function_tool")

    symbol_map = {
        str(item["symbol"]).zfill(6): item for item in context_payload.get("summary_items", []) if item.get("symbol")
    }

    @function_tool
    def load_review_context() -> dict[str, Any]:
        """Load today's structured review context, including watchlist, positions, risk rules, and per-symbol summaries."""

        return context_payload

    @function_tool
    def get_symbol_context(symbol: str) -> dict[str, Any]:
        """Fetch the detailed structured summary for one symbol by six-digit code."""

        normalized = str(symbol).zfill(6)
        return symbol_map[normalized]

    instructions = """
你是 A 股盘后复盘研究员 review_agent。
你的任务只有一个：基于工具提供的结构化 summary，输出像研究员写的复盘结论和次日观察计划。

严格要求：
1. 先调用 load_review_context 获取全量上下文；如有必要再调用 get_symbol_context。
2. 只能依据工具返回的结构化数据做结论，不要编造新闻、公告、盘口、资金流、板块消息。
3. 不要下达真实交易执行指令，不要涉及自动下单或交易接口操作。
4. 每个标的都必须输出，symbol 不能遗漏，stance 必须与风险约束兼容。
5. 结论要像研究员，但保持克制，重点解释趋势、均线、量能、持仓风险和次日观察点。
6. headline、thesis、next_day_plan 和 invalidation 必须具体、可执行、可审计。
7. 如果存在持仓，优先覆盖持仓风险，再讨论观察池。
"""

    if not structured_output:
        instructions += """

8. Final answer must be exactly one JSON object.
9. Do not wrap the JSON in markdown or code fences.
10. The top-level keys must be review_date, run_mode, market_summary, portfolio_summary, next_day_overview, symbols, warnings.
11. run_mode must be openai_agent.
12. Each item in symbols must include symbol, name, category, stance, headline, thesis, key_signals, risk_flags, next_day_plan, focus_price, invalidation, confidence.
"""

    agent_kwargs: dict[str, Any] = {
        "name": "review_agent",
        "instructions": instructions,
        "tools": [load_review_context, get_symbol_context],
        "model": model,
    }
    if structured_output:
        agent_kwargs["output_type"] = DailyResearchReview
    return Agent(**agent_kwargs)


def run_review_agent(
    config_dir: Path,
    data_dir: Path,
    review_date: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    disable_tracing: bool = False,
    allow_rule_fallback: bool = True,
) -> tuple[DailyResearchReview, list[dict[str, Any]]]:
    resolved_review_date = review_date or datetime.now().strftime("%Y-%m-%d")
    context_payload, base_results = build_review_context(config_dir, data_dir, review_date=resolved_review_date)
    print(f"[AGENT] loaded {len(base_results)} review targets for {resolved_review_date}")

    try:
        sdk = _import_openai_agents_sdk()
        print("[AGENT] imported openai-agents sdk")
        if disable_tracing and hasattr(sdk, "set_tracing_disabled"):
            sdk.set_tracing_disabled(True)
            print("[AGENT] tracing disabled")
        Runner = getattr(sdk, "Runner")
        RunConfig = getattr(sdk, "RunConfig", None)
        MultiProvider = getattr(sdk, "MultiProvider", None)
        run_config = None
        if (api_key or base_url) and RunConfig is not None and MultiProvider is not None:
            provider_kwargs: dict[str, Any] = {
                "openai_prefix_mode": "model_id",
                "unknown_prefix_mode": "model_id",
            }
            if api_key:
                provider_kwargs["openai_api_key"] = api_key
            if base_url:
                provider_kwargs["openai_base_url"] = base_url
            provider = MultiProvider(**provider_kwargs)
            run_config = RunConfig(model_provider=provider)
            safe_kwargs = {key: ("***" if key == "openai_api_key" else value) for key, value in provider_kwargs.items()}
            print(f"[AGENT] configured MultiProvider: {safe_kwargs}")
        prefer_text_json = True
        agent = create_review_agent(context_payload, model=model, structured_output=not prefer_text_json)
        print(f"[AGENT] agent created with model={model or 'default'}, structured_output={not prefer_text_json}")
        prompt = (
            f"请为 {resolved_review_date} 生成一份盘后复盘研究结论。"
            "先读取工具中的结构化上下文，再输出结构化结果。"
        )
        print("[AGENT] prompt:")
        print(prompt)
        try:
            if run_config is not None:
                result = Runner.run_sync(agent, prompt, run_config=run_config)
            else:
                result = Runner.run_sync(agent, prompt)
        except Exception as exc:
            if prefer_text_json or not _is_json_mode_unsupported_error(exc):
                raise
            print("[AGENT] structured output unsupported, retrying with plain JSON text mode")
            text_agent = create_review_agent(context_payload, model=model, structured_output=False)
            if run_config is not None:
                result = Runner.run_sync(text_agent, prompt, run_config=run_config)
            else:
                result = Runner.run_sync(text_agent, prompt)
        print(f"[AGENT] run completed, result_type={type(result).__name__}")
        raw_output = getattr(result, "final_output", None)
        print(f"[AGENT] raw_final_output_type={type(raw_output).__name__}")
        try:
            if hasattr(raw_output, "model_dump"):
                print("[AGENT] raw_final_output:")
                print(json.dumps(raw_output.model_dump(), ensure_ascii=False, indent=2))
            else:
                print("[AGENT] raw_final_output:")
                print(json.dumps(raw_output, ensure_ascii=False, indent=2, default=str))
        except Exception as log_exc:
            print(f"[AGENT] failed to print raw_final_output: {log_exc}")
        if isinstance(raw_output, str):
            final_output = _coerce_review_payload(_extract_json_object(raw_output), base_results)
        else:
            final_output = _coerce_review_payload(raw_output, base_results)
        if final_output is not raw_output:
            print("[AGENT] normalized_final_output:")
            print(json.dumps(final_output, ensure_ascii=False, indent=2, default=str))
        if isinstance(final_output, DailyResearchReview):
            return final_output.model_copy(update={"run_mode": "openai_agent"}), base_results
        return DailyResearchReview.model_validate(
            {
                **DailyResearchReview.model_validate(final_output).model_dump(),
                "run_mode": "openai_agent",
            }
        ), base_results
    except Exception as exc:
        print(f"[AGENT] exception: {type(exc).__name__}: {exc}")
        if not allow_rule_fallback:
            raise
        print("[AGENT] switching to rule_fallback")
        return build_rule_fallback_review(base_results, resolved_review_date, warning=str(exc)), base_results
