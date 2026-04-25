from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from agents.data_agent import DataAgentOutput, run_data_agent
from agents.plan_agent import PlanAgentOutput, run_plan_agent
from agents.research_agent import ResearchAgentOutput, run_research_agent
from agents.risk_agent import RiskAgentOutput, run_risk_agent


def _log(message: str) -> None:
    print(f"[GRAPH {datetime.now().strftime('%H:%M:%S')}] {message}")


class MultiAgentGraphState(TypedDict, total=False):
    review_date: str
    config_dir: str
    data_dir: str
    model: str | None
    api_key: str | None
    base_url: str | None
    disable_tracing: bool
    allow_rule_fallback: bool
    prefer_text_json: bool
    human_review_enabled: bool
    context_loaded: bool
    data_output: dict[str, Any]
    research_output: dict[str, Any]
    risk_output: dict[str, Any]
    plan_output: dict[str, Any]
    human_review_required: bool
    human_review_payload: dict[str, Any]
    human_review_decision: dict[str, Any]
    final_status: str


def run_sequential_agent_pipeline(
    config_dir: Path,
    data_dir: Path,
    review_date: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    disable_tracing: bool = False,
    allow_rule_fallback: bool = True,
    prefer_text_json: bool = False,
    human_review_enabled: bool = True,
) -> tuple[DataAgentOutput, ResearchAgentOutput, RiskAgentOutput, PlanAgentOutput]:
    _log(
        "run_sequential_agent_pipeline started "
        f"review_date={review_date} model={model or 'default'} "
        f"prefer_text_json={prefer_text_json} allow_rule_fallback={allow_rule_fallback}"
    )
    data_output = run_data_agent(config_dir=config_dir, data_dir=data_dir, review_date=review_date)
    _log(f"data_agent completed symbols={len(data_output.symbols)} warnings={len(data_output.warnings)}")
    research_output = run_research_agent(
        data_output=data_output,
        model=model,
        api_key=api_key,
        base_url=base_url,
        disable_tracing=disable_tracing,
        allow_rule_fallback=allow_rule_fallback,
        prefer_text_json=prefer_text_json,
    )
    _log(f"research_agent completed symbols={len(research_output.symbols)} run_mode={research_output.run_mode}")
    risk_output = run_risk_agent(
        data_output=data_output,
        research_output=research_output,
        config_dir=config_dir,
    )
    _log(f"risk_agent completed symbols={len(risk_output.symbols)} warnings={len(risk_output.warnings)}")
    plan_output = run_plan_agent(
        data_output=data_output,
        research_output=research_output,
        risk_output=risk_output,
        model=model,
        api_key=api_key,
        base_url=base_url,
        disable_tracing=disable_tracing,
        allow_rule_fallback=allow_rule_fallback,
        prefer_text_json=prefer_text_json,
    )
    _log(f"plan_agent completed symbols={len(plan_output.symbols)} run_mode={plan_output.run_mode}")
    return data_output, research_output, risk_output, plan_output


def _normalize_initial_state(
    config_dir: Path,
    data_dir: Path,
    review_date: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    disable_tracing: bool,
    allow_rule_fallback: bool,
    prefer_text_json: bool,
    human_review_enabled: bool,
) -> MultiAgentGraphState:
    return {
        "review_date": review_date,
        "config_dir": str(config_dir),
        "data_dir": str(data_dir),
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "disable_tracing": disable_tracing,
        "allow_rule_fallback": allow_rule_fallback,
        "prefer_text_json": prefer_text_json,
        "context_loaded": False,
        "human_review_enabled": human_review_enabled,
        "human_review_required": human_review_enabled,
        "final_status": "running",
    }


def _load_context_node(state: MultiAgentGraphState) -> dict[str, Any]:
    _log(
        "node load_context start "
        f"review_date={state['review_date']} config_dir={state['config_dir']} data_dir={state['data_dir']}"
    )
    return {
        "context_loaded": True,
        "review_date": state["review_date"],
        "config_dir": state["config_dir"],
        "data_dir": state["data_dir"],
    }


def _data_agent_node(state: MultiAgentGraphState) -> dict[str, Any]:
    _log("node data_agent start")
    data_output = run_data_agent(
        config_dir=Path(state["config_dir"]),
        data_dir=Path(state["data_dir"]),
        review_date=state["review_date"],
    )
    _log(f"node data_agent done symbols={len(data_output.symbols)} warnings={len(data_output.warnings)}")
    return {"data_output": data_output.model_dump()}


def _research_agent_node(state: MultiAgentGraphState) -> dict[str, Any]:
    _log("node research_agent start")
    data_output = DataAgentOutput.model_validate(state["data_output"])
    _log(f"node research_agent input symbols={len(data_output.symbols)}")
    research_output = run_research_agent(
        data_output=data_output,
        model=state.get("model"),
        api_key=state.get("api_key"),
        base_url=state.get("base_url"),
        disable_tracing=bool(state.get("disable_tracing", False)),
        allow_rule_fallback=bool(state.get("allow_rule_fallback", True)),
        prefer_text_json=bool(state.get("prefer_text_json", False)),
    )
    _log(
        f"node research_agent done symbols={len(research_output.symbols)} "
        f"run_mode={research_output.run_mode} warnings={len(research_output.warnings)}"
    )
    return {"research_output": research_output.model_dump()}


def _risk_agent_node(state: MultiAgentGraphState) -> dict[str, Any]:
    _log("node risk_agent start")
    data_output = DataAgentOutput.model_validate(state["data_output"])
    research_output = ResearchAgentOutput.model_validate(state["research_output"])
    _log(
        f"node risk_agent input data_symbols={len(data_output.symbols)} "
        f"research_symbols={len(research_output.symbols)}"
    )
    risk_output = run_risk_agent(
        data_output=data_output,
        research_output=research_output,
        config_dir=Path(state["config_dir"]),
    )
    _log(f"node risk_agent done symbols={len(risk_output.symbols)} warnings={len(risk_output.warnings)}")
    return {"risk_output": risk_output.model_dump()}


def _plan_agent_node(state: MultiAgentGraphState) -> dict[str, Any]:
    _log("node plan_agent start")
    data_output = DataAgentOutput.model_validate(state["data_output"])
    research_output = ResearchAgentOutput.model_validate(state["research_output"])
    risk_output = RiskAgentOutput.model_validate(state["risk_output"])
    _log(
        f"node plan_agent input data_symbols={len(data_output.symbols)} "
        f"research_symbols={len(research_output.symbols)} risk_symbols={len(risk_output.symbols)}"
    )
    plan_output = run_plan_agent(
        data_output=data_output,
        research_output=research_output,
        risk_output=risk_output,
        model=state.get("model"),
        api_key=state.get("api_key"),
        base_url=state.get("base_url"),
        disable_tracing=bool(state.get("disable_tracing", False)),
        allow_rule_fallback=bool(state.get("allow_rule_fallback", True)),
        prefer_text_json=bool(state.get("prefer_text_json", False)),
    )
    _log(
        f"node plan_agent done symbols={len(plan_output.symbols)} "
        f"run_mode={plan_output.run_mode} warnings={len(plan_output.warnings)}"
    )
    if not bool(state.get("human_review_enabled", True)):
        _log("node plan_agent auto-approving because human_review_enabled=False")
        return {
            "plan_output": plan_output.model_dump(),
            "final_status": "auto_approved",
            "human_review_required": False,
        }
    return {"plan_output": plan_output.model_dump()}


def _route_after_plan(state: MultiAgentGraphState) -> str:
    if bool(state.get("human_review_enabled", True)):
        _log("route_after_plan -> human_review")
        return "human_review"
    _log("route_after_plan -> END")
    return "END"


def _build_human_review_payload(state: MultiAgentGraphState) -> dict[str, Any]:
    plan_output = PlanAgentOutput.model_validate(state["plan_output"])
    risk_output = RiskAgentOutput.model_validate(state["risk_output"])
    top_actions = [
        {
            "symbol": item.symbol,
            "name": item.name,
            "final_action": item.final_action,
            "risk_level": item.risk_level,
            "focus_price": item.focus_price,
        }
        for item in plan_output.symbols
    ]
    _log(f"build human_review payload actions={len(top_actions)}")
    return {
        "review_date": plan_output.review_date,
        "step": "human_review",
        "instruction": "请人工确认次日计划，可批准、驳回，或编辑 next_day_overview。",
        "portfolio_summary": risk_output.portfolio_risk_summary,
        "next_day_overview": plan_output.next_day_overview,
        "actions": top_actions,
        "expected_response_schema": {
            "approved": "bool, 必填",
            "notes": "str, 可选",
            "edited_next_day_overview": "str, 可选",
        },
    }


def _apply_human_review_decision(
    plan_output: PlanAgentOutput,
    decision: dict[str, Any] | None,
) -> tuple[PlanAgentOutput, dict[str, Any], str]:
    normalized_decision = dict(decision or {})
    approved = bool(normalized_decision.get("approved", False))
    notes = str(normalized_decision.get("notes") or "").strip()
    edited_overview = str(normalized_decision.get("edited_next_day_overview") or "").strip()

    updates: dict[str, Any] = {"approved": approved}
    if notes:
        updates["notes"] = notes

    if edited_overview:
        plan_output = plan_output.model_copy(update={"next_day_overview": edited_overview})
        updates["edited_next_day_overview"] = edited_overview

    final_status = "approved" if approved else "rejected"
    _log(
        "apply human_review decision "
        f"approved={approved} notes_present={bool(notes)} edited_overview={bool(edited_overview)}"
    )
    return plan_output, updates, final_status


def _human_review_node(state: MultiAgentGraphState) -> dict[str, Any]:
    from langgraph.types import interrupt

    _log("node human_review start")
    plan_output = PlanAgentOutput.model_validate(state["plan_output"])
    payload = _build_human_review_payload(state)
    decision = interrupt(payload)
    _log(f"node human_review resumed decision_keys={sorted(dict(decision or {}).keys())}")
    updated_plan_output, normalized_decision, final_status = _apply_human_review_decision(plan_output, decision)
    _log(f"node human_review done final_status={final_status}")
    return {
        "human_review_payload": payload,
        "human_review_decision": normalized_decision,
        "plan_output": updated_plan_output.model_dump(),
        "final_status": final_status,
    }


def build_multi_agent_state_graph(checkpointer: Any | None = None) -> Any:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    _log(f"build_multi_agent_state_graph checkpointer={'custom' if checkpointer is not None else 'InMemorySaver'}")
    workflow = StateGraph(MultiAgentGraphState)
    workflow.add_node("load_context", _load_context_node)
    workflow.add_node("data_agent", _data_agent_node)
    workflow.add_node("research_agent", _research_agent_node)
    workflow.add_node("risk_agent", _risk_agent_node)
    workflow.add_node("plan_agent", _plan_agent_node)
    workflow.add_node("human_review", _human_review_node)

    workflow.add_edge(START, "load_context")
    workflow.add_edge("load_context", "data_agent")
    workflow.add_edge("data_agent", "research_agent")
    workflow.add_edge("research_agent", "risk_agent")
    workflow.add_edge("risk_agent", "plan_agent")
    workflow.add_conditional_edges(
        "plan_agent",
        _route_after_plan,
        {
            "human_review": "human_review",
            "END": END,
        },
    )
    workflow.add_edge("human_review", END)

    return workflow.compile(checkpointer=checkpointer or InMemorySaver())


def invoke_langgraph_agent_pipeline(
    *,
    config_dir: Path,
    data_dir: Path,
    review_date: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    disable_tracing: bool = False,
    allow_rule_fallback: bool = True,
    prefer_text_json: bool = False,
    human_review_enabled: bool = True,
    graph: Any | None = None,
    thread_id: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    compiled_graph = graph or build_multi_agent_state_graph()
    resolved_review_date = review_date or datetime.now().strftime("%Y-%m-%d")
    resolved_thread_id = thread_id or f"review-{resolved_review_date}"
    _log(
        "invoke_langgraph_agent_pipeline "
        f"review_date={resolved_review_date} thread_id={resolved_thread_id} "
        f"model={model or 'default'} prefer_text_json={prefer_text_json}"
    )
    initial_state = _normalize_initial_state(
        config_dir=config_dir,
        data_dir=data_dir,
        review_date=resolved_review_date,
        model=model,
        api_key=api_key,
        base_url=base_url,
        disable_tracing=disable_tracing,
        allow_rule_fallback=allow_rule_fallback,
        prefer_text_json=prefer_text_json,
        human_review_enabled=human_review_enabled,
    )
    config = {"configurable": {"thread_id": resolved_thread_id}}
    result = compiled_graph.invoke(initial_state, config=config)
    _log(f"invoke_langgraph_agent_pipeline returned keys={sorted(result.keys())}")
    return compiled_graph, result


def resume_langgraph_agent_pipeline(
    *,
    graph: Any,
    thread_id: str,
    review_decision: dict[str, Any],
) -> dict[str, Any]:
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}
    _log(
        "resume_langgraph_agent_pipeline "
        f"thread_id={thread_id} decision_keys={sorted(dict(review_decision or {}).keys())}"
    )
    result = graph.invoke(Command(resume=review_decision), config=config)
    _log(f"resume_langgraph_agent_pipeline returned keys={sorted(result.keys())}")
    return result
