import json
from types import SimpleNamespace

import pytest

from agent.task_contracts import build_named_workflow_artifact

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _apply_specialist_tool_restrictions,
    _normalize_task_contract,
    _resolve_named_agent_config,
    _resolve_route_category_entry,
    _resolve_runtime_mode_entry,
    _resolve_task_delegation_profile_details,
    _resolve_task_inputs,
)


def _valid_contract(required_tools=None):
    return {
        "task": "Review the design",
        "expected_outcome": "Evidence-backed review",
        "required_skills": ["github-code-review"],
        "required_tools": required_tools or ["read_file"],
        "must_do": ["cite exact files"],
        "must_not_do": ["merge"],
        "context": {"pr": 123},
    }


def test_schema_exposes_wave2_delegate_fields_top_level_and_per_task():
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    task_props = props["tasks"]["items"]["properties"]
    expected = {
        "agent",
        "subagent_type",
        "category",
        "archetype",
        "specialist",
        "route_category",
        "delegation_profile",
        "runtime_mode",
        "skills",
        "task_contract",
        "named_workflow",
    }
    assert expected.issubset(props)
    assert expected.issubset(task_props)
    assert "max_iterations" not in props


def test_named_agent_resolution_canonicalizes_aliases_and_preserves_identity():
    cfg = _resolve_named_agent_config("MultiModal_Looker", {})
    assert cfg["agent"] == "multimodal-looker"
    assert cfg["named_agent"] == "multimodal-looker"
    assert cfg["specialist"] == "looker"
    assert _resolve_named_agent_config("oracle", {})["archetype"] == "verifier"
    assert _resolve_named_agent_config("missing", {}) is None


def test_route_and_runtime_entries_reject_unknowns_and_allow_config_overrides():
    cfg = {
        "route_categories": {"visual": {"summary": "custom visual", "intensity": "custom"}},
        "runtime_modes": {"execution_supervisor": {"description": "custom runtime", "operating_posture": "custom posture"}},
    }
    route = _resolve_route_category_entry("visual", config=cfg)
    assert route["name"] == "visual"
    assert route["summary"] == "custom visual"
    runtime = _resolve_runtime_mode_entry("execution-supervisor", config=cfg)
    assert runtime["name"] == "execution_supervisor"
    assert runtime["description"] == "custom runtime"
    with pytest.raises(ValueError, match="Unknown route_category"):
        _resolve_route_category_entry("missing", config=cfg)
    with pytest.raises(ValueError, match="Unknown runtime_mode"):
        _resolve_runtime_mode_entry("missing", config=cfg)


def test_delegation_profile_marks_legacy_category_compatibility_only():
    legacy = _resolve_task_delegation_profile_details(category="visual-engineering")
    assert legacy["name"] == "visual-engineering"
    assert legacy["source"] == "category"
    assert legacy["compatibility_only"] is True
    explicit = _resolve_task_delegation_profile_details(delegation_profile="verification", category="quick")
    assert explicit["name"] == "verification"
    assert explicit["source"] == "delegation_profile"
    assert explicit["compatibility_only"] is False


def test_task_inputs_keep_literal_route_profile_runtime_distinct_and_preserve_contract():
    resolved = _resolve_task_inputs(
        {
            "goal": "review",
            "agent": "oracle",
            "category": "visual-engineering",
            "delegation_profile": "verification",
            "runtime_mode": "execution-supervisor",
            "task_contract": _valid_contract(),
        },
        full_config={},
        delegation_config={},
    )
    assert resolved["named_agent"] == "oracle"
    assert resolved["category"] == "visual-engineering"
    assert resolved["route_category"] == "visual"
    assert resolved["delegation_profile"] == "verification"
    assert resolved["runtime_mode"] == "execution_supervisor"
    assert resolved["category"] != resolved["runtime_mode"]
    assert resolved["task_contract"]["task"] == "Review the design"
    assert "OMO / Named-Agent Runtime Contract" in resolved["overlay_prompt"]



def test_task_inputs_attach_planner_named_workflow_artifact_and_activation_block():
    goal = "Plan the candidate upgrade proof packet"

    resolved = _resolve_task_inputs(
        {"goal": goal, "specialist": "planner"},
        full_config={},
        delegation_config={},
    )

    expected = build_named_workflow_artifact(
        objective=goal,
        specialist="planner",
        archetype=resolved["archetype"],
        route_category=resolved["route_category"],
        runtime_mode=resolved["runtime_mode"],
        delegation_profile=resolved["delegation_profile"],
        task_contract=None,
    )
    assert resolved["named_workflow"] == expected
    assert resolved["named_workflow"]["schema"] == "hermes/named-workflow"
    assert resolved["named_workflow"]["workflow_name"] == "planner"
    assert resolved["named_workflow"]["mode"] == "plan"
    assert resolved["named_workflow"]["taxonomy"]["named_workflow"] == "planner"
    assert resolved["task_contract"] == resolved["named_workflow"]["execution_task_contract"]
    assert resolved["task_contract"]["task"] == goal
    assert "Named workflow activated: planner" in resolved["overlay_prompt"]
    assert "<named-workflow>" in resolved["overlay_prompt"]
    assert "execution_task_contract" in resolved["overlay_prompt"]


def test_task_inputs_attach_deep_worker_named_workflow_only_when_task_contract_present():
    contract = _valid_contract()

    deep = _resolve_task_inputs(
        {"goal": "execute handoff", "specialist": "builder", "task_contract": contract},
        full_config={},
        delegation_config={},
    )
    assert deep["named_workflow"]["workflow_name"] == "deep_worker"
    assert deep["named_workflow"]["mode"] == "execute"
    assert deep["named_workflow"]["execution_task_contract"] == deep["task_contract"]
    assert "<named-workflow>" in deep["overlay_prompt"]

    generic = _resolve_task_inputs(
        {"goal": "generic child work", "specialist": "builder"},
        full_config={},
        delegation_config={},
    )
    assert generic["named_workflow"] is None
    assert "<named-workflow>" not in generic["overlay_prompt"]


def test_task_inputs_derive_task_contract_from_explicit_named_workflow():
    goal = "Run planner handoff"
    artifact = build_named_workflow_artifact(
        objective=goal,
        specialist="planner",
        archetype="generalist",
        route_category="deep",
        runtime_mode="default",
        delegation_profile="general",
        task_contract=None,
    )

    resolved = _resolve_task_inputs(
        {"goal": goal, "named_workflow": artifact},
        full_config={},
        delegation_config={},
    )

    assert resolved["named_workflow"] == artifact
    assert resolved["task_contract"] == artifact["execution_task_contract"]

def test_task_route_category_wins_over_literal_category_mapping():
    resolved = _resolve_task_inputs(
        {"goal": "look", "category": "visual-engineering", "route_category": "quick"},
        full_config={},
        delegation_config={},
    )
    assert resolved["category"] == "visual-engineering"
    assert resolved["route_category"] == "quick"


def test_unknown_named_agent_and_category_are_rejected_instead_of_silent_fallback():
    with pytest.raises(ValueError, match="Unknown named agent"):
        _resolve_task_inputs({"goal": "review", "agent": "missing"}, full_config={}, delegation_config={})
    with pytest.raises(ValueError, match="Unknown category"):
        _resolve_task_inputs({"goal": "review", "category": "not-a-real-category"}, full_config={}, delegation_config={})


@pytest.mark.parametrize("enabled_tools", [["read_file"], []])
def test_task_contract_required_tools_must_be_enabled_by_delegation_profile(enabled_tools):
    cfg = {"profiles": {"review": {"enabled_tools": enabled_tools}}}
    with pytest.raises(ValueError, match="task_contract.required_tools"):
        _resolve_task_inputs(
            {
                "goal": "review",
                "delegation_profile": "review",
                "task_contract": _valid_contract(required_tools=["patch"]),
            },
            full_config={},
            delegation_config=cfg,
        )


def test_normalize_task_contract_validates_and_reports_goal_context():
    assert _normalize_task_contract(_valid_contract(), goal="review")["required_tools"] == ["read_file"]
    with pytest.raises(ValueError, match="delegated task 'review'"):
        _normalize_task_contract({"task": "missing required fields"}, goal="review")


def test_read_only_required_mutating_tools_raise_value_error():
    with pytest.raises(ValueError, match="read-only policy"):
        _resolve_task_inputs(
            {"goal": "review", "agent": "oracle", "task_contract": _valid_contract(required_tools=["patch"])},
            full_config={},
            delegation_config={},
        )


def test_apply_specialist_tool_restrictions_filters_child_tools_and_valid_names():
    child = SimpleNamespace(
        tools=[
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "patch"}},
            {"type": "function", "function": {"name": "delegate_task"}},
            {"type": "function", "function": {"name": "search_files"}},
        ],
        valid_tool_names={"read_file", "patch", "delegate_task", "search_files"},
    )
    _apply_specialist_tool_restrictions(
        child,
        delegate_resolution={"named_agent": "oracle", "archetype": "verifier", "specialist": "consultant"},
    )
    assert child.valid_tool_names == {"read_file", "search_files"}
    assert "patch" not in json.dumps(child.tools)
    assert "delegate_task" not in json.dumps(child.tools)


def test_mcp_inheritance_cannot_restore_blocked_tools_after_filtering():
    child = SimpleNamespace(
        tools=[
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "mcp_write_file"}},
            {"type": "function", "function": {"name": "write_file"}},
        ],
        valid_tool_names={"read_file", "mcp_write_file", "write_file"},
    )
    _apply_specialist_tool_restrictions(
        child,
        delegate_resolution={
            "named_agent": "multimodal-looker",
            "archetype": "researcher",
            "specialist": "looker",
            "allowed_tools": ["read_file", "search_files", "vision_analyze"],
        },
    )
    assert child.valid_tool_names == {"read_file"}
