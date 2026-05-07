
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.task_contracts import (
    NAMED_WORKFLOW_SCHEMA,
    NAMED_WORKFLOW_VERSION,
    ORCHESTRATION_HINTS_SCHEMA,
    ORCHESTRATION_HINTS_VERSION,
    REQUIRED_TASK_CONTRACT_FIELDS,
    NamedWorkflowArtifact,
    OrchestrationHints,
    TaskContract,
    build_named_workflow_artifact,
    is_task_contract,
    validate_named_workflow_artifact,
    validate_orchestration_hints,
    validate_task_contract,
)


def valid_contract(**overrides):
    payload = {
        "task": "Review PR #123",
        "expected_outcome": "evidence-backed review",
        "required_skills": ["github-code-review"],
        "required_tools": ["read_file", "search_files"],
        "must_do": ["inspect diff", {"cite": "files"}],
        "must_not_do": ["merge", "push"],
        "context": {"pr": 123, "ok": True},
    }
    payload.update(overrides)
    return payload


def test_required_task_contract_fields_are_preserved():
    assert REQUIRED_TASK_CONTRACT_FIELDS == (
        "task",
        "expected_outcome",
        "required_skills",
        "required_tools",
        "must_do",
        "must_not_do",
        "context",
    )


def test_validate_task_contract_accepts_valid_payload_and_existing_model():
    contract = validate_task_contract(valid_contract())
    assert isinstance(contract, TaskContract)
    assert validate_task_contract(contract) is contract
    assert contract.task == "Review PR #123"


def test_task_contract_forbids_extra_fields_and_requires_all_fields():
    with pytest.raises(ValidationError):
        validate_task_contract(valid_contract(extra=1))
    for field in REQUIRED_TASK_CONTRACT_FIELDS:
        payload = valid_contract()
        payload.pop(field)
        with pytest.raises(ValidationError):
            validate_task_contract(payload)


@pytest.mark.parametrize("field", ["required_skills", "required_tools"])
@pytest.mark.parametrize("bad", ["python", ["python", " testing "], ["python", ""], ["python", 1]])
def test_task_contract_requires_skill_and_tool_lists_of_trimmed_strings(field, bad):
    with pytest.raises((ValidationError, TypeError, ValueError)):
        validate_task_contract(valid_contract(**{field: bad}))


@pytest.mark.parametrize("field", ["must_do", "must_not_do", "context"])
def test_task_contract_requires_structured_sections(field):
    with pytest.raises(ValidationError):
        validate_task_contract(valid_contract(**{field: "plain prose"}))


def test_task_contract_rejects_non_json_compatible_nested_values_and_non_string_keys():
    with pytest.raises(ValidationError, match="JSON-compatible"):
        validate_task_contract(valid_contract(context={"path": Path("/tmp")}))
    with pytest.raises(ValidationError, match="string keys"):
        validate_task_contract(valid_contract(context={1: "x"}))


def test_task_contract_allows_json_scalar_leaves_inside_structured_sections():
    contract = validate_task_contract(valid_contract(must_do=["step", 1, True, None]))
    assert contract.must_do == ["step", 1, True, None]


def test_is_task_contract_distinguishes_structured_contracts_from_prompt_strings():
    assert is_task_contract(valid_contract()) is True
    assert is_task_contract("TASK: do it") is False
    assert is_task_contract(valid_contract(required_tools="terminal")) is False


def test_orchestration_hints_accept_aliases_and_extra_but_enforce_schema_version():
    hints = validate_orchestration_hints({
        "schema": ORCHESTRATION_HINTS_SCHEMA,
        "schema_version": ORCHESTRATION_HINTS_VERSION,
        "command": "/task",
        "loop_style": "ultrawork",
        "request": "ship it",
        "bounded_context": {"wave": 1},
        "extra": "allowed",
    })
    assert isinstance(hints, OrchestrationHints)
    assert hints.model_dump(by_alias=True)["schema"] == ORCHESTRATION_HINTS_SCHEMA
    with pytest.raises(ValidationError):
        validate_orchestration_hints({**hints.model_dump(by_alias=True), "schema": "wrong"})
    with pytest.raises(ValidationError):
        validate_orchestration_hints({**hints.model_dump(by_alias=True), "bounded_context": []})


def valid_artifact(**overrides):
    payload = {
        "schema": NAMED_WORKFLOW_SCHEMA,
        "schema_version": NAMED_WORKFLOW_VERSION,
        "workflow_name": "planner",
        "mode": "plan",
        "objective": "make a plan",
        "plan": [
            "capture objective, constraints, and success criteria before execution",
            "decompose work into ordered executable steps with dependencies",
            "emit a structured handoff contract that a deep worker can execute",
        ],
        "acceptance": [
            "structured workflow artifact is present",
            "execution handoff remains machine-readable",
        ],
        "taxonomy": {"named_workflow": "planner", "workflow": "planner", "specialist": "planner"},
        "consumption": {"consumes": "execution_task_contract"},
        "execution_task_contract": valid_contract(task="make a plan"),
    }
    payload.update(overrides)
    return payload


def test_validate_named_workflow_artifact_accepts_valid_payload_and_rejects_mismatches():
    artifact = validate_named_workflow_artifact(valid_artifact())
    assert isinstance(artifact, NamedWorkflowArtifact)
    with pytest.raises(ValidationError):
        validate_named_workflow_artifact(valid_artifact(workflow_name="unknown"))
    with pytest.raises(ValidationError):
        validate_named_workflow_artifact(valid_artifact(mode="execute"))
    with pytest.raises(ValidationError):
        validate_named_workflow_artifact(valid_artifact(taxonomy={"named_workflow": "deep_worker", "workflow": "planner"}))
    with pytest.raises(ValidationError):
        validate_named_workflow_artifact(valid_artifact(plan=["non-canonical plan"]))
    with pytest.raises(ValidationError):
        validate_named_workflow_artifact(valid_artifact(acceptance=["non-canonical acceptance"]))
    with pytest.raises(ValidationError):
        validate_named_workflow_artifact(valid_artifact(extra=1))


def test_build_named_workflow_artifact_handles_planner_and_deep_worker_paths():
    assert build_named_workflow_artifact(
        objective="   ",
        specialist=None,
        archetype="generalist",
        route_category="deep",
        runtime_mode="default",
        delegation_profile="general",
    ) is None
    planner = build_named_workflow_artifact(
        objective="  create plan  ",
        specialist="planner",
        archetype="generalist",
        route_category="deep",
        runtime_mode="default",
        delegation_profile="general",
    )
    assert planner is not None
    assert planner["workflow_name"] == "planner"
    assert planner["mode"] == "plan"
    assert planner["objective"] == "create plan"
    assert planner["execution_task_contract"]["task"] == "create plan"
    deep = build_named_workflow_artifact(
        objective="execute",
        specialist="builder",
        archetype="implementer",
        route_category="deep",
        runtime_mode="default",
        delegation_profile="implementation",
        task_contract=valid_contract(task="execute"),
    )
    assert deep is not None
    assert deep["workflow_name"] == "deep_worker"
    assert deep["mode"] == "execute"
    assert deep["execution_task_contract"]["task"] == "execute"
    assert build_named_workflow_artifact(
        objective="execute",
        specialist="builder",
        archetype="implementer",
        route_category="deep",
        runtime_mode="default",
        delegation_profile="implementation",
        task_contract=None,
    ) is None


def test_named_workflow_taxonomy_preserves_distinct_runtime_fields():
    artifact = build_named_workflow_artifact(
        objective="create plan",
        specialist="Planner",
        archetype="generalist",
        route_category="deep",
        runtime_mode="default",
        delegation_profile="general",
    )
    taxonomy = artifact["taxonomy"]
    assert {"named_workflow", "workflow", "specialist", "archetype", "route_category", "runtime_mode", "delegation_profile"}.issubset(taxonomy)
    assert taxonomy["named_workflow"] == "planner"
    assert taxonomy["specialist"] == "planner"
    assert taxonomy["archetype"] != taxonomy["runtime_mode"]
    with pytest.raises(ValueError, match="archetype"):
        build_named_workflow_artifact(
            objective="create plan",
            specialist="planner",
            archetype=" generalist ",
            route_category="deep",
            runtime_mode="default",
            delegation_profile="general",
        )
