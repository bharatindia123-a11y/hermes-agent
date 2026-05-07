from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from agent.archetypes import resolve_named_workflow, resolve_specialist_mapping

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = dict[str, Any] | list[Any] | JSONScalar
StructuredSection: TypeAlias = dict[str, JSONValue] | list[JSONValue]

ORCHESTRATION_HINTS_SCHEMA = "hermes/orchestration-hints"
ORCHESTRATION_HINTS_VERSION = "1.0"
NAMED_WORKFLOW_SCHEMA = "hermes/named-workflow"
NAMED_WORKFLOW_VERSION = "1.0"

REQUIRED_TASK_CONTRACT_FIELDS: tuple[str, ...] = (
    "task",
    "expected_outcome",
    "required_skills",
    "required_tools",
    "must_do",
    "must_not_do",
    "context",
)


def _validate_json_compatible_value(value: Any, path: str) -> Any:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} must use string keys")
            _validate_json_compatible_value(nested_value, f"{path}.{key}")
        return value

    if isinstance(value, list):
        for index, nested_value in enumerate(value):
            _validate_json_compatible_value(nested_value, f"{path}[{index}]")
        return value

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, (tuple, set, bytes, bytearray, memoryview, Path)):
        raise ValueError(f"{path} must contain only JSON-compatible values")

    if callable(value):
        raise ValueError(f"{path} must contain only JSON-compatible values")

    raise ValueError(f"{path} must contain only JSON-compatible values")


def _validate_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an ordered list of non-empty trimmed strings")

    normalized_values: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{field_name} must be an ordered list of non-empty trimmed strings")
        trimmed_item = item.strip()
        if not trimmed_item or trimmed_item != item:
            raise ValueError(f"{field_name} must contain only non-empty trimmed strings")
        normalized_values.append(trimmed_item)
    return normalized_values


def _require_trimmed_taxonomy_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    trimmed = value.strip()
    if not trimmed or trimmed != value:
        raise ValueError(f"{field_name} must be a non-empty trimmed string")
    return trimmed


class TaskContract(BaseModel):
    """Canonical structured payload for delegated work."""

    model_config = ConfigDict(extra="forbid")

    task: str
    expected_outcome: str
    required_skills: list[str]
    required_tools: list[str]
    must_do: Any
    must_not_do: Any
    context: Any

    @field_validator("required_skills", "required_tools", mode="before")
    @classmethod
    def _require_ordered_string_lists(cls, value: Any, info: Any) -> list[str]:
        return _validate_string_list(value, str(info.field_name))

    @field_validator("must_do", "must_not_do", "context")
    @classmethod
    def _require_structured_sections(cls, value: Any, info: Any) -> StructuredSection:
        if not isinstance(value, (dict, list)):
            raise ValueError(f"{info.field_name} must remain structured as a dict or list")
        _validate_json_compatible_value(value, str(info.field_name))
        return value


CanonicalTaskContract = TaskContract


class OrchestrationHints(BaseModel):
    """Canonical structured hints payload for work-command orchestration."""

    model_config = ConfigDict(extra="allow", populate_by_name=True, serialize_by_alias=True)

    schema_id: str = Field(alias="schema")
    schema_version_id: str = Field(alias="schema_version")
    command: str
    loop_style: str
    request: str
    bounded_context: dict[str, JSONValue]

    @field_validator("schema_id")
    @classmethod
    def _require_known_schema(cls, value: Any) -> str:
        normalized = str(value or "").strip()
        if normalized != ORCHESTRATION_HINTS_SCHEMA:
            raise ValueError(f"schema must equal {ORCHESTRATION_HINTS_SCHEMA!r}")
        return normalized

    @field_validator("schema_version_id")
    @classmethod
    def _require_known_schema_version(cls, value: Any) -> str:
        normalized = str(value or "").strip()
        if normalized != ORCHESTRATION_HINTS_VERSION:
            raise ValueError(f"schema_version must equal {ORCHESTRATION_HINTS_VERSION!r}")
        return normalized

    @field_validator("command", "loop_style", "request")
    @classmethod
    def _require_non_empty_trimmed_strings(cls, value: Any, info: Any) -> str:
        normalized = str(value or "")
        trimmed = normalized.strip()
        if not trimmed or trimmed != normalized:
            raise ValueError(f"{info.field_name} must be a non-empty trimmed string")
        return trimmed

    @field_validator("bounded_context")
    @classmethod
    def _require_structured_bounded_context(cls, value: Any) -> dict[str, JSONValue]:
        if not isinstance(value, dict):
            raise ValueError("bounded_context must remain structured as a dict")
        _validate_json_compatible_value(value, "bounded_context")
        return value


class NamedWorkflowArtifact(BaseModel):
    """Canonical runtime-visible workflow artifact for planner/deep-worker semantics."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    schema_id: str = Field(alias="schema")
    schema_version_id: str = Field(alias="schema_version")
    workflow_name: str
    mode: str
    objective: str
    plan: list[str]
    acceptance: list[str]
    taxonomy: dict[str, JSONValue]
    execution_task_contract: TaskContract | None = None
    consumption: dict[str, JSONValue] | None = None

    @field_validator("schema_id")
    @classmethod
    def _require_named_workflow_schema(cls, value: Any) -> str:
        normalized = str(value or "").strip()
        if normalized != NAMED_WORKFLOW_SCHEMA:
            raise ValueError(f"schema must equal {NAMED_WORKFLOW_SCHEMA!r}")
        return normalized

    @field_validator("schema_version_id")
    @classmethod
    def _require_named_workflow_schema_version(cls, value: Any) -> str:
        normalized = str(value or "").strip()
        if normalized != NAMED_WORKFLOW_VERSION:
            raise ValueError(f"schema_version must equal {NAMED_WORKFLOW_VERSION!r}")
        return normalized

    @field_validator("workflow_name", "mode", "objective")
    @classmethod
    def _require_named_workflow_strings(cls, value: Any, info: Any) -> str:
        normalized = str(value or "")
        trimmed = normalized.strip()
        if not trimmed or trimmed != normalized:
            raise ValueError(f"{info.field_name} must be a non-empty trimmed string")
        return trimmed

    @model_validator(mode="after")
    def _require_registered_named_workflow(self) -> "NamedWorkflowArtifact":
        workflow = resolve_named_workflow(self.workflow_name)
        if workflow is None:
            raise ValueError(f"Unknown workflow_name: {self.workflow_name}")
        if self.mode != workflow.mode:
            raise ValueError(
                f"mode must equal canonical named workflow mode {workflow.mode!r} for {workflow.name!r}"
            )
        if tuple(self.plan) != tuple(workflow.plan):
            raise ValueError(f"plan must equal canonical named workflow plan for {workflow.name!r}")
        if tuple(self.acceptance) != tuple(workflow.acceptance):
            raise ValueError(f"acceptance must equal canonical named workflow acceptance for {workflow.name!r}")
        for taxonomy_field in ("named_workflow", "workflow"):
            taxonomy_value = self.taxonomy.get(taxonomy_field)
            if taxonomy_value != workflow.name:
                raise ValueError(
                    f"taxonomy.{taxonomy_field} must equal canonical workflow identity {workflow.name!r}"
                )
        return self

    @field_validator("plan", "acceptance", mode="before")
    @classmethod
    def _require_named_workflow_lists(cls, value: Any, info: Any) -> list[str]:
        return _validate_string_list(value, str(info.field_name))

    @field_validator("taxonomy")
    @classmethod
    def _require_named_workflow_taxonomy(cls, value: Any) -> dict[str, JSONValue]:
        if not isinstance(value, dict):
            raise ValueError("taxonomy must remain structured as a dict")
        _validate_json_compatible_value(value, "taxonomy")
        return value

    @field_validator("consumption")
    @classmethod
    def _require_named_workflow_consumption(cls, value: Any) -> dict[str, JSONValue] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("consumption must remain structured as a dict")
        _validate_json_compatible_value(value, "consumption")
        return value


def build_named_workflow_artifact(
    *,
    objective: str,
    specialist: str | None,
    archetype: str,
    route_category: str,
    runtime_mode: str,
    delegation_profile: str,
    task_contract: dict[str, Any] | TaskContract | None = None,
) -> dict[str, Any] | None:
    normalized_objective = str(objective or "").strip()
    if not normalized_objective:
        return None

    validated_task_contract = validate_task_contract(task_contract) if task_contract is not None else None
    normalized_specialist = resolve_specialist_mapping(specialist).name if specialist else None
    workflow = resolve_named_workflow("planner") if normalized_specialist == "planner" else None
    if workflow is None and validated_task_contract is not None:
        workflow = resolve_named_workflow("deep_worker")
    if workflow is None:
        return None

    taxonomy = {
        "named_workflow": workflow.name,
        "workflow": workflow.name,
        "specialist": normalized_specialist or None,
        "archetype": _require_trimmed_taxonomy_string(archetype, "archetype"),
        "route_category": _require_trimmed_taxonomy_string(route_category, "route_category"),
        "runtime_mode": _require_trimmed_taxonomy_string(runtime_mode, "runtime_mode"),
        "delegation_profile": _require_trimmed_taxonomy_string(delegation_profile, "delegation_profile"),
    }

    execution_task_contract = validated_task_contract
    if workflow.default_task_contract is not None and execution_task_contract is None:
        execution_task_contract = TaskContract.model_validate(
            {
                "task": normalized_objective,
                **dict(workflow.default_task_contract),
                "context": {"workflow_source": workflow.name, "objective": normalized_objective},
            }
        )

    artifact = NamedWorkflowArtifact.model_validate(
        {
            "schema": NAMED_WORKFLOW_SCHEMA,
            "schema_version": NAMED_WORKFLOW_VERSION,
            "workflow_name": workflow.name,
            "mode": workflow.mode,
            "objective": normalized_objective,
            "plan": list(workflow.plan),
            "acceptance": list(workflow.acceptance),
            "taxonomy": taxonomy,
            "execution_task_contract": execution_task_contract.model_dump() if execution_task_contract else None,
            "consumption": dict(workflow.consumption) if workflow.consumption is not None else None,
        }
    )
    return artifact.model_dump(by_alias=True)


def validate_task_contract(payload: dict[str, Any] | TaskContract) -> TaskContract:
    if isinstance(payload, TaskContract):
        return payload
    return TaskContract.model_validate(payload)


def validate_orchestration_hints(payload: dict[str, Any] | OrchestrationHints) -> OrchestrationHints:
    if isinstance(payload, OrchestrationHints):
        return payload
    return OrchestrationHints.model_validate(payload)


def validate_named_workflow_artifact(
    payload: dict[str, Any] | NamedWorkflowArtifact,
) -> NamedWorkflowArtifact:
    if isinstance(payload, NamedWorkflowArtifact):
        return payload
    return NamedWorkflowArtifact.model_validate(payload)


def is_task_contract(payload: Any) -> bool:
    try:
        validate_task_contract(payload)
    except (ValidationError, TypeError):
        return False
    return True


__all__ = [
    "CanonicalTaskContract",
    "JSONScalar",
    "JSONValue",
    "NAMED_WORKFLOW_SCHEMA",
    "NAMED_WORKFLOW_VERSION",
    "NamedWorkflowArtifact",
    "ORCHESTRATION_HINTS_SCHEMA",
    "ORCHESTRATION_HINTS_VERSION",
    "OrchestrationHints",
    "REQUIRED_TASK_CONTRACT_FIELDS",
    "StructuredSection",
    "TaskContract",
    "build_named_workflow_artifact",
    "is_task_contract",
    "validate_named_workflow_artifact",
    "validate_orchestration_hints",
    "validate_task_contract",
]
