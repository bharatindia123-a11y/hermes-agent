
import pytest

from agent.archetypes import (
    ARCHETYPE_SCHEMA_FIELDS,
    ARCHETYPES_BY_NAME,
    BUILTIN_ARCHETYPES,
    DEFAULT_ARCHETYPE_NAME,
    REQUIRED_ARCHETYPE_FIELDS,
    SPECIALIST_MAPPINGS_BY_NAME,
    SpecialistMapping,
    get_default_archetype,
    get_tool_restrictions,
    list_archetypes,
    resolve_archetype,
    resolve_archetype_defaults,
    resolve_named_workflow,
    resolve_specialist_defaults,
    resolve_specialist_mapping,
    validate_specialist_mapping,
    validate_specialist_mappings,
)


def test_builtin_archetypes_are_preserved_in_order():
    assert [a.name for a in BUILTIN_ARCHETYPES] == ["generalist", "researcher", "implementer", "verifier"]
    assert DEFAULT_ARCHETYPE_NAME == "generalist"
    assert tuple(ARCHETYPES_BY_NAME) == tuple(a.name for a in BUILTIN_ARCHETYPES)
    assert list_archetypes() is BUILTIN_ARCHETYPES
    assert get_default_archetype() is ARCHETYPES_BY_NAME["generalist"]


def test_required_archetype_fields_are_default_bearing_contract():
    assert REQUIRED_ARCHETYPE_FIELDS == (
        "default_route_category",
        "default_delegation_profile",
        "default_skills",
        "default_required_tools",
        "permission_preset",
        "fallback_policy",
    )
    assert set(REQUIRED_ARCHETYPE_FIELDS).issubset(set(ARCHETYPE_SCHEMA_FIELDS))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "generalist"), ("", "generalist"), ("unknown", "generalist"), ("Implementer", "implementer")],
)
def test_resolve_archetype_defaults_unknown_to_generalist(raw, expected):
    assert resolve_archetype(raw).name == expected


def test_resolve_archetype_defaults_returns_only_required_fields_and_listifies_tuples():
    resolved = resolve_archetype_defaults("implementer")
    assert tuple(resolved) == REQUIRED_ARCHETYPE_FIELDS
    assert isinstance(resolved["default_skills"], list)
    assert isinstance(resolved["default_required_tools"], list)
    assert resolved["default_delegation_profile"] == "implementation"


def test_resolve_archetype_defaults_applies_safe_overrides():
    resolved = resolve_archetype_defaults(
        "generalist",
        overrides={
            "default_skills": [" python ", "", "testing"],
            "default_required_tools": (" read_file ", "terminal"),
            "fallback_policy": " explicit ",
        },
    )
    assert resolved["default_skills"] == ["python", "testing"]
    assert resolved["default_required_tools"] == ["read_file", "terminal"]
    assert resolved["fallback_policy"] == "explicit"
    with pytest.raises(TypeError, match="list or tuple"):
        resolve_archetype_defaults("generalist", overrides={"default_skills": "python"})


def test_validate_specialist_mapping_rejects_collisions_and_unknowns():
    with pytest.raises(ValueError, match="SpecialistMapping instance"):
        validate_specialist_mapping({})
    with pytest.raises(ValueError, match="kind"):
        validate_specialist_mapping(SpecialistMapping("custom", "generalist", kind="archetype"))
    with pytest.raises(ValueError, match="collides with archetype"):
        validate_specialist_mapping(SpecialistMapping("researcher", "generalist"))
    with pytest.raises(ValueError, match="collides with route category"):
        validate_specialist_mapping(SpecialistMapping("deep", "generalist"))
    with pytest.raises(ValueError, match="collides with runtime mode"):
        validate_specialist_mapping(SpecialistMapping("ultrawork", "generalist"))
    with pytest.raises(ValueError, match="Unknown archetype"):
        validate_specialist_mapping(SpecialistMapping("custom", "missing"))
    with pytest.raises(ValueError, match="Unknown default route category"):
        validate_specialist_mapping(SpecialistMapping("custom", "generalist", default_route_category="missing"))
    with pytest.raises(ValueError, match="cannot both block and allow"):
        validate_specialist_mapping(SpecialistMapping("custom", "generalist", blocked_tools=("patch",), allowed_tools=("patch",)))


def test_validate_specialist_mappings_rejects_key_mismatch():
    with pytest.raises(ValueError, match="registry key"):
        validate_specialist_mappings({"alias": SpecialistMapping("real", "generalist")})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("reviewer", "code_reviewer"), ("visual", "multimodal_specialist"), ("vision", "multimodal_specialist"), ("oracle", "consultant")],
)
def test_resolve_specialist_aliases_without_collapsing_archetype_taxonomy(raw, expected):
    mapping = resolve_specialist_mapping(raw)
    assert mapping is SPECIALIST_MAPPINGS_BY_NAME[expected]
    assert mapping.name == expected
    assert mapping.name != mapping.archetype_name


def test_resolve_specialist_unknown_and_defaults():
    assert resolve_specialist_mapping(None) is None
    assert resolve_specialist_mapping("missing") is None
    assert resolve_specialist_defaults("missing") == {}
    assert resolve_specialist_defaults("builder")["default_delegation_profile"] == "implementation"


def test_get_tool_restrictions_merges_read_only_specialist_rules():
    blocked, allowed = get_tool_restrictions("verifier", "consultant")
    assert {"write_file", "patch", "terminal", "execute_code", "delegate_task"}.issubset(blocked)
    assert not (set(allowed) & set(blocked))


def test_multimodal_specialist_is_allowlisted_read_visual_surface():
    blocked, allowed = get_tool_restrictions("researcher", "looker")
    assert blocked == frozenset()
    assert {"read_file", "search_files", "vision_analyze", "browser_vision"}.issubset(set(allowed))


def test_named_workflows_resolve_separately_from_specialists_and_runtime_modes():
    assert resolve_named_workflow("Planner").name == "planner"
    assert resolve_named_workflow("deep-worker").name == "deep_worker"
    assert resolve_named_workflow(None) is None
    assert resolve_named_workflow("missing") is None
    assert resolve_named_workflow("planner").mode == "plan"
    assert resolve_named_workflow("deep_worker").mode == "execute"
