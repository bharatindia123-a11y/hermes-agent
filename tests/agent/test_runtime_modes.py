
import pytest

from agent.runtime_modes import (
    BUILTIN_RUNTIME_MODES,
    DEFAULT_RUNTIME_MODE_NAME,
    RUNTIME_MODES_BY_NAME,
    RuntimeMode,
    get_default_runtime_mode,
    list_runtime_modes,
    resolve_runtime_mode,
    validate_runtime_mode,
    validate_runtime_modes,
)


def test_builtin_runtime_modes_are_preserved_in_order():
    assert [mode.name for mode in BUILTIN_RUNTIME_MODES] == [
        "default",
        "ultrawork",
        "ralph",
        "interview_planning",
        "execution_supervisor",
    ]
    assert DEFAULT_RUNTIME_MODE_NAME == "default"
    assert tuple(RUNTIME_MODES_BY_NAME) == tuple(mode.name for mode in BUILTIN_RUNTIME_MODES)


def test_default_and_list_helpers_return_canonical_objects():
    assert list_runtime_modes() is BUILTIN_RUNTIME_MODES
    assert get_default_runtime_mode() is RUNTIME_MODES_BY_NAME["default"]


def test_validate_runtime_mode_rejects_bad_shape_and_kind():
    with pytest.raises(ValueError, match="RuntimeMode instance"):
        validate_runtime_mode({})
    with pytest.raises(ValueError, match="kind"):
        validate_runtime_mode(RuntimeMode("custom", "desc", "posture", kind="archetype"))


@pytest.mark.parametrize("field", ["name", "description", "operating_posture"])
def test_validate_runtime_mode_rejects_empty_or_whitespace_padded_strings(field):
    kwargs = {"name": "custom", "description": "desc", "operating_posture": "posture"}
    kwargs[field] = " x "
    with pytest.raises(ValueError):
        validate_runtime_mode(RuntimeMode(**kwargs))
    kwargs[field] = ""
    with pytest.raises(ValueError):
        validate_runtime_mode(RuntimeMode(**kwargs))


@pytest.mark.parametrize("cap", [0, -1, "10", True, False])
def test_validate_runtime_mode_rejects_non_positive_non_int_or_bool_iteration_cap(cap):
    with pytest.raises(ValueError, match="positive integer"):
        validate_runtime_mode(RuntimeMode("custom", "desc", "posture", iteration_cap=cap))


def test_validate_runtime_modes_rejects_duplicates_and_missing_default():
    modes = (
        RuntimeMode("custom", "desc", "posture"),
        RuntimeMode("custom", "desc", "posture"),
    )
    with pytest.raises(ValueError, match="duplicate runtime mode"):
        validate_runtime_modes(modes, default_name="custom")
    with pytest.raises(ValueError, match="default runtime mode"):
        validate_runtime_modes((RuntimeMode("custom", "desc", "posture"),), default_name="default")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "default"),
        ("", "default"),
        ("   ", "default"),
        ("unknown", "default"),
        ("ULTRAWORK", "ultrawork"),
            ("ultra-work", "default"),
        ("Interview Planning", "interview_planning"),
        ("execution-supervisor", "execution_supervisor"),
    ],
)
def test_resolve_runtime_mode_normalizes_known_values_and_defaults_unknown(raw, expected):
    assert resolve_runtime_mode(raw).name == expected


def test_runtime_mode_taxonomy_does_not_collapse_into_routes_or_archetypes():
    from agent.archetypes import ARCHETYPES_BY_NAME
    from agent.route_categories import BUILTIN_ROUTE_CATEGORIES

    assert "default" not in ARCHETYPES_BY_NAME
    assert "ultrawork" not in BUILTIN_ROUTE_CATEGORIES
    assert {mode.kind for mode in BUILTIN_RUNTIME_MODES} == {"runtime_mode"}
