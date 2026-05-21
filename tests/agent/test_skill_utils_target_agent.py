"""Tests for target-agent scoped skill helpers."""

from agent.skill_utils import (
    canonical_agent_key,
    runtime_target_agent,
    skill_matches_target_agent,
    skill_target_agents,
    skill_target_mismatch_error,
)


def test_canonical_agent_key_normalizes_case_space_and_underscore():
    assert canonical_agent_key("Code_Agent") == "code-agent"
    assert canonical_agent_key("CODE agent") == "code-agent"
    assert canonical_agent_key("code---agent") == "code-agent"


def test_skill_target_agents_accepts_target_agent_and_agent_alias_scalar_or_list():
    fm = {"target_agent": ["Builder_Agent", "Reviewer"], "agent": "QA Agent"}
    assert skill_target_agents(fm) == {"builder-agent", "reviewer", "qa-agent"}


def test_public_skills_match_without_runtime_target():
    assert skill_matches_target_agent({}, None) is True


def test_targeted_skills_require_matching_explicit_agent_identity():
    fm = {"target_agent": ["builder", "reviewer"]}
    assert skill_matches_target_agent(fm, target_agent="Builder") is True
    assert skill_matches_target_agent(fm, target_agent="researcher") is False
    assert skill_matches_target_agent(fm, target_agent=None) is False


def test_runtime_target_agent_ignores_profile_category_route_and_delegation_profile():
    resolution = {
        "category": "builder",
        "literal_category": "builder",
        "route_category": "builder",
        "delegation_profile": "builder",
        "runtime_mode": "builder",
        "archetype": "builder",
        "specialist": "builder",
        "profile": "builder",
    }
    assert runtime_target_agent(resolution) == ""
    assert skill_matches_target_agent({"target_agent": "builder"}, delegate_resolution=resolution) is False


def test_runtime_target_agent_uses_only_agent_or_named_agent():
    assert runtime_target_agent({"agent": "Builder_Agent"}) == "builder-agent"
    assert runtime_target_agent({"named_agent": "Reviewer Agent"}) == "reviewer-agent"


def test_target_mismatch_error_is_explicit_and_hides_body_callers_can_refuse():
    err = skill_target_mismatch_error("deploy-skill", {"target_agent": "deployer"}, target_agent="reviewer")
    assert "deploy-skill" in err
    assert "deployer" in err
    assert "reviewer" in err
